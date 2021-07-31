import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Type, Union

from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import huber_loss
from ray.rllib.utils.typing import TensorType
from ray.rllib.agents.sac.sac_torch_policy import _get_dist_class, SACTorchPolicy, \
    ComputeTDErrorMixin, TargetNetworkMixin
import DEFAULT_CONFIG
from utils import compute_policy_entropy

from ray.rllib.utils.schedules import ConstantSchedule
from schedules.stable_policy_entropy_exp_schedule import StablePolicyEntropyExpSchedule
from schedules.piecewise_schedule import PiecewiseSchedule


torch, nn = try_import_torch()
F = nn.functional

logger = logging.getLogger(__name__)

def actor_critic_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for the Soft Actor Critic.
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[TorchDistributionWrapper]: The action distr. class.
        train_batch (SampleBatch): The training data.
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Should be True only for debugging purposes (e.g. test cases)!
    deterministic = policy.config["_deterministic_loss"]

    model_out_t, _ = model({
        "obs": train_batch[SampleBatch.CUR_OBS],
        "is_training": True,
    }, [], None)

    model_out_tp1, _ = model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": True,
    }, [], None)

    target_model_out_tp1, _ = policy.target_model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": True,
    }, [], None)

    alpha = torch.exp(model.log_alpha)
    alpha = torch.clamp(alpha, min=0, max=10000)   # clip alpha

    # Discrete case.
    if model.discrete:
        # Get all action probs directly from pi and form their logp.
        log_pis_t = F.log_softmax(model.get_policy_output(model_out_t), dim=-1)
        policy_t = torch.exp(log_pis_t)
        log_pis_tp1 = F.log_softmax(model.get_policy_output(model_out_tp1), -1)
        policy_tp1 = torch.exp(log_pis_tp1)
        # Q-values.
        q_t = model.get_q_values(model_out_t)
        # Target Q-values.
        q_tp1 = policy.target_model.get_q_values(target_model_out_tp1)
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(model_out_t)
            twin_q_tp1 = policy.target_model.get_twin_q_values(
                target_model_out_tp1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)
        q_tp1 -= alpha * log_pis_tp1

        # Actually selected Q-values (from the actions batch).
        one_hot = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(),
            num_classes=q_t.size()[-1])
        q_t_selected = torch.sum(q_t * one_hot, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        q_tp1_best_masked = \
            (1.0 - train_batch[SampleBatch.DONES].float()) * \
            q_tp1_best
    # Continuous actions case.
    else:
        # Sample single actions from distribution.
        action_dist_class = _get_dist_class(policy.config, policy.action_space)
        action_dist_t = action_dist_class(
            model.get_policy_output(model_out_t), policy.model)
        policy_t = action_dist_t.sample() if not deterministic else \
            action_dist_t.deterministic_sample()
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
        action_dist_tp1 = action_dist_class(
            model.get_policy_output(model_out_tp1), policy.model)
        policy_tp1 = action_dist_tp1.sample() if not deterministic else \
            action_dist_tp1.deterministic_sample()
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)

        # Q-values for the actually selected actions.
        q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS])

        # Q-values for current policy in given current state.
        q_t_det_policy = model.get_q_values(model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy = model.get_twin_q_values(
                model_out_t, policy_t)
            q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

        # Target q network evaluation.
        q_tp1 = policy.target_model.get_q_values(target_model_out_tp1,
                                                 policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1 = policy.target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1)
            # Take min over both twin-NNs.
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        q_t_selected = torch.squeeze(q_t, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)
        q_tp1 -= alpha * log_pis_tp1

        q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * \
            q_tp1_best

    # compute RHS of bellman equation
    q_t_selected_target = (
        train_batch[SampleBatch.REWARDS] +
        (policy.config["gamma"]**policy.config["n_step"]) * q_tp1_best_masked
    ).detach()

    # Compute the TD-error (potentially clipped).
    base_td_error = torch.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    critic_loss = [
        torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(base_td_error))
    ]
    if policy.config["twin_q"]:
        critic_loss.append(
            torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error)))

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    if model.discrete:
        weighted_log_alpha_loss = policy_t.detach() * (
            -model.log_alpha * (log_pis_t + model.target_entropy).detach())
        # Sum up weighted terms and mean over all batch items.
        alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
        # Actor loss.
        actor_loss = torch.mean(
            torch.sum(
                torch.mul(
                    # NOTE: No stop_grad around policy output here
                    # (compare with q_t_det_policy for continuous case).
                    policy_t,
                    alpha.detach() * log_pis_t - q_t.detach()),
                dim=-1))
    else:
        alpha_loss = -torch.mean(model.log_alpha *
                                 (log_pis_t + model.target_entropy).detach())
        # Note: Do not detach q_t_det_policy here b/c is depends partly
        # on the policy vars (policy sample pushed through Q-net).
        # However, we must make sure `actor_loss` is not used to update
        # the Q-net(s)' variables.
        actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)

    # Init some model params
    if policy.global_timestep == 32:
        model.EMAvg = model.target_entropy.detach().cpu().numpy()[0]
        model.EMStd = 0

    # Save for stats function.
    policy.q_t = q_t
    policy.policy_t = policy_t
    policy.log_pis_t = log_pis_t

    # Store td-error in model, such that for multi-GPU, we do not override
    # them during the parallel loss phase. TD-error tensor in final stats
    # can then be concatenated and retrieved for each individual batch item.
    model.td_error = td_error

    policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.alpha_loss = alpha_loss
    policy.log_alpha_value = model.log_alpha
    policy.alpha_value = alpha
    policy.target_entropy = model.target_entropy
    policy.EMAvg = model.EMAvg
    policy.EMStd = model.EMStd
    policy.MSE_error = torch.mean(base_td_error**2)

    # Return all loss terms corresponding to our optimizers.
    return tuple([policy.actor_loss] + policy.critic_loss +
                 [policy.alpha_loss])


def stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Stats function for SAC. Returns a dict with important loss stats.
    Args:
        policy (Policy): The Policy to generate stats for.
        train_batch (SampleBatch): The SampleBatch (already) used for training.
    Returns:
        Dict[str, TensorType]: The stats dict.
    """
    # td_error = torch.cat(
    #     [
    #         getattr(t, "td_error", torch.tensor([0.0]))
    #         for t in policy.model_gpu_towers
    #     ],
    #     dim=0)
    return {
        # "td_error": td_error,
        # "mean_td_error": torch.mean(td_error),
        "td_error": torch.mean(policy.td_error),
        "actor_loss": torch.mean(policy.actor_loss),
        "critic_loss": torch.mean(torch.stack(policy.critic_loss)),
        "alpha_loss": torch.mean(policy.alpha_loss),
        "alpha_value": torch.mean(policy.alpha_value),
        "log_alpha_value": torch.mean(policy.log_alpha_value),
        "target_entropy": torch.tensor(policy.target_entropy[0].clone().detach(), dtype=torch.float32),
        "policy_t": torch.mean(policy.policy_t),
        "mean_q": torch.mean(policy.q_t),
        "max_q": torch.max(policy.q_t),
        "min_q": torch.min(policy.q_t),
        "policy_entropy": compute_policy_entropy(policy.log_pis_t, policy.policy_t),
        "exponential_moving_average": torch.tensor(policy.EMAvg).to('cuda' if torch.cuda.is_available() else 'cpu'),
        "exponential_moving_std": torch.tensor(policy.EMStd).to('cuda' if torch.cuda.is_available() else 'cpu'),
    }

# piecewise target entropy schedule
@DeveloperAPI
class TargetEntropyPiecewiseSchedule:
    """ Mixin that adds a target entropy schedule. """
    @DeveloperAPI
    def __init__(self, target_entropy, target_entropy_schedule):
        self.target_entropy = target_entropy
        if target_entropy_schedule is None:
            self.target_entropy_schedule = ConstantSchedule(target_entropy, framework=None)
        else:
            self.target_entropy_schedule = PiecewiseSchedule(
                target_entropy_schedule, outside_value=target_entropy_schedule[-1][-1], framework=None
            )

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(TargetEntropyPiecewiseSchedule, self).on_global_var_update(global_vars)
        self.model.target_entropy = torch.tensor(data=[self.target_entropy_schedule.value(global_vars["timestep"])],
                                           dtype=torch.float32, requires_grad=False).to(
                                            'cuda' if torch.cuda.is_available() else 'cpu')


# stable policy entropy exp window schedule
@DeveloperAPI
class TargetEntropyExpSchedule:
    """ Mixin that adds a target entropy schedule based on policy entropy. (exponential window) """
    @DeveloperAPI
    def __init__(self, target_entropy, target_entropy_discount):
        self.target_entropy = target_entropy
        if target_entropy_discount is None:
            self.target_entropy_schedule = ConstantSchedule(target_entropy, framework=None)
        else:
            self.target_entropy_schedule = StablePolicyEntropyExpSchedule(
                target_entropy, target_entropy_discount, framework=None,
                init_train_step=DEFAULT_CONFIG.DEFAULT_CONFIG["init_train_step"],
                total_conditioned_num=DEFAULT_CONFIG.DEFAULT_CONFIG["total_conditioned_num"],
                force_drop_steps=DEFAULT_CONFIG.DEFAULT_CONFIG["force_drop_steps"],
                _lambda=DEFAULT_CONFIG.DEFAULT_CONFIG["lambda"],
                avg_threshold=DEFAULT_CONFIG.DEFAULT_CONFIG["avg_threshold"],
                std_threshold=DEFAULT_CONFIG.DEFAULT_CONFIG["std_threshold"],
            )

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(TargetEntropyExpSchedule, self).on_global_var_update(global_vars)
        self.model.target_entropy = torch.tensor(data=[self.target_entropy_schedule.value(
                                                        (compute_policy_entropy(self.log_pis_t, self.policy_t), global_vars["timestep"])
                                                        )],
                                                dtype=torch.float32, requires_grad=False).to(
                                                'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.EMAvg = self.target_entropy_schedule.EMAvg
        self.model.EMStd = np.sqrt(self.target_entropy_schedule.EMVar)


def setup_late_mixins(policy, obs_space, action_space, config):
    policy.target_model = policy.target_model.to(policy.device)
    policy.model.log_alpha = policy.model.log_alpha.to(policy.device)
    policy.model.target_entropy = policy.model.target_entropy.to(policy.device)
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)
    if DEFAULT_CONFIG.DEFAULT_CONFIG["scheduling"] == "stable_policy_exp":
        TargetEntropyExpSchedule.__init__(policy, policy.model.target_entropy.detach().cpu().numpy()[0],
                                          config["target_entropy_stable_policy_discount"])
    elif DEFAULT_CONFIG.DEFAULT_CONFIG["scheduling"] == "piecewise":
        TargetEntropyPiecewiseSchedule.__init__(policy, policy.model.target_entropy.detach().cpu().numpy()[0],
                                                config["target_entropy_schedule"])


# set different mixins for different scheduling.
if DEFAULT_CONFIG.DEFAULT_CONFIG["scheduling"] == "stable_policy_exp":
    mixins = [TargetEntropyExpSchedule, TargetNetworkMixin, ComputeTDErrorMixin]
elif DEFAULT_CONFIG.DEFAULT_CONFIG["scheduling"] == "piecewise":
    mixins = [TargetEntropyPiecewiseSchedule, TargetNetworkMixin, ComputeTDErrorMixin]
else:
    mixins = [TargetNetworkMixin, ComputeTDErrorMixin]


MySACTorchPolicy = SACTorchPolicy.with_updates(
    name="MySACTorchPolicy",
    loss_fn=actor_critic_loss,
    get_default_config=lambda: DEFAULT_CONFIG.DEFAULT_CONFIG,
    stats_fn=stats,
    before_loss_init=setup_late_mixins,
    mixins=mixins,
)

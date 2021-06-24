import DEFAULT_CONFIG
from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer
from policy import MySACTorchPolicy
from ray.rllib.agents.sac.sac_tf_policy import SACTFPolicy
from ray.rllib.agents.sac.sac import validate_config

def get_policy_class(config):
    if config["framework"] == "torch":
        return MySACTorchPolicy
    else:
        return SACTFPolicy


MySACTrainer = GenericOffPolicyTrainer.with_updates(
    name=DEFAULT_CONFIG.DEFAULT_CONFIG["trainer_name"],
    default_config=DEFAULT_CONFIG.DEFAULT_CONFIG,
    default_policy=MySACTorchPolicy,
    get_policy_class=get_policy_class,
    validate_config=validate_config,
)

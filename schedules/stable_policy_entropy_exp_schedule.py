from ray.rllib.utils.annotations import override
from ray.rllib.utils.schedules.schedule import Schedule
import numpy as np


class StablePolicyEntropyExpSchedule(Schedule):
    """ Target entropy schedule based on policy entropy (exponential window). """
    def __init__(self,
                 initTargetEntropy,
                 targetEntropyDiscount,
                 framework,
                 init_train_step=0,
                 total_conditioned_num=2000,
                 force_drop_steps=180000,
                 _lambda=0.999,
                 avg_threshold=0.01,
                 std_threshold=0.03):
        super().__init__(framework=framework)

        self.targetEntropy = initTargetEntropy
        self.targetEntropyDiscount = targetEntropyDiscount
        self.init_train_step = init_train_step

        self.EMAvg = initTargetEntropy
        self.EMVar = 0
        self._lambda = _lambda

        self.avg_threshold = avg_threshold
        self.std_threshold = std_threshold

        self.total_conditioned_num = total_conditioned_num  # total number of conditioned steps
        self.conditioned_count = 0

        self.force_drop_steps = force_drop_steps # force to drop target entropy after these steps
        self.step_count = 0


    @override(Schedule)
    def _value(self, info):
        current_policy_entropy = info[0]
        current_policy_entropy_num = current_policy_entropy.cpu().detach().numpy().item()
        timestep = info[1]

        # if the training just began, return the first one
        if timestep <= self.init_train_step:
            return self.targetEntropy
        # if the target entropy approaches 0, return target entropy
        if self.targetEntropy < 1e-2:
            return self.targetEntropy
        # if step_count reach force_drop_steps, drop target entropy
        if self.step_count >= self.force_drop_steps:
            self.step_count = 0
            self.targetEntropy *= self.targetEntropyDiscount

        sigma = current_policy_entropy_num - self.EMAvg
        self.EMAvg += (1-self._lambda) * sigma
        self.EMVar = self._lambda * (self.EMVar + (1-self._lambda) * sigma**2)

        if timestep % 1000 == 0:
            print("mean:", self.EMAvg)
            print("std:", np.sqrt(self.EMVar))

        # if the mean of exp window policy entropy is different from the target entropy more 0.01,
        # or the std is larger than 0.03, don't drop the target entropy, otherwise drop.
        if not(self.targetEntropy-self.avg_threshold < self.EMAvg < self.targetEntropy+self.avg_threshold) \
                or np.sqrt(self.EMVar) > self.std_threshold:
            self.step_count += 1
            return self.targetEntropy

        self.conditioned_count += 1
        print("error_count:", self.conditioned_count)

        # drop only when we exceed the total number of conditioned steps
        if self.conditioned_count >= self.total_conditioned_num:
            print("conditioned mean:", self.EMAvg)
            print("conditioned std:", np.sqrt(self.EMVar))
            self.conditioned_count = 0
            self.targetEntropy *= self.targetEntropyDiscount

        return self.targetEntropy

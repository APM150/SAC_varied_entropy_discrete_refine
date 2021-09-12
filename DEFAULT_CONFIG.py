import ray.rllib.agents.sac as sac
import numpy as np

DEFAULT_CONFIG = sac.DEFAULT_CONFIG.copy()
# piecewise schedule configs
maxE = np.log(18)
DEFAULT_CONFIG["target_entropy_schedule"] = [(0, 0.98*maxE), (200_000, 0.75*maxE), (400_000, 0.5*maxE), (600_000, 0.25*maxE), (800_000, 0.01*maxE)]
# DEFAULT_CONFIG["target_entropy_stable_policy_schedule"] = [1.0767, 0.8789, 0.7141, 0.4394, 0.1099, 0.0549]

# exp schedule configs
DEFAULT_CONFIG["target_entropy_stable_policy_discount"] = 0.90
DEFAULT_CONFIG["init_train_step"] = 0
DEFAULT_CONFIG["total_conditioned_num"] = 2000
DEFAULT_CONFIG["force_drop_steps"] = 180_000
DEFAULT_CONFIG["lambda"] = 0.999
DEFAULT_CONFIG["avg_threshold"] = 0.01
DEFAULT_CONFIG["std_threshold"] = 0.05

DEFAULT_CONFIG["trainer_name"] = "MySAC(512,512,stable_policy_exp,discount=0.9,lambda=0.999,stdth=0.05)"
DEFAULT_CONFIG["scheduling"] = "stable_policy_exp"  # vanilla
                                                    # stable_policy_exp
                                                    # piecewise

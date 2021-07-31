import ray.rllib.agents.sac as sac

DEFAULT_CONFIG = sac.DEFAULT_CONFIG.copy()
# piecewise schedule configs
DEFAULT_CONFIG["target_entropy_schedule"] = [(0, 1.0767), (26500, 0.8789), (51500, 0.7141), (76500, 0.4394), (101500, 0.1099), (126500, 0.0549)]
DEFAULT_CONFIG["target_entropy_stable_policy_schedule"] = [1.0767, 0.8789, 0.7141, 0.4394, 0.1099, 0.0549]

# exp schedule configs
DEFAULT_CONFIG["target_entropy_stable_policy_discount"] = 0.90
DEFAULT_CONFIG["init_train_step"] = 0
DEFAULT_CONFIG["total_conditioned_num"] = 2000
DEFAULT_CONFIG["force_drop_steps"] = 180_000
DEFAULT_CONFIG["lambda"] = 0.999
DEFAULT_CONFIG["avg_threshold"] = 0.01
DEFAULT_CONFIG["std_threshold"] = 0.05

DEFAULT_CONFIG["trainer_name"] = "MySAC(512,512,vanilla,TE=0.98)" #stable_policy_exp,discount=0.9,lambda=0.999,stdth=0.05)"
DEFAULT_CONFIG["scheduling"] = "vanilla"  # vanilla
                                                    # stable_policy_exp
                                                    # piecewise

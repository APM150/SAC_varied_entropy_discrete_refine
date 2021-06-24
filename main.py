import ray
from ray import tune
from ray.rllib.agents.sac import SACTrainer
from trainer import MySACTrainer


ray.init()
config = {
    "framework": "torch",
    "env": 'BreakoutNoFrameskip-v4',
    "num_gpus": 1,
    "buffer_size": int(1e5),
    "Q_model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [512, 512],
    },
    "policy_model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [512, 512],
    },
    "optimization": {
        "actor_learning_rate": 3e-5,
        "critic_learning_rate": 3e-4,
        "entropy_learning_rate": 3e-4,
    },
}

tune.run(MySACTrainer, config=config, checkpoint_freq=10, keep_checkpoints_num=2)

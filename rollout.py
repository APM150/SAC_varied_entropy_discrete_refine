from trainer import MySACTrainer
import ray
import gym
import time
import torch
from utils import compute_policy_entropy
from ray.rllib.env.atari_wrappers import wrap_deepmind

ray.init()
env = wrap_deepmind(gym.make('HeroNoFrameskip-v4'))
# env = make_atari('HeroNoFrameskip-v4')
config = {
    "framework": "torch",
    "num_gpus": 1,
    # "target_entropy": 0.05129 * np.log(6),
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
trainer = MySACTrainer(env="HeroNoFrameskip-v4", config=config)

# Load model
trainer.restore("/media/apm150/3C78B17878B1320E/Users/13729/rllib_experiments/Hero/"
                "MySAC(512,512,stable_policy_exp,discount=0.9,lambda=0.999,stdth=0.05)_2021-07-27_19-02-41/"
                "MySAC(512,512,stable_policy_exp,discount=0.9,lambda=0.999,stdth=0.05)_HeroNoFrameskip-v4_e4117_00000_0_2021-07-27_19-02-41/"
                "checkpoint_12520/checkpoint-12520")

total_episode = 50
episode_cnt = total_episode
obs = env.reset()
act = trainer.compute_action(obs)
obs, reward, done, info = env.step(act)
episode_return = 0
while episode_cnt:
    act = trainer.compute_action(obs)
    # print(act)
    env.render()
    obs, reward, done, info = env.step(act)
    episode_return += reward
    time.sleep(0.05)

    if done:
        print(f"Episode {total_episode - episode_cnt + 1} finished. Return={episode_return}")
        obs = env.reset()
        episode_cnt -= 1
        episode_return = 0

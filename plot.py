import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()


# Example data
envs = (
    # "Lunarlander",
    # "CartPole-v0",
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "BankHeist",
    "BattleZone",
    "Boxing",
    "Breakout",
    "ChopperCommand",
    "CrazyClimber",
    "DemonAttack",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Hero",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "MsPacman",
)
exp = [
    685.93,
    42.07,
    337.03,
    378.5,
    35.40,
    5790.0,
    -1.87,
    2.65,
    546.37,
    4.0,
    118.60,
    13.57,
    81.03,
    154.87,
    1908.67,
    31.33,
    307.33,
    4717.67,
    1408.0,
]
ppo = [
    361.8,
    77.21,
    366,
    296.5,
    73.5,
    4960,
    -42.8,
    8.95,
    600,
    13943,
    63.33,
    0,
    0,
    604.8,
    7172,
    46.5,
    386,
    2780,
    828.1,
]
dqn = [
    235.7,
    49.63,
    376.1,
    510,
    7,
    3080,
    -29.74,
    10.23,
    705.6,
    956,
    600,
    21.79,
    133,
    71.4,
    534.8,
    14,
    202,
    239.8,
    341.4,
]
randoms = [
    184.8,
    11.8,
    233.7,
    248.8,
    15.0,
    2895.0,
    0.3,
    0.9,
    671.0,
    7339.5,
    140.0,
    0.0,
    74.0,
    245.9,
    224.6,
    29.2,
    42.0,
    1543.3,
    235.2,
]
human = [
    7128.0,
    1720.0,
    742.0,
    8503.0,
    753.0,
    37188.0,
    12.0,
    30.0,
    7388.0,
    35829.0,
    1971.0,
    30.0,
    4334.7,
    2412.0,
    30826.0,
    303.0,
    3035.0,
    2666.0,
    6952.0,
]
TE0_98 = [
    196.7,
    1.36,
    245.07,
    220.67,
    10.57,
    3816.67,
    0.64,
    1.32,
    802.17,
    3560.67,
    165.4,
    0.82,
    54.43,
    273.0,
    289.77,
    41.67,
    45.33,
    1647.0,
    253.2,
]
TE0_5 = [
    966.07,
    29.65,
    288.97,
    421.83,
    25.63,
    5103.33,
    -1.58,
    2.60,
    206.67,
    2036.33,
    182.93,
    8.26,
    245.47,
    344.40,
    481.25,
    60.67,
    241.33,
    3383.67,
    1213.33,
]
TE0_01 = [
    640.77,
    31.14,
    366.43,
    209.0,
    2.2,
    4885.0,
    -28.5,
    1.84,
    606.15,
    169.5,
    99.55,
    21.95,
    32.03,
    214.2,
    200.60,
    6.33,
    325.33,
    4716.66,
    1315.83,
]

fig, ax = plt.subplots()
y_pos = np.arange(len(envs))
exp_dqn_Norm = []
for i in range(len(randoms)):
    exp_dqn_Norm.append((exp[i] - randoms[i]) / (dqn[i] - randoms[i]))
    if dqn[i] - randoms[i] <= 0:
        exp_dqn_Norm[-1] = (exp[i] - randoms[i] + 1) / (dqn[i] - randoms[i] + 1)
    if dqn[i] - randoms[i] < 0 and exp[i] - randoms[i] > 0:
        exp_dqn_Norm[-1] = -exp_dqn_Norm[-1]
# exp_dqn_Norm = (np.array(exp) - randoms) / (np.array(dqn) - randoms).tolist()
ax.barh(y_pos, exp_dqn_Norm, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(envs)
ax.invert_yaxis()  # labels read top-to-bottom
ax.vlines(1, ymin=-0.5, ymax=len(envs)-0.5, colors='black')
ax.set_xlabel('Performance')
ax.set_title("Normalized performance compared to RLlib defult DQN")
plt.savefig("plots/norm_exp_vs_dqn")

fig, ax = plt.subplots()
exp_human_Norm = []
for i in range(len(randoms)):
    exp_human_Norm.append((exp[i] - randoms[i]) / (human[i] - randoms[i]))
    if human[i] - randoms[i] <= 0:
        exp_human_Norm[-1] = (exp[i] - randoms[i] + 1) / (human[i] - randoms[i] + 1)
    if human[i] - randoms[i] < 0 and exp[i] - randoms[i] > 0:
        exp_human_Norm[-1] = -exp_human_Norm[-1]
# exp_human_Norm = (np.array(exp) - randoms) / (np.array(human) - randoms).tolist()
ax.barh(y_pos, exp_human_Norm, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(envs)
ax.invert_yaxis()  # labels read top-to-bottom
ax.vlines(1, ymin=-0.5, ymax=len(envs)-0.5, colors='black')
ax.set_xlabel('Performance')
ax.set_title("Normalized performance compared to Human")
plt.savefig("plots/norm_exp_vs_hum")

fig, ax = plt.subplots()
exp_ppo_Norm = []
for i in range(len(randoms)):
    if ppo[i] - randoms[i] <= 0:
        exp_ppo_Norm.append(0)
    else:
        exp_ppo_Norm.append((exp[i] - randoms[i]) / (ppo[i] - randoms[i]))
    if ppo[i] - randoms[i] <= 0:
        exp_ppo_Norm[-1] = (exp[i] - randoms[i] + 1) / (ppo[i] - randoms[i] + 1)
    if ppo[i] - randoms[i] < 0 and exp[i] - randoms[i] > 0:
        exp_ppo_Norm[-1] = -exp_ppo_Norm[-1]
# exp_ppo_Norm = (np.array(exp) - randoms) / (np.array(ppo) - randoms).tolist()
ax.barh(y_pos, exp_ppo_Norm, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(envs)
ax.invert_yaxis()  # labels read top-to-bottom
ax.vlines(1, ymin=-0.5, ymax=len(envs)-0.5, colors='black')
ax.set_xlabel('Performance')
ax.set_title("Normalized performance compared to RLlib defult PPO")
plt.savefig("plots/norm_exp_vs_ppo")

fig, ax = plt.subplots()
exp_001_Norm = []
for i in range(len(randoms)):
    exp_001_Norm.append((exp[i] - TE0_98[i]) / (TE0_01[i] - TE0_98[i]))
    if TE0_01[i] - TE0_98[i] <= 0:
        exp_001_Norm[-1] = (exp[i] - TE0_98[i] + 1) / (TE0_01[i] - TE0_98[i] + 1)
    if TE0_01[i] - TE0_98[i] < 0 and exp[i] - TE0_98[i] > 0:
        exp_001_Norm[-1] = -exp_001_Norm[-1]
# exp_001_Norm = (np.array(exp) - TE0_98) / (np.array(TE0_01) - TE0_98).tolist()
ax.barh(y_pos, exp_001_Norm, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(envs)
ax.invert_yaxis()  # labels read top-to-bottom
ax.vlines(1, ymin=-0.5, ymax=len(envs)-0.5, colors='black')
ax.set_xlabel('Performance')
ax.set_title("Normalized performance compared to TE=0.01")
plt.savefig("plots/norm_exp_vs_TE001")

fig, ax = plt.subplots()
exp_05_Norm = []
for i in range(len(randoms)):
    exp_05_Norm.append((exp[i] - TE0_98[i]) / (TE0_5[i] - TE0_98[i]))
    if TE0_5[i] - TE0_98[i] <= 0:
        exp_05_Norm[-1] = (exp[i] - TE0_98[i] + 1) / (TE0_5[i] - TE0_98[i] + 1)
    if TE0_5[i] - TE0_98[i] < 0 and exp[i] - TE0_98[i] > 0:
        exp_05_Norm[-1] = -exp_05_Norm[-1]
# exp_05_Norm = (np.array(exp) - TE0_98) / (np.array(TE0_5) - TE0_98).tolist()
ax.barh(y_pos, exp_05_Norm, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(envs)
ax.invert_yaxis()  # labels read top-to-bottom
ax.vlines(1, ymin=-0.5, ymax=len(envs)-0.5, colors='black')
ax.set_xlabel('Performance')
ax.set_title("Normalized performance compared to TE=0.5")
plt.savefig("plots/norm_exp_vs_TE05")

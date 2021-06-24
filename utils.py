import torch

def compute_policy_entropy(log_pis_t, policy_t):
    """
    :param log_pis_t: log probability of each action, shape (batch, action_space)
    :param policy_t: probability of each action, shape (batch, action_space)
    :return: expectation of policy entropy, shape (1,)
    """
    return -torch.mean(torch.sum(log_pis_t * policy_t, dim=1))

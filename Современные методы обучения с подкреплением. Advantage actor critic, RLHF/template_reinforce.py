import numpy as np
import torch
import torch.nn as nn

def to_one_hot(y_tensor, ndims):
    """ helper: take an integer vector and convert it to 1-hot matrix. """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot


def predict_probs(states):
    """
    Predict action probabilities given states.
    :param states: numpy array of shape [batch, state_shape]
    :returns: numpy array of shape [batch, n_actions]
    """
    # Use no_grad to suppress gradient calculation.
    with torch.no_grad():
        #перевод в tensor для model
        nn_states=torch.tensor(states,dtype=torch.float32)
        logi=model(nn_states)
        probs = torch.softmax(logi, dim=-1)
        #обратно
        return probs.numpy()

def get_cumulative_rewards(rewards,  # rewards at each step
                           gamma=0.99  # discount for reward
                           ):
    """
    Take a list of immediate rewards r(s,a) for the whole session
    and compute cumulative returns (a.k.a. G(s,a) in Sutton '16).

    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    A simple way to compute cumulative rewards is to iterate from the last
    to the first timestep and compute G_t = r_t + gamma*G_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """
    #G_t = r_t + gamma*G_{t+1}
    #то есть для Gt+1=0 так как нет будущ наград
    cumulative_rewards=[]
    Gt=0
    #считаем с конца по формулке выше
    for i in reversed(rewards):
        Gt=i+gamma*Gt
        #cumulative_rewards.insert(0,Gt) за O(n), так что возможно развернув будет даже быстрее
        cumulative_rewards.append(Gt)
    return cumulative_rewards[::-1]


def get_loss(states, actions, rewards, gamma=0.99, entropy_coef=1e-2):
    """
    Compute the loss for the REINFORCE algorithm.
    """
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)  # тут похоже опечатка так как int32 не работает, он наоборот в минимум стремится почему то
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

    # predict logits, probas and log-probas using an agent.
    logits = model(states)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)

    # как по лекции через one-hot, но он медленный немножко, но работают оба
    # one_hot= to_one_hot(actions, n_actions)
    # log_probs_for_actions=torch.sum(log_probs*one_hot, dim=1)

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    # прочитал, что gather будет быстрее правда чтобы он работал, нужно сделать столько же размерностей сколько и у входа
    log_probs_for_actions = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    J_hat = torch.mean(cumulative_returns * log_probs_for_actions)
    p_loss = -J_hat

    entropy = -torch.sum(probs * log_probs, dim=1).mean()
    loss = p_loss - entropy_coef * entropy

    return loss
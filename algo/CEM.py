import numpy as np
import gym
from gym.spaces import Discrete, Box

# ================================================================
# Policies
# Code based on the starter code from John Schulman 
# ================================================================

class CEMPolicy(object):

    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n
        assert len(theta) == (dim_ob + 1) * n_actions
        self.W = theta[0 : dim_ob * n_actions].reshape(dim_ob, n_actions)
        self.b = theta[dim_ob * n_actions : None].reshape(1, n_actions)

    def act(self, ob):
        """
        """
        y = ob.dot(self.W) + self.b
        a = y.argmax()
        return a

def do_episode(policy, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = policy.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew

def noisy_evaluation(theta,env,num_steps):
    policy = make_policy(theta,env)
    rew = do_episode(policy, env, num_steps)
    return rew

def make_policy(theta,env):
    if isinstance(env.action_space, Discrete):
        return CEMPolicy(theta,
            env.observation_space, env.action_space)
    else:
        raise NotImplementedError

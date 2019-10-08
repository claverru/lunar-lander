import time
import logging
from multiprocessing import Pool

import numpy as np
import torch
import gym


def run_agent(agent, show_game=False):
	agent.eval() # set evaluation mode on (for dropout and stuff)
	env = gym.make('LunarLander-v2')
	observation = env.reset()
	r = 0
	start = time.time()
	done = False
	while not done:
		if show_game:
			env.render()
		inp = torch.Tensor(observation).view(1, -1)
		o_prob = agent(inp).detach().numpy()[0]
		action = np.random.choice([0,1,2,3], size=1, p=o_prob).item()
		observation, reward, done, info = env.step(action)
		r += reward
		# Stop condition
		if r < -250 or (time.time() - start) > 120:
			done = True
	env.close()
	return r       


# TODO: Consider multiprocessing
def run_agents(agents, p=None, show_game=False):
	logging.info('Running {} agents.'.format(len(agents)))
	if p:
		map = p.map
	return list(map(run_agent, agents))


def run_agents_n_times(agents, times, p=None, show_game=False):
	"""Return average agents' reward over *times* runs."""
	return np.mean(
			[run_agents(agents, p=p, show_game=show_game) for _ in range(times)], 
			axis=0
		)


def get_elite(contestants, times=3, p=None):
	logging.info('Elite pitting.')
	rewards = run_agents_n_times(contestants, times=3, p=p)
	return contestants[np.argmax(rewards)], max(rewards)
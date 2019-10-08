import logging
import copy

import torch
import numpy as np


# ~200x faster
def mutate(agent, mutation_power=0.002):
	child = copy.deepcopy(agent)
	for p in child.parameters():
		p += torch.Tensor((mutation_power * np.random.rand(*p.size())))
	return child


def mutate_agents(agents, mutation_power=0.002):
	logging.info('Mutating agents.')
	return [mutate(agent) for agent in agents]


# ~150x faster, plus returns two childs
# Valid if parents only mate once cause they got modified inplace
def mate(agent1, agent2):
	for p1, p2 in zip(agent1.parameters(), agent2.parameters()):
		mask = np.random.randint(low=0, high=2, size=p1.size())
		p1, p2 = (torch.Tensor(np.where(mask, p1, p2)), 
				torch.Tensor(np.where(mask, p2, p1)))
	return agent1, agent2


def mate_agents(sorted_agents):
	logging.info('Mating agents.')
	for i in range(1, len(sorted_agents), 2):
		sorted_agents[i-1], sorted_agents[i] = mate(
				sorted_agents[i-1], sorted_agents[i])
	return sorted_agents


def order_agents_by_reward(agents, rewards):
	"""Order from worst to best."""
	logging.info('Ordering agents by their reward.')
	return [agents[i] for i in np.argsort(rewards)], np.sort(rewards)
import logging
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import gym

from magic import play, operations, agents

def evolve(
		generations = 500,
		N = 501,
		T = 50,
		n_elite = 10,
		n_pits = 30,
		env = gym.make('LunarLander-v2'),
		population=[],
		elite = None,
		elite_reward = None):
	"""Simple Genetic Algorithm.

	# References:
		- [Deep Neuroevolution](https://arxiv.org/abs/1712.06567), Section 7.1
	"""
	for g in range(1, generations + 1):
		logging.info('*'*50)
		logging.info('Generation {}'.format(g))
		logging.info('*'*50)
		logging.info('Creating or evolving population.')
		offspring = []
		p = Pool(cpu_count()//2) # Optional for boooooosting
		mutation_power = 0.4 / g
		for i in range(1, N):
			if g is 1:
				# generate new population
				child = agents.get_agent()
			else:
				# operate over old population
				k = np.random.randint(1, T + 1)
				child = operations.mutate(
						population[-k], 
						mutation_power=mutation_power)
			offspring.append(child)
		logging.info('Evaluating population.')

		rewards = play.run_agents(offspring, p=p)
		logging.info('Ordering population by reward.')
		offspring, rewards = operations.order_agents_by_reward(
				offspring, 
				rewards)
		logging.info('Adding elite.')
		if g is 1:
			elite_candidates = offspring[-n_elite:]
		else:
			elite_candidates = offspring[-(n_elite-1):] + [elite]
		elite, elite_reward = play.get_elite(
				elite_candidates,
				times = n_pits,
				p = p)
		population = offspring + [elite]
		np.append(rewards, elite_reward)
		logging.info('*'*50)
		logging.debug('Population size ----- {}'.format(len(population)))
		logging.info('Best reward:  {}'.format(max(rewards)))
		logging.info('Mean rewards: {}'.format(np.mean(rewards)))
		logging.info('Elite reward: {}'.format(elite_reward))
	return elite, elite_reward


def main():
	torch.set_grad_enabled(False)
	evolve()	
		
	
if __name__ == '__main__':
	FORMAT = '%(asctime)-15s %(levelname)s (%(funcName)s) %(message)s'
	logging.basicConfig(format=FORMAT, level=logging.DEBUG)
	main()
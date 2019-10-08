import torch
import torch.nn as nn


def _init_weights(m):
	if type(m) == nn.Linear:
		torch.nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(.0)


def _non_trainable(model):
	for p in model.parameters():
		p.requires_grad = False


def get_agent():
	agent = nn.Sequential(
			nn.Linear(8, 256, bias=True),
			nn.ReLU(),
			nn.Linear(256, 4, bias=True),
			nn.Softmax(dim=1)
		)
	agent.apply(_init_weights)
	_non_trainable(agent)
	return agent
			
				
def get_agents(n):
	logging.info('Getting agents.')
	return [get_agent() for _ in range(n)]
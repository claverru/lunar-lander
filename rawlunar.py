%reset -f
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import datetime
import pickle
import pandas as pd

from gym.wrappers import Monitor



import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import math
import copy



        


class LunarLanderAI(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                        nn.Linear(8,128, bias=True), #8 movimientos iniciales posibles
                        nn.ReLU(),
                        nn.Linear(128,4, bias=True), #4 movimientos posibles en la salida: izquierda, derecha, abajo, no hacer nada
                        nn.Softmax(dim=1) #Generalización de función logística
                        )

                
        def forward(self, inputs):
            x = self.fc(inputs)
            return x
        
        
        
def init_weights(m):
        
        # Inicialización de xavier
        
        if ((type(m) == nn.Linear)):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.00)
            
            
            
def return_random_agents(num_agents):
    
    #Función para devolver num_agents con inicialización de Xavier.
    
    agents = []
    for _ in range(num_agents):
        
        agent = LunarLanderAI()
        
        for param in agent.parameters():
            param.requires_grad = False
            
        init_weights(agent)
        agents.append(agent)
        
        
    return agents




def run_agents(agents,show_game=True):
    
    #Aquí se se ejecuta el lunar lander, si quieres que se muestre el juego se hace "show_game=True"
    
    reward_agents = []
    
    env = gym.make("LunarLander-v2")

    for agent in agents:
        agent.eval()
    
        observation = env.reset()
        r=0
        s=0
        
        tini=datetime.datetime.now()

        while True:

            if(show_game):
                env.render()

            inp = torch.tensor(observation).type('torch.FloatTensor').view(1,-1)
            output_probabilities = agent(inp).detach().numpy()[0]
            action = np.random.choice([0,1,2,3], 1, p=output_probabilities).item()

            observation, reward, done, info = env.step(action)
                
            r=r+reward
        
            s=s+1

            tfin=datetime.datetime.now()
            
            #No puede quedarse volando por siempre, si se queda mucho tiempo volando done=True
            if((r<-250) or ((tfin-tini).seconds >120) ):
                done=True

            if(done):
                env.close()
                break


        reward_agents.append(r)        
        #reward_agents.append(s)
        
    
    return reward_agents





def return_average_score(agent, runs,show_game=True):
    #Para calcular el promedio de la fitness
    score = 0.
    for i in range(runs):
        score_temp=run_agents([agent],show_game)[0]
        score += score_temp
        print("Score for run",i,"has been",score_temp)
    return score/runs





def run_agents_n_times(agents, runs,show_game=True):
    #¿cuántas veces quieres ejecutar el juego?
    avg_score = []
    for agent in agents:
        avg_score.append(return_average_score(agent,runs,show_game))
    return avg_score




def mutate(agent):

    child_agent = copy.deepcopy(agent)
    
    mutation_power = 0.02 # Potencia de mutación \sigma * N(0,1) = N(0,\sigma^2)


    
    for param in child_agent.parameters():
            

            if(len(param.shape)==2): #weights of linear layer
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        
                        param[i0][i1]+= mutation_power * np.random.randn()
                            
    
            elif(len(param.shape)==1): #biases of linear layer or conv layer
                for i0 in range(param.shape[0]):
                    
                    param[i0]+=mutation_power * np.random.randn()
            

        


    return child_agent






def cruce(mother_agent,father_agent):
    
    child_agent = copy.deepcopy(mother_agent)
    
    dim_father=[]
    
    #función de cruce, se lanza una moneda, si sale cara le asignas los pesos del padre
    
    for j in range(len(list(father_agent.parameters()))):
        dim_father.append(list(father_agent.parameters())[j].shape)


    
    for param in child_agent.parameters():
        for j in range(len(dim_father)):
            if(dim_father[j]==param.shape):
                idx=j
        

        if(len(param.shape)==2): 
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    if np.random.uniform(0,1) <= 0.5:
                        param[i0][i1]= list(father_agent.parameters())[idx][i0][i1]

                        

        elif(len(param.shape)==1): 
            for i0 in range(param.shape[0]):
                if np.random.uniform(0,1) <= 0.5:
                    param[i0]= list(father_agent.parameters())[idx][i0]
        


    return child_agent    

    
    
    
    

def return_children(agents, sorted_parent_indexes, elite_index):
    
    #esta función es la que devuelve todos los hijos, mutados y cruzados.
    
    children_agents = []
    
    print(datetime.datetime.now(),"Start: Crossing and Muting agents...")
    
    max_idx=1
    while max_idx<=len(sorted_parent_indexes):
        
        for i in range(max_idx):
            children=cruce(
                mother_agent=agents[sorted_parent_indexes[i]],
                father_agent=agents[max_idx])
            children_agents.append(mutate(children))
            if(len(children_agents)>=(num_agents-1)):
                break
        
        max_idx=max_idx+1
           
    
    print(datetime.datetime.now(),"End: Crossing and Muting agents...")
    
    
    mutated_agents=[]
    
    
    #now add one elite
    elite_child = add_elite(agents, sorted_parent_indexes, elite_index)
    children_agents.append(elite_child)
    elite_index=len(children_agents) -1 #it is the last one
    
    return children_agents, elite_index



def add_elite(agents, sorted_parent_indexes, elite_index=None, only_consider_top_n=10):
    
    #Torneo de elitismo, añade a la lista de agentes el élite.
    
    candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]
    
    if(elite_index is not None):
        candidate_elite_index = np.append(candidate_elite_index,[elite_index])
        
    top_score = None
    top_elite_index = None
    
    print(datetime.datetime.now(),"Start: Playing candidates to elite...")
    for i in candidate_elite_index:

        if(i%10==0):
            show=True
        else:
            show=False
        
        score = return_average_score(agents[i],runs=3,show_game=show)
        print("Score for elite i ", i, " is on average", score)
        
        if(top_score is None):
            top_score = score
            top_elite_index = i
        elif(score > top_score):
            top_score = score
            top_elite_index = i
    
    print(datetime.datetime.now(),"End: Playing candidates to elite...")    
    print("Elite selected with index ",top_elite_index, " and average score", top_score)
    
    child_agent = copy.deepcopy(agents[top_elite_index])
    return child_agent



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

#------------------------------------------------------------------------------------------#


#No usaremos gradientes 
torch.set_grad_enabled(False)

# incializamos con 500 redes neuronales
num_agents = 500
agents = return_random_agents(num_agents)


#solucion de la ecuacion de segundo grado n(n+1)/2= num_agents, se toma este numero de esta manera
#por el bucle que se realiza en el cruce.
top_limit = int((-1 + np.sqrt(1 + 4*2*num_agents))/2)+1 


# run evolution until X generations
generations = 5000
input_dimensions=80*80
elite_index = None


df_rewards_performance=pd.DataFrame(columns=["generation","reward_best_agent","average_top_elite","average_all_agents",
                       "median_top_elite","median_all_agents", "sd_top_elite","sd_all_agents"])
    

df_rewards_cummulative_performance=pd.DataFrame(columns=["generation","cumm_reward_best_agent","cumm_average_top_elite",
                                                         "cumm_average_all_agents","cumm_median_top_elite","cumm_median_all_agents"])


    
    
for generation in range(generations):

    # return rewards of agents
    rewards = run_agents_n_times(agents, 1,show_game=False) #return average of 3 runs
    
    # sort by rewards
    sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] 
    print("")
    print("")
    
    top_rewards = []
    for best_parent in sorted_parent_indexes:
        top_rewards.append(rewards[best_parent])
    
    
    #----------------- Almacenamiento de medidas de rendimiento -----------------------#
    
    
    df_rewards_performance.loc[generation] = [generation,top_rewards[0], np.mean(top_rewards),np.mean(rewards),
                               np.median(top_rewards),np.median(rewards),np.std(top_rewards),np.std(rewards)]
    
    if generation!=0:
        df_rewards_cummulative_performance.loc[generation,0:6]=np.append(generation,df_rewards_performance.iloc[generation,1:6].values+df_rewards_performance.iloc[(generation-1),1:6].values)
    else:
        df_rewards_cummulative_performance.loc[generation]=df_rewards_performance.iloc[generation,0:6].values
        
        
    
    
    
    plt.figure()
    norm_average=(df_rewards_cummulative_performance.cumm_average_top_elite-np.mean(df_rewards_cummulative_performance.cumm_average_top_elite))/np.std(df_rewards_cummulative_performance.cumm_average_top_elite)
    plt.plot(df_rewards_performance.generation,norm_average,"go-")
    plt.show()
    
    #--------------- Fin: Almacenamiento de medidas de rendimiento --------------------#
    
    
    print("Generation ", generation, " | Mean rewards all players: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))
    #print(rewards)
    print("Top ",top_limit," scores", sorted_parent_indexes)
    print("Rewards for top: ",top_rewards)
    
    
    children_agents, elite_index = return_children(agents, sorted_parent_indexes, elite_index)
    
    agents = children_agents
    

     
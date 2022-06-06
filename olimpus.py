import sys
import numpy as np
from time import perf_counter

print("Importando torch...")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
print("Torch importado!")

import gym
import gym_microrts
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

# Librerias propias
from utils import layer_init, get_unit_masks, sample
from parse import valid_action_mask, reduce_dim_observation

np.set_printoptions(threshold=sys.maxsize)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)





class PPOMemory:

    # Constructor
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size
    
    # Genera los batches para actualizar la red neuronal de accion
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.values),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    # Permite almacenar en la clase el conjunto de valores para almacenar
    # en los batches
    def store_memory(self,state, action, probs, values, reward, done):
        self.states.append(state)
        self.probs.append(probs)
        self.values.append(values)
        self.reward.append(reward)
        self.done.append(done)

    # Limpia las listas con los valores guardados para 
    # actualizar nuevamente
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []

# END PPO MEMORY CLASS

class Olimpus(nn.Module):

    # Constructor
    def __init__(self, map_size, env):
        super(Olimpus, self).__init__()

        # Crear red convolucional "Atenea"
        self.atenea = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size = 3, stride = 2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size = 2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(6*6*32, 256)),
                nn.ReLU())

        # Aqui crear Red Lineal Ares  (entrada de 256 neuronas y salida de 19.968 (mapa 16x16)
        self.ares = layer_init(nn.Linear(256, env.action_space.nvec.sum()), std=0.01)

        # Aqui crear Red Lineal Hefesto (misma entrada y salida de Ares)
        self.hefesto = layer_init(nn.Linear(256, env.action_space.nvec.sum()), std = 0.01)
        

        # Optimizadores
        optim_atenea = torch.optim.Adam(self.atenea.parameters(), lr = 1e-4)
        optim_ares = torch.optim.Adam(self.ares.parameters(), lr = 1e-4)
        optim_hefesto = torch.optim.Adam(self.hefesto.parameters(), lr = 1e-4)


        # Puntero al ambiente
        self.env = env
        
        # Para guardar eventualmente las source unit mask de milicias y productores
        self.ares_source_unit_mask = None
        self.hefesto_source_unit_mask = None

    # END CONSTRUCTOR

    # Obtener output de la red
    def forward(self, x, observation):
        # Rotar tensor de dimensiones (1, h, w, 27) a (1, 27, h, w)
        #print("Permutando vector")
        x = x.permute((0, 3, 1, 2))
        # Ingresar tensor rotado a la red convolucional (se procesa como una imagen)
        # Retorna una impresion global de la observacion
        output_atenea = self.atenea(x)

        # Retornar movimientos de Hefesto y Ares combinados
        return self.get_action(output_atenea, observation, self.env)
    
    # Toma la vision global de atenea y escoge acciones para la milicia y para las unidades productivas
    def get_action(self, input_tensor, observation, env):
        # x: Salida de la red convolucional

        # Step 0

        hefesto_logits = self.hefesto(input_tensor)       # Tensor (78hw, )
        ares_logits = self.ares(input_tensor)             # Tensor (78hw, )

        # Obtener mascara de acciones
        action_mask = torch.from_numpy(env.get_action_mask().reshape((1, -1)))


        # Step 1      

        # Esto da 1972 tensores de tamaños variables siguiendo la forma del nvec.tolist()
        split_hefesto_logits = torch.split(hefesto_logits, env.action_space.nvec.tolist(), dim= 1)  # Split Hefesto
        split_ares_logits = torch.split(ares_logits, env.action_space.nvec.tolist(), dim = 1)       # Split Ares
        action_mask = torch.split(action_mask, env.action_space.nvec.tolist(), dim = 1)             # Split action mask


        
        hefesto_actions = []
        hefesto_probs = 0 #hefesto_probs = []
        ares_actions = []
        ares_probs = 0    #ares_probs = []

        # Aplicar mascara a cada tensor de [6, 4, 4, 4, 4, 7, 49]
        for i in range(len(action_mask)):
            # Aplicar mascara con where a Hefesto
            logits_hefesto = torch.where(action_mask[i].type(torch.BoolTensor).to(DEVICE), split_hefesto_logits[i], torch.tensor(-1e+8).to(DEVICE))
            # Aplicar mascara con where a Ares
            logits_ares = torch.where(action_mask[i].type(torch.BoolTensor).to(DEVICE), split_ares_logits[i], torch.tensor(-1e+8).to(DEVICE))  
            
            # Aplicar softmax sobre los tensores de Hefesto y Ares
            #soft_hefesto = F.softmax(logits_hefesto, dim = 1)
            #soft_ares = F.softmax(logits_ares, dim = 1)
            # Aplicar Categorical para obtener la distribucion
            m_hefesto = Categorical(logits=logits_hefesto)
            m_ares = Categorical(logits=logits_ares)
            # Obtener acciones y probabilidades
            action_hefesto = m_hefesto.sample()
            action_ares = m_ares.sample()
            # Append de las acciones y probabilidades a las listas de accion de cada uno
            hefesto_actions.append(action_hefesto.cpu().detach().numpy()[0])
            hefesto_probs += m_hefesto.log_prob(action_hefesto).cpu().detach().numpy()[0]
            ares_actions.append(action_ares.cpu().detach().numpy()[0])
            ares_probs += m_ares.log_prob(action_ares).cpu().detach().numpy()[0]
        #END FOR

        # Fusionar acciones de hefesto y ares
        #print("source unit mask shape: ", env.source_unit_mask[0].shape)
        for i in range(len(env.source_unit_mask[0])):
            # Si no es una unidad, siguiente posicion
            if env.source_unit_mask[0][i] != 1:
                continue
            
            # Si es una unidad propia...
            # Y si es una unidad militar 5: light  6: heavy o 7: ranged
            if np.argmax(observation.ravel()[27 * i + 13: 27 * i + 21]) > 4:
                # Copiar las 7 acciones de Ares en las acciones de Hefesto 
                hefesto_actions[7 * i : 7 * i + 7] = ares_actions[7 * i : 7 * i + 7]




        
        # Imprimir acciones
        #print(len(hefesto_actions))
        #print(hefesto_actions)
        
        #print("--------_")
        #print(len(ares_actions))

        return hefesto_actions, hefesto_probs, ares_probs
           

class CriticNetwork(nn.Module):
    def __init__(self, l_rate, map_size):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(map_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr = l_rate)

        self.to(DEVICE)

    def forward(self, state):
        value = self.critic(state)

        return value



class Agent:

    def __init__(self, env, n_actions, h_map=16, w_map=16, gamma=0.99, l_rate=0.0003, gae_lambda=0.95, p_clip=0.2, batch_size = 64, n_epochs=20):
        self.gamma = gamma
        self.l_rate = l_rate
        self.gae_lambda = gae_lambda
        self.p_clip = p_clip

        self.olimpus = Olimpus(h_map*w_map, env).to(DEVICE)
        self.critic = CriticNetwork(l_rate, h_map*w_map)
        self.memory = PPOMemory(batch_size)


    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def select_action(self, observation):
        input_tensor = torch.from_numpy(observation).float().to(DEVICE)
        return self.olimpus(input_tensor, obs)

    def learn(self):

        # Aprenderemos por T epochs
        for _ in range(self.n_epochs):
            # Obtenremos los batches generados por la memoria
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr,\
            done_arr, batches = self.memory.generate_batches()

            # Cambiamos nomenclatura del arreglo de valores por values
            values = vals_arr
            # Creamos un arreglo para los valores de ventaja a calcular
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Calcularemos T-1 valores de ventaja
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0

                # Calcular el lambda_t de la formula, que va de t a T-1
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda

                # Se añade el valor de ventaja del estado al arreglo
                advantage[t] = a_t

            # Se convierte en un tensor
            advantage = torch.tensor(advantage).to(DEVICE)
            # Se convierten los valores en tensor
            values = torch.tensor(values).to(DEVICE)

            # Para cada batch
            for batch in batches:
                # Estados, probabilidades viejas y acciones a tensores
                states = torch.tensor(state_arr[batch], dtype=T.float).to(DEVICE)
                old_probs = torch.tensor(old_prob_arr[batch]).to(DEVICE)
                actions = torch.tensor(action_arr[batch]).to(DEVICE)

                # Obtenemos nuevas probabilidades con la nueva politica
                dist = self.select_action(states)
                # Tambien los nuevos valores del critico
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)




# --------- Main --------- #
print("Creando ambiente...")
start = perf_counter()
env = MicroRTSGridModeVecEnv(
        num_selfplay_envs = 0,
        num_bot_envs = 1,
        max_steps = 2000,
        render_theme = 1,
        ai2s = [microrts_ai.coacAI for _ in range(1)],
        map_paths = ["maps/16x16/basesWorkers16x16.xml"],
        reward_weight = np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
        )
end = perf_counter()
print(end - start)


# Creamos red neuronal del agente
print("Creando red neuronal Olimpus...")
start = perf_counter()
olimpus = Olimpus(16 * 16, env).to(DEVICE)
end = perf_counter()
print(end - start)


obs = env.reset()
#print("obs type:", type(obs))


print("Creando Agente...")
start = perf_counter()
agent = Agent(env, 10)
end = perf_counter()
print("Agente creado")
print(end - start)
# Y con eso, hefesto y ares mueven sus unidades
print("Obteniendo accion...")
start = perf_counter()
action, hefesto_probs, ares_probs = agent.select_action(obs)
end = perf_counter()
print(end - start)

#print("Prob obtenida en main:", hefesto_probs, ares_probs)
#print("Action:\n")
#print(action)


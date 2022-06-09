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
        self.h_probs = []          # Probs de hefesto
        self.a_probs = []          # Probs de Ares
        self.values = []
        self.actions = []          # Acciones de Olimpus
        self.h_rewards = []        # Recompensas de Hefesto
        self.a_rewards = []        # Recompensas de Ares
        self.dones = []
        self.obs = []

        self.batch_size = batch_size
    
    # Genera los batches para actualizar la red neuronal de accion
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.obs),\
                np.array(self.actions),\
                np.array(self.h_probs),\
                np.array(self.a_probs),\
                np.array(self.values),\
                np.array(self.h_rewards),\
                np.array(self.a_rewards),\
                np.array(self.dones),\
                batches

    # Permite almacenar en la clase el conjunto de valores para almacenar
    # en los batches
    def store_memory(self,state, obs, action, h_prob, a_prob,  values, h_reward, a_reward, done):
        self.states.append(state)
        self.obs.append(obs)
        self.actions.append(action)
        self.h_probs.append(h_prob)
        self.a_probs.append(a_prob)
        self.values.append(values)
        self.h_rewards.append(h_reward)
        self.a_rewards.append(a_reward)
        self.dones.append(done)

    # Limpia las listas con los valores guardados para 
    # actualizar nuevamente
    def clear_memory(self):
        self.states = []
        self.h_probs = []
        self.a_probs = []
        self.actions = []
        self.h_rewards = []
        self.a_rewards = []
        self.dones = []
        self.values = []
        self.obs = []

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
        self.optim_atenea = torch.optim.Adam(self.atenea.parameters(), lr = 1e-4)
        self.optim_ares = torch.optim.Adam(self.ares.parameters(), lr = 1e-4)
        self.optim_hefesto = torch.optim.Adam(self.hefesto.parameters(), lr = 1e-4)


        # Puntero al ambiente
        self.env = env
        
        # Para guardar eventualmente las source unit mask de milicias y productores
        self.ares_source_unit_mask = None
        self.hefesto_source_unit_mask = None

    # END CONSTRUCTOR

    # Obtener output de la red
    def forward(self, x):
        # Rotar tensor de dimensiones (1, h, w, 27) a (1, 27, h, w)
        #print("Permutando vector")
        x = x.permute((0, 3, 1, 2))
        # Ingresar tensor rotado a la red convolucional (se procesa como una imagen)
        # Retorna una impresion global de la observacion
        output_atenea = self.atenea(x)

        # Retornar movimientos de Hefesto y Ares combinados
        return self.get_action(output_atenea, x.cpu().detach().numpy(), self.env)
    
    # Toma la vision global de atenea y escoge acciones para la milicia y para las unidades productivas
    def get_action(self, input_tensor, observation, env):
        # x: Salida de la red convolucional

        # Step 0
        # TO DO: FIX THIS DIMENSION ISSUE
        hefesto_logits = self.hefesto(input_tensor)       # Tensor (78hw, )
        ares_logits = self.ares(input_tensor)             # Tensor (78hw, )

        print("Hefesto logits:", hefesto_logits[0].shape)

        print("Hefesto logits:", hefesto_logits[1].shape)
        
        print("Hefesto logits:", hefesto_logits[2].shape)
        # Obtener mascara de acciones
        action_mask = torch.from_numpy(env.get_action_mask().reshape((1, -1)))
        print(action_mask.shape)


        # Step 1      

        # Esto da 1972 tensores de tamaños variables siguiendo la forma del nvec.tolist()
        split_hefesto_logits = torch.split(hefesto_logits, env.action_space.nvec.tolist(), dim= 1)  # Split Hefesto
        print(len(split_hefesto_logits))
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
                nn.Linear(map_size * 27, 256),
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
        self.n_epochs = n_epochs

        self.olimpus = Olimpus(h_map*w_map, env).to(DEVICE)
        self.critic = CriticNetwork(l_rate, h_map*w_map)
        self.memory = PPOMemory(batch_size)


    def remember(self, state, obs, action, h_probs, a_probs, vals, h_reward, a_reward, done):
        self.memory.store_memory(state, obs, action,  h_probs, a_probs, vals, h_reward, a_reward, done)

    def select_action(self, observation):
        input_tensor = torch.from_numpy(observation).float().to(DEVICE)
        return *self.olimpus(input_tensor), self.critic(input_tensor.flatten())

    def learn(self):

        # Aprenderemos por T epochs
        for _ in range(self.n_epochs):
            # Obtenremos los batches generados por la memoria
            state_arr, obs_arr, action_arr, h_old_prob_arr, a_old_prob_arr, vals_arr, h_reward_arr, a_reward_arr,\
            done_arr, batches = self.memory.generate_batches()

            # Cambiamos nomenclatura del arreglo de valores por values
            values = vals_arr
            # Creamos un arreglo para los valores de ventaja a calcular
            advantage = np.zeros(len(h_reward_arr), dtype=np.float32)

            # Calcularemos T-1 valores de ventaja
            for t in range(len(h_reward_arr) - 1):
                discount = 1
                a_t = 0

                # Calcular el lambda_t de la formula, que va de t a T-1
                for k in range(t, len(h_reward_arr)-1):
                    a_t += discount * ((h_reward_arr[k] + a_reward_arr[k]).mean() + self.gamma*values[k+1]*(1-int(done_arr[k])) - values[k])
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
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(DEVICE)
                h_old_probs = torch.tensor(h_old_prob_arr[batch]).to(DEVICE)
                a_old_probs = torch.tensor(a_old_prob_arr[batch]).to(DEVICE)

                # Obtenemos nuevas probabilidades con la nueva politica
                # TO DO: Arreglar las dimensiones de esta wea, el obs_arr creo que no va
                print("Batch size:", len(batch))
                h_new_probs = [] 
                a_new_probs =[] 
                critic_value = []
                for x in states:
                    # Insertamos los estados del batch en el actor y obtenemos las probabilidades
                    _, h_aux, a_aux = self.olimpus(x)
                    h_new_probs.append(h_aux)
                    a_new_probs.append(a_aux)

                    # Insertamos los estados del batch en el critico y obtenemos el valor del estado
                    critic_value.append(self.critic(x.ravel()))
        
                # Pasamos lo del for anterior a tensores
                h_new_probs = torch.tensor(np.array(h_new_probs), device=DEVICE, requires_grad=True)
                a_new_probs = torch.tensor(np.array(a_new_probs), device=DEVICE, requires_grad = True) 
                critic_value = torch.tensor(critic_value, device = DEVICE, requires_grad =True)
                critic_value = torch.squeeze(critic_value)



                # Obtenemos el ratio
                h_prob_ratio = (h_new_probs - h_old_probs).exp()
                a_prob_ratio = (a_new_probs - a_old_probs).exp()

                # Obtenemos r_t * At sin clippear
                h_weighted_probs = advantage[batch] * h_prob_ratio
                a_weighted_probs = advantage[batch] * a_prob_ratio

                # Obtenemos r_t * At clippeado
                h_weighted_clipped_probs = torch.clamp(h_prob_ratio, 1 - self.p_clip, 1 + self.p_clip) * advantage[batch]
                a_weighted_clipped_probs = torch.clamp(a_prob_ratio, 1 - self.p_clip, 1 + self.p_clip) * advantage[batch]

                # Obtenemos el loss para hefesto y ares
                h_loss = -torch.min(h_weighted_probs, h_weighted_clipped_probs).mean()
                a_loss = -torch.min(a_weighted_probs, a_weighted_clipped_probs).mean()

                # Loss de la red Critico
                returns = advantage[batch] + values[batch]
                critic_loss = ((returns - critic_value)**2).mean()

                # Total loss (aqui falta un termino, el de la entropía)
                h_total_loss = h_loss + 0.5 * critic_loss
                a_total_loss = a_loss + 0.5 * critic_loss
                atenea_loss = (h_total_loss + a_total_loss).mean()

                # Actualizar las redes
                self.olimpus.optim_atenea.zero_grad()
                atenea_loss.backward(retain_graph=True)
                self.olimpus.optim_atenea.step()
            

                self.olimpus.optim_ares.zero_grad()
                a_total_loss.backward(retain_graph=True)
                self.olimpus.optim_ares.step()
                self.olimpus.optim_hefesto.zero_grad()
                h_total_loss.backward(retain_graph=True)
                self.olimpus.optim_hefesto.step()
        self.memory.clear_memory()





# --------- Main --------- #
print("Creando ambiente...")
start = perf_counter()
env = MicroRTSGridModeVecEnv(
        num_selfplay_envs = 2,
        num_bot_envs = 1,
        max_steps = 2000,
        render_theme = 1,
        ai2s = [microrts_ai.coacAI for _ in range(1)],
        map_paths = ["maps/16x16/basesWorkers16x16.xml"],
        reward_weight = np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
        )
end = perf_counter()
print(end - start)

# Hiperparametros
n_juegos = 300       # Cantos juegos como máximo se juegan para entrenar
N = 20               # Cada cuantos pasos se actualiza la red

# Indicadores extra
h_reward_history = []
a_reward_history = []


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
agent = Agent(env, 10000)
end = perf_counter()
print("Agente creado")
print(end - start)



"""
# Y con eso, hefesto y ares mueven sus unidades
print("Obteniendo accion...")
start = perf_counter()
action, hefesto_probs, ares_probs = agent.select_action(obs)
end = perf_counter()
print(end - start)

print("Prob obtenida en main:", hefesto_probs, ares_probs)
#print("Action:\n")
#print(action)
"""
# Recordatorio de indices de rewards:
# 0: win, lose, draw   1: Harvest  2: produce worker  3: construct building   4: valid attack action  5: produce militia
# info[0]["raw_rewards"][5]    # Recompensas por crear una milicia

learn_iters = 0
n_steps = 0

for i in range(n_juegos):
    obs = env.reset()
    done = False
    h_score = 0
    a_score = 0

    while not done:
        # Obtener accion
        action, h_probs, a_probs, val = agent.select_action(obs)
        next_obs, reward, done, info = env.step(np.array(action))
        n_steps += 1

        # Calculamos las reward de cada red
        h_reward = reward                                                     # La suma de todas las recompensas
        a_reward = info[0]["raw_rewards"][0] + info[0]["raw_rewards"][4]      # Ganar partida + atacar unidades
        
        # Añadir al puntaje total de la partida
        h_score += h_reward
        a_score += a_reward

        # Añadir a la memoria los valores
        agent.remember(obs,obs,  action, h_probs, a_probs, val.cpu().detach().numpy(), h_reward, a_reward, done)

        # Actualizar red cada N pasos
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        obs = next_obs
    
    # Guardar registro del puntaje de la partida
    h_reward_history.append(h_score)
    a_reward_history.append(a_score)
    

    print("Episodio", i, "h_score:", h_score, "a_score:", a_score)












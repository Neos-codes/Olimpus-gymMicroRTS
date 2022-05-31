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




# Para obtener las probabilidades de accion de cada red
class CategoricalMasked(Categorical):
    def __init__(self, probs = None, logits = None, validate_args = None, mask = None):

        self.mask = mask
        
        self.mask = mask.type(torch.BoolTensor).to(DEVICE)
        logits = torch.where(self.mask, logits, torch.tensor(-1e+8).to(DEVICE))
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)

        def entropy(self):
            if len(self.mask) == 0:
                return super(CategoricalMasked, self).entropy()
            p_log_p = self.logits *self.probs
            p_log_p = torch.where(self.mask, p_log_p, torch.tensor(0.).to(DEVICE))
            return -p_log_p.sum(-1)





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
        #print("Ingresando en atenea")
        return self.atenea(x)
    
    # Toma la vision global de atenea y escoge acciones para la milicia y para las unidades productivas
    def get_action(self, input_tensor, observation, env):
        # x: Salida de la red convolucional

        # Obtener salida de atenea
        logits = self.forward(input_tensor)    # Tensor (256, )

        # Obtener salidas (acciones) para ares y efesto

        

        # Step 0

        hefesto_logits = self.hefesto(logits)       # Tensor (78hw, )
        ares_logits = self.ares(logits)             # Tensor (78hw, )

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


print("Creando tensor input")
start = perf_counter()
input_tensor = torch.from_numpy(obs).float().to(DEVICE)
end = perf_counter()
print("Tensor input creado")
print(end - start)
# Y con eso, hefesto y ares mueven sus unidades
print("Obteniendo accion...")
start = perf_counter()
action, hefesto_probs, ares_probs = olimpus.get_action(input_tensor, obs, env)
end = perf_counter()
print(end - start)

#print("Prob obtenida en main:", hefesto_probs, ares_probs)
#print("Action:\n")
#print(action)


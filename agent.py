import sys
import numpy as np
from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym
import gym_microrts
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

from utils import separate_su_mask


# Setting para imprimir arreglos y tensores
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# Esto sirve para evitar gradiente desvaneciente
# Inicializacion de los pesos de una capa, con esta función puedes instanciar una capa
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # Inicializacion de pesos ortogonal
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# A "simple" olympus
class Atenea(nn.Module):

    # Constructor de clase
    def __init__(self, h, w, env):
        # Constructor super clase
        super(Atenea, self).__init__()

        # Guardar dimensiones
        self.h = h
        self.w = w
        
        # Observation
        obs = None      # Shape (1, h, w, 27)
        next_obs = None

        # Crear red Atenea, por ahora lineal (upgradear a convolucional)
        self.will = nn.Sequential(
                layer_init(nn.Linear(h * w * 27, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, h * w * 78))   # h * w * 78
                ).to(DEVICE)


        # Puntero al ambiente
        self.env = env

        # Action mask
        self.action_mask = None

    # END CONSTRUCTOR
    
    def forward(self, obs):
        # Convertir en tensor y aplanar input
        input_tensor = torch.from_numpy(obs).float().squeeze().flatten()
        output = self.will(input_tensor.to(DEVICE))   # Insertar en red
        return output

    def get_parameters(self):
        return self.will.parameters()

    def set_obs(self, obs):
        self.obs = obs

    def set_next_obs(self, next_obs):
        self.next_obs = next_obs

# Red con estrategia propia
class God(nn.Module):

    # Constructor de clase
    def __init__(self, h, w, env):
        # Constructor de super clase
        super(God, self).__init__()

        # Guardar dimensiones
        self.h = h
        self.w = w

        # Crear red de voluntad del dios de tipo lineal
        self.will = nn.Sequential(
                layer_init(nn.Linear(h * w * 78, 256)),    # Esta capa debe eliminarse cuando Atenea sea Conv2d
                nn.ReLU(),
                layer_init(nn.Linear(256, env.action_space.nvec.sum()), std=0.01),
                ).to(DEVICE)


        # Puntero al ambiente
        self.env = env


    def forward(self, input_tensor):
        return self.will(input_tensor)

    def get_parameters(self):
        return self.will.parameters()

   
    
class Olympus:

    # Constructor
    def __init__(self, h, w, env):

        # Dimensiones del mapa
        self.h = h
        self.w = w

        # Observaciones
        self.obs = None
        self.next_obs = None

        # Puntero al ambiente
        self.env = env

        # Action mask
        self.action_mask = None

        # Atenea
        self.atenea = Atenea(h, w, env)

        # Hephaestus
        self.heph = God(h, w, env)

        # Ares
        self.ares = God(h, w, env)

        # Optimizadores
        self.atenea_optim = torch.optim.Adam(self.atenea.get_parameters(), lr= 1e-4)
        self.ares_optim = torch.optim.Adam(self.ares.get_parameters(), lr = 1e-4)
        self.heph_optim = torch.optim.Adam(self.heph.get_parameters(), lr = 1e-4)

    # Setear observaciones para Olympus y Atenea (puede que atenea no las necesite)
    def set_obs(self, obs):
        self.obs = obs
        self.atenea.obs = obs


    # Obtener mascaras para unidades milicia y productoras (devuelve las 2 mascaras y 2 banderas si hay o no unidades militares o productoras)
    def type_masks(self):
        # Para obtener el source_unit_mask
        self.action_mask = env.get_action_mask()
        # Devuelve 2 mascaras y 2 banderas de unidades
        return separate_su_mask(env.source_unit_mask, self.obs, self.h, self.w)

    # Obtener la accion en base a la ultima observacion en el ambiente
    def get_action(self):
        # Despues de haber seteado las obs con "set_obs"
        # Ingresar obs en atenea
        at_output = self.atenea(self.obs)

        # Copiar salida de atenea en tensores para ares y hephaestus
        ares_in = at_output.clone().detach().requires_grad_(True)
        heph_in = at_output.clone().detach().requires_grad_(True)

        print(ares_in.requires_grad)
        print(heph_in.requires_grad)

        # Obtener salida de ares y separar en arreglos mas chicos
        # [6, 4, 4, 4, 4, 7, 49]
        ares_out = self.ares(ares_in)
        split_logits = self.split_(ares_out)

        # Lo mismo para Hephaestus
        heph_out = self.heph(heph_in)
        split_logits = self.split_(heph_out)

        return ares_out

    
    # Recorta el tensor en arreglos de [6, 4, 4, 4, 4, 7, 4] hasta tomar todas las salidas de la red  (Privado)
    def split_(self, logits):
        split_logits = torch.split(logits, self.env.action_space.nvec.tolist(), dim = 0)
        return split_logits




# Main

env = MicroRTSGridModeVecEnv(
        num_selfplay_envs = 0,
        num_bot_envs = 1,
        max_steps = 2000,
        render_theme = 1,
        ai2s = [microrts_ai.coacAI for _ in range(1)],
        map_paths = ["maps/8x8/basesWorkers8x8.xml"],
        reward_weight = np.array([1.0, 1.0, 1.0, 0.2, 1.0, 4.0])
        )



olympus = Olympus(8, 8, env)

olympus.set_obs(env.reset()) 

# Hiperparametros
epochs = 100

for i in range(epochs):
    obs = env.reset()
    olympus.set_obs(obs)
    done = False
    
    while not done:
        start = perf_counter()
        env.render()
        
        # Obtener action mask
        action_mask = torch.from_numpy(env.get_action_mask().ravel())   # en cpu (no afecta)
        # Obtener accion con las observaciones ya guardadas
        action = olympus.get_action()    # En cuda:0
"""
        # Aplicar mascara de acciones posibles   [6, 4, 4, 4, 4, 7, 49]
        # 1) Dividir en arreglos mas pequeños
        split_logits = torch.split(action, env.action_space.nvec.tolist(), dim = 0)   # Esto funciona bien   cuda:0
        split_action_mask = torch.split(action_mask, env.action_space.nvec.tolist(), dim = 0)  # Esto tambien funciona bien
        
        # 2) Aplicar la mascara a cada "sub tensor"
        logits = None
        entropy = torch.zeros(1, requires_grad = True, dtype = torch.float, device=DEVICE)
        action = []
        for i in range(len(split_action_mask)):
            logits = torch.where(split_action_mask[i].type(torch.BoolTensor).to(DEVICE), split_logits[i], torch.tensor(-1e8).to(DEVICE))

            # 2.1) Aplicar Softmax al subtensor
            m = nn.Softmax(dim = 0)
            logits = m(logits)
            m = Categorical(logits)
            # 2.2) Escoger accion
            sub_action = m.sample()    # Devuelve el indice de la accion escogida
            log_prob = -m.log_prob(sub_action)
            num = sub_action * log_prob
            tens = torch.tensor(num, requires_grad=True).to(DEVICE)
            entropy = entropy + tens
            # 2.3) Guardar accion
            action.append(sub_action.cpu().detach().numpy())

        # 3) Accion total
        action = np.array(action).reshape((atenea.h * atenea.w, -1))

        atenea.next_obs, rewards, done, info = env.step(action)
        end = perf_counter()

        print("Tiempo en armar accion:", end-start)

        start = perf_counter()
        
        atenea.optim.zero_grad()
        entropy.backward()
        atenea.optim.step()
        
        end = perf_counter()
        print("Tiempo en actualizar:", end - start)
"""

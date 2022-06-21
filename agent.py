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

# Setting para imprimir arreglos y tensores
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# Esto sirve para evitar gradiente desvaneciente
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # Inicializacion de pesos ortogonal
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# A "simple" olympus
class Olympus:
    def __init__(self, h, w, env):
        # Guardar dimensiones
        self.h = h
        self.w = w

        # Crear red Atenea, por ahora linea (upgradear a convolucional)
        self.atenea = nn.Sequential(
                layer_init(nn.Linear(h * w * 27, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, h * w * 78))
                ).to(DEVICE)

        # Optimizador
        self.optim = torch.optim.Adam(self.atenea.parameters(), lr = 1e-4)

        # Puntero al ambiente
        self.env = env

    # END CONSTRUCTOR
    
    def forward(self, input_tensor):
        input_tensor = input_tensor.flatten()           # Aplanar input
        output = self.atenea(input_tensor.to(DEVICE))   # Insertar en red
        print("Output atenea size:", output.size(), end="\n\n")
        return output



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

obs = env.reset() 

# Hiperparametros
epochs = 100

for i in range(epochs):
    obs = env.reset()
    done = False
    
    while not done:
        env.render()
        
        action_mask = torch.from_numpy(env.get_action_mask().ravel())
        print("Action mask shape:", action_mask.shape)

        # Obtener accion
        input_tensor = torch.from_numpy(obs).float().squeeze()
        print("Input tensor size:", input_tensor.size())
        action = olympus.forward(input_tensor)

        # Aplicar mascara de acciones posibles   [6, 4, 4, 4, 4, 7, 49]
        # 1) Dividir en arreglos mas pequeños
        split_logits = torch.split(action, env.action_space.nvec.tolist(), dim = 0)   # Esto funciona bien
        split_action_mask = torch.split(action_mask, env.action_space.nvec.tolist(), dim = 0)  # Esto tambien funciona bien
        
        # 2) Aplicar la mascara a cada "sub tensor"
        logits = None
        for i in range(len(split_action_mask)):
            logits = torch.where(split_action_mask[i].type(torch.BoolTensor).to(DEVICE), split_logits[i], torch.tensor(-1e8).to(DEVICE))
            print(logits)
        # Aqui quedé, hay que hacer el sample con Categorical Distribution

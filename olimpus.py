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
        print("Permutando vector")
        x = x.permute((0, 3, 1, 2))
        # Ingresar tensor rotado a la red convolucional (se procesa como una imagen)
        # Retorna una impresion global de la observacion
        print("Ingresando en atenea")
        return self.atenea(x)
    
    # Toma la vision global de atenea y escoge acciones para la milicia y para las unidades productivas
    def get_action(self, x, observation, env, action_mask):
        # x: Salida de la red convolucional

        # Obtener salidas (acciones) para ares y efesto
        ares_logits = self.ares(x).cpu().detach().numpy()
        hefesto_logits = self.hefesto(x).cpu().detach().numpy()

        # Obtener units mask para ares y hefesto
        reduced_obs = reduce_dim_observation(observation)
        self.ares_source_unit_mask, self.hefesto_source_unit_mask = get_unit_masks(env.source_unit_mask.ravel(), reduced_obs)

        # Ahora falta hacer masking con la funcion valid_action_mask() en el modulo "parse"
        start = perf_counter()

        ares_valid_actions = valid_action_mask(ares_logits[0], self.ares_source_unit_mask, action_mask) # Shape (19968, )
        hefesto_valid_actions = valid_action_mask(hefesto_logits[0],  self.hefesto_source_unit_mask, action_mask)# Shape (19968, )
        
        # Aqui hay que hacer el sample con softmax
        nvec_list = env.action_space.nvec.tolist()
        ares_output = []
        hefesto_output = []
        ares_probs = []
        hefesto_probs = []
        i = 0
        for x in nvec_list:
            action, action_prob = sample(ares_valid_actions[i : i + x])
            ares_output.append(action)
            ares_probs.append(action_prob.numpy())
            action, action_prob = sample(hefesto_valid_actions[i : i + x])
            hefesto_output.append(action)
            hefesto_probs.append(action_prob.numpy())

        print("Ares output shape:", len(ares_output))
        print("Hefesto output shape:", len(hefesto_output))
        #print(hefesto_output)

        end = perf_counter()
        print("Tiempo en aplicar mascaras:", end - start)
        hefesto_probs, ares_probs = self.merge_and_probs(hefesto_output, ares_output, hefesto_probs, ares_probs, reduced_obs)
        # Retornan las acciones y las probabilidades de hefesto y ares
        return hefesto_output, hefesto_probs, ares_probs


    # Aqui juntan los outputs de hefesto y ares, ademas se retornan las probabilidades de accion de ambas redes
    def merge_and_probs(self, hefesto_output, ares_output, hefesto_probs, ares_probs, reduced_obs):
        
        # Aqui guardaremos las probabilidades de las acciones para cada red
        ares_prob = 1
        hefesto_prob = 1
        
        # Obtener la source unit mask
        for i in range(len(self.ares_source_unit_mask)):
            # Si encontramos un 1 en la mascara de ares, agregarla a la de hefesto
            if self.ares_source_unit_mask == 1:
                print("Encontre una milicia en", i)
                self.hefesto_source_unit_mask[i : i + 7]   # Aqui se mergea ares con hefesto en hefesto
                # Probabilidad de accion
                accion = ares_output[7 * i]
                ares_prob *= ares_probs[7 * i]

                if ares_output[7 * i] != 0:
                    # Si la unidad de milicia ataca, multiplicar probabilidad de casilla a atacar
                    if accion == 5:
                        ares_prob *= [7 * i + 6]
                    # Si no ataca, solo le queda moverse
                    else:
                        ares_prob *= [7 * i + accion]
            
            # Ahora recorremos la mascara de hefesto
            if self.hefesto_source_unit_mask[i] == 1:
                print("Encontre un productor en", i)

                # Calculamos la probabilidad de la accion
                accion = hefesto_output[7 * i]
                hefesto_prob *= hefesto_probs[7 * i]
                
                # Si la accion es NOOP
                if hefesto_output[7 * i] != 0:
                    # Si es atacar, se multiplica por la accion 6
                    if accion == 5:
                        hefesto_prob *= hefesto_probs[7 * i + 6]
                    # Si es producir, por la accion 4 y 5
                    elif accion == 4:
                        hefesto_prob *= hefesto_probs[7 * i + accion]
                        hefesto_prob *= hefesto_probs[7 * i + accion + 1]
                    # Si es cualquier otra, por su correspondiente
                    else:
                        hefesto_prob *= hefesto_probs[7 * i + accion]
        
        # Se retornan las probabilidades de las acciones de hefesto y ares
        return hefesto_prob, ares_prob



env = MicroRTSGridModeVecEnv(
        num_selfplay_envs = 0,
        num_bot_envs = 1,
        max_steps = 2000,
        render_theme = 1,
        ai2s = [microrts_ai.coacAI for _ in range(1)],
        map_paths = ["maps/16x16/basesWorkers16x16.xml"],
        reward_weight = np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
        )


# Creamos red neuronal del agente
olimpus = Olimpus(16 * 16, env).to(DEVICE)


obs = env.reset()


action_mask = env.get_action_mask()
print("Action mask shape", action_mask.ravel().shape)

input_tensor = torch.from_numpy(obs).float().to(DEVICE)
# Obtenemos la observacion de atenea
strategy = olimpus(input_tensor).to(DEVICE)
# Y con eso, hefesto y ares mueven sus unidades
action, hefesto_probs, ares_probs = olimpus.get_action(strategy, obs, env, action_mask.ravel())

print("Prob obtenida en main:", hefesto_probs, ares_probs)
print("Action:\n")
print(action)


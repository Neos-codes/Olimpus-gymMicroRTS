import sys
import numpy as np
from parse import reduce_dim_observation
from AC import Actor, Critic, select_action
from utils import info_unit, sample, get_masks
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
print("Importando torch...")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
print("Torch importado!")
np.set_printoptions(threshold=sys.maxsize)

# Instanciar Redes Actor y Critico

# ------- IMPORTANTE ------- #
""" Cambio de planes:
    La red desde ahora por cada unidad, recibirá como input:
    - Posicion: h * 16 + w de la posicion en la grilla
    - unit type: Tipo de unidad a la que pertenece
    - Map obs: observacion del mapa dado por reset() o step()
    
    La red devuelve:
    - 78 one-hot encoding values correspondientes a las acciones posibles

    Se hará para cada unidad, es lo que se me ocurre de momento, una red
    centralizada
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

actor = Actor(16 * 16).to(DEVICE)  # Input 16x16 + 2
critic = Critic(16 * 16)

# Crear ambiente de pruebas
env = MicroRTSGridModeVecEnv(
        num_selfplay_envs = 0,
        num_bot_envs = 1,
        max_steps = 2000,
        render_theme = 1,
        ai2s = [microrts_ai.coacAI for _ in range(1)],
        map_paths = ["maps/16x16/basesWorkers16x16.xml"],
        reward_weight = np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
        )

#                   0       1           2                 3                    4                     5
# reward_weight: [ganar, harvest, produce worker, construct building, valid attack action, produce military unit]

obs = env.reset()
nvec = env.action_space.nvec
"""
print("nvec:\n", nvec)
print("shape:", nvec.shape)  # 16x16x7 = map_size * actions
print("sum:", nvec.sum())    # 16x16x78 = map_size * actions in one-hot
"""



action_mask = env.get_action_mask()
action_mask = action_mask.reshape(-1, action_mask.shape[-1])
action_mask[action_mask == 0] = -9e8

# ----- Action_mask features ----- #
#print("Action_mask shape\n", action_mask.shape)
#print("Mono:\n", action_mask[17])  # Asi se obtiene la mascara de accion de una casilla especifica 

# ----- Observation features ----- #
#print("Obs shape:\n", obs.shape)     # 1, 16, 16, 27   =  6912 valores
#print(obs[0][0][0])  # Así se obtiene la observacion de una casilla en particular
#print(obs.ravel())

# ----- Unit_source_mask features ----- #
#print(env.source_unit_mask)       # Asi se obtiene la mascara para distinguir unidades



# Hiperparametros
DISCOUNT_FACTOR = 0.99
NUM_EPISODES = 10
MAX_STEPS = 100000


# Entrenar

ep = 0
for i in range(NUM_EPISODES):
    old_observation = env.reset()                           # One-hot encoded observation
    observation = reduce_dim_observation(old_observation)   # Version simplificada de la observacion  [hw, 5]
    #print("Observation shape:", observation.shape)
    env.action_space.seed(0)
    done = False
    ep_reward = 0
    ep_loss = 0

    aux_pos = 17
    # Para cada episodio, mientras no se termine...
    while not done:
        env.render()
        
        # Actualizar action mask y source_unit_mask
        action_mask = env.get_action_mask()   # Hay que usar la [0] para efectos de masking
        # Get mascara de unidades y hacerla arreglo
        units_mask = env.source_unit_mask.ravel()    # Esta se usa solo con ravel con ningun problema
        military, workers = get_masks(units_mask, observation)
        #print(units_mask, end="\n---\n")

        
        global_action = []

        # Recorrer la mascara para encontrar unidades disponibles para ejecutar acciones
        #print(observation)
        for current_pos in range(len(units_mask)):
 

            # Si no hay unidad propia, pasar a siguiente posicion
            if units_mask[current_pos] == 0:
                global_action += [0, 0, 0, 0, 0, 0, 0]
                continue
            
            # Obtener tipo de unidad
            #info_unit(observation[0][x][y])                      # Printea toda la info de la unidad
            unit_type = observation[current_pos][3]    # Devuelve el tipo de unidad  [0 - 7]
            #print("unit_type:", unit_type)

            state_tensor = torch.from_numpy(observation.ravel()).float().to(DEVICE) 
            #print(state_tensor.shape)

            # Aplicar mascara de acciones posibles
            #action, probs = select_action(actor, state_tensor, action_mask, current_pos, env)
            ares_action, hefestus_action = select_action(actor, state_tensor, [1], current_pos, env)
            input()
            print("Action:", action)
            global_action += action

            """
            if units_mask[current_pos] == 1 and observation[current_pos][3] == 4:
                info_unit(old_observation[0][row][int(column)])
                global_action += [1, 1, 0, 0, 0, 0, 0]

            elif units_mask[current_pos] == 1 and observation[current_pos][3] == 2:
                global_action += [4, 0, 0, 0, 1, 3, 0]
                #print("Worker en fila", row, "columna",  column)
            """
        old_observation , reward, done, info = env.step(np.array(global_action))
        #print("reward:", reward)
        if reward != 0:
            print("info:\n", info)
            print("reward:", reward)
            input()
        observation = reduce_dim_observation(old_observation)
        
        # Descomentar para ir paso a paso ejecutando acciones
        #input("Enter to next step")
    

    ep += 1














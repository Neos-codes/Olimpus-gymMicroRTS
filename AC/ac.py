import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.distributions.categorical import Categorical

import argparse
from matplotlib import pyplot as plt

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

from time import perf_counter
import random
import os

torch.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)

# --- Parsear argumentos
# Crear parser
parser = argparse.ArgumentParser(description="AC Version")

# Cargar Neural Network
parser.add_argument("--load", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True, help="Set it True if you want to load a PyTorch Neural Network")

# Cargar Optimizer
parser.add_argument("--load_opt", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True, help="Set it True if you want to load a PyTorch Adam Optimizer")

# Guardar Neural Network
parser.add_argument("--save", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True, help="Set it True if you want to save the Neural Network with the highest rewards")

# Guardar optimizador
parser.add_argument("--save_opt", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True, help="Set it True if you want to save the Optimizer with the Neural Network with the highset rewards")

# Capturar Video
parser.add_argument("--capture_video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Set this flag \"True\" if you want to capture videos of the epochs")

# Set cantidad de epochs
parser.add_argument("--epochs", type=int, default=10, help="Num of epochs to train the NN")

# Set cantidad maxima de steps por epoch
parser.add_argument("--steps", type=int, default=500, help="Num of max steps per epoch")

# Procesar argumentos
args = parser.parse_args()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], sw=None):
        self.masks = masks
        #print("Mask recibida en numpy:", masks)
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            #print("Mask recibida:\n", masks)
            # EN CASO DE ERROR, BORRAR EL TO DEVICE DE ABAJO
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8, device=device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        #print("Si lees esto, la mascara es disntito de vacia!")
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)


# Agente
class Agent(nn.Module):
    def __init__(self, envs, mapsize=8*8):
        super(Agent, self).__init__()
        self.mapsize = mapsize
        self.envs = envs
        self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(128, 256)),
                nn.ReLU(), )

        self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.0)
        #print("actor output size:", mapsize, envs.action_space.nvec.sum())
        
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        return self.network(x.permute((0, 3, 1, 2)))

    def get_action(self, x, action=None, action_masks=None, envs=None):
        logits = self.actor(self.forward(x))
        #print("logits size:", logits.size())
        split_logits = torch.split(logits, self.envs.action_space.nvec.tolist(), dim=1)   #  (24, 7 * 64)   (7 actions * grid_size)  7 * 64 = 448
        
        action_mask = torch.Tensor(self.envs.get_action_mask().reshape((12, -1))).to(device)    # shape (24, 64, 78)  sin reshape
        #print("action mask in get_action:", action_mask)
        #print("action_mask shape:", action_masks.shape)
        #split_action_mask = torch.split(logits, self.envs.action_space.nvec.tolist(), dim=1)

        split_action_mask = torch.split(action_mask, self.envs.action_space.nvec.tolist(), dim=1)
        multi_categoricals = [CategoricalMasked(logits=logits, masks=ams) for (logits, ams) in zip(split_logits, split_action_mask)]
        action = torch.stack([categorical.sample() for categorical in multi_categoricals])

        # Retornar logpobs, entropia, accion y action_mask
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        #print("logprob size:", logprob.size())

        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])

        num_predicted_parameters = len(self.envs.action_space.nvec)
        logprob = logprob.T.view(-1, num_predicted_parameters)
        entropy = entropy.T.view(-1, num_predicted_parameters)
        action = action.T.view(-1, num_predicted_parameters)
        #print("nvec.sum:", self.envs.action_space.nvec.sum())
        action_mask = action_mask.view(-1, self.envs.action_space.nvec.sum())
        #print("action mask shape:", action_mask.size())
        #print("action size:", action.size())  # same num_predicted_par 448

        #print("action:\n", action.size())
        #print("logprob size:", logprob.size())
        #print("entropy size:", entropy.size())
        #print("logprob sum:", logprob.sum())
        #print("entroy sum:", entropy.sum())
        #print(logprob)
        #print(entropy)
        
        return action, logprob.sum(1).sum(), entropy.sum(1).sum(), action_mask
    

    def get_value(self, x):
        return self.critic(self.forward(x))

# Hiperparametros
num_epochs = args.epochs
lr = 2.5e-4
steps_per_episode = args.steps
num_bot_envs = 12
num_steps = 256
gamma = 0.99

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Miscelaneo
random.seed(1)
np.random.seed(1)

envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=num_bot_envs,   # 12
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(12)],
    map_paths=["maps/8x8/basesWorkers8x8.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
# envs = VecVideoRecorder(envs, 'videos', record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)

# Inicializar Red y optimizador como None para evitar bugs de declaracion de variable
agent = None
optimizer = None

# Cargar red neuronal
if args.load:
    route = input("Indique la ruta para cargar la red neuronal:")
    agent = torch.load(route)

else:
    agent = Agent(envs).to(device)


# Cargar optimizador
if args.load_opt:
    route = input("Indique la ruta para cargar el optimizador:")
    optimizer = torch.load(route)

else:
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

# Ruta para guardar al red
route = ""
route_opt = ""
if args.save:
    route = "ac_parameters.pth"

# Ruta para guardar el optimizador
if args.save_opt:
    route_opt = "ac_optim.pth" 


# Aqui guardaremos las rewards por episodio de cada ambiente
rewards_per_episode = []
time_steps_endings = []
epochs_per_env = [1] * 12             # Aqui guardaremos la cantidad de epochs finalizadas por ambiente
steps_per_env = [0] * 12              # Aqui guardamos la cantidad de pasos que lleva la epoch de cada ambiente


for i in range(12):
    rewards_per_episode.append([])    # Guardaremos en cada sublista, las recompensas de cada epoch
    time_steps_endings.append([])      # Guardaremos en cada sublista, el global step en el que termino cada epoch


global_steps = 0
graph_steps = []

max_reward = -100


envs.action_space.seed(0)
#obs = torch.Tensor(envs.reset()).to(device)

#agent.get_action(obs, envs=envs)

#print("Obs size:", obs.shape)    # (24, 8, 8, 27)
nvec = envs.action_space.nvec
#print("nvec:", nvec)   # [6, 4, 4, 4, 4, 7, 49, .....]


for epoch in range(num_epochs):
    acum_rewards = np.array([0] * 12)                           # Aqui guardaremos las recompensas obtenidas por cada epoch de cada ambiente

    obs = next_obs = torch.Tensor(envs.reset()).to(device)      # Obtener observacion inicial

    print("Epoch", epoch)
    start = perf_counter()

    #print("Epochs in envs:\n", epochs_per_env)
    print("Epochs per env sum:", sum(epochs_per_env))

    # Si llegamos a las epochs max en todos los envs... terminar y graficar    (Quick and Dirty pero funciona)
    if sum(epochs_per_env) >= 12 * num_epochs + 12:
        print("Finished!")
        break
    
    # Comienza la epoch
    for step in range(steps_per_episode):
        global_steps += 1
        envs.render()
        action_mask = envs.get_action_mask()
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])
        action_mask[action_mask == 0] = -9e8


        # Get new action
        action, logprob, entropy, masks = agent.get_action(obs, action_mask)
        state_value = agent.get_value(obs).reshape(-1)
        
        # Step en environment
        next_obs, reward, done, info = envs.step(action.cpu())
        acum_rewards = np.add(reward, acum_rewards)


       
        # Si algun ambiente termina la partida o es el ultimo step de la epoch,
        # revisamos las recompensas y vemos si guardar red
        if done.max() == 1:
            #print("Done size:", len(done), done)
            # Revisamos los ambientes que terminaron
            for i in range(len(done)):
                # Si termina, añadir punto a las listas  (agrega punto solo si no supera los epochs max)
                if done[i] == 1 and epochs_per_env[i] <= num_epochs:
                    rw = acum_rewards.tolist()[i]
                    rewards_per_episode[i].append(rw)                # Eje Y del grafico
                    time_steps_endings[i].append(global_steps)       # Eje X del grafico
                    acum_rewards[i] = 0                              # Reiniciar recompensas
                    epochs_per_env[i] += 1                           # Aumentar las epocas jugadas en 1
                    #print("Env", i, "termino epoch", epochs_per_env[i - 1], "con reward", rewards_per_episode[i])
                    print("Epochs per env sum:", sum(epochs_per_env))

            
            if sum(epochs_per_env) >= 12 * num_epochs + 12:
                print("Termino en for de steps!")
                break

            # SI SE ACABA LA EPOCH "GLOBAL" TODOS LOS ENVS SE REINICIAN Y COMIENZAN DE 0


            # ----- EXPERIMENTAL ------ #

            # Aqui no deberia entrar nunca por el momento... no se guarda la red   
            # Si decidimos guardar la red neuronal...
            if args.save:
                if total_epoch_reward_mean / num_bot_envs > max_reward:
                    #print("Saving NN")
                    torch.save(agent, route)
                if args.save_opt:
                    torch.save(optimizer, route_opt)

            # ----- END EXPERIMENTAL ---- #

            stop = perf_counter()
            #print("Time:", stop - start)
            #print("-------\n\n")
            #break

        # ----- TD optimization ----- #

        rs = torch.Tensor(reward).to(device)
        #print("Rewards shape:", rs.size())

        next_obs = torch.Tensor(next_obs).to(device) 
        new_state_value = agent.get_value(next_obs).reshape(-1).to(device)
        #print("Reward shape:", reward.shape)
        #print("new_state_value:", new_state_value.reshape(-1).shape)
        #print("State_value:", state_value.reshape(-1).shape)
        delta = rs + gamma*new_state_value*(torch.ones(len(done)).to(device) - torch.Tensor(done).to(device)) - state_value
        #print("delta:", delta)
        
        critic_loss = (delta * delta).mean()
        actor_loss = (delta * -logprob).mean()

        total_loss = critic_loss + actor_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        obs = next_obs

        #print("\n--------")

        # AYUDA MEMORIA, BORRAR
        """ rewards_per_episode = []
time_steps_endings = []
epochs_per_env = [1] * 12             # Aqui guardaremos la cantidad de epochs finalizadas por ambiente
steps_per_env = [0] * 12              # Aqui guardamos la cantidad de pasos que lleva la epoch de cada ambiente"""

for i in range(num_bot_envs):
    plt.plot(list(range(num_epochs)), rewards_per_episode[i])
    plt.xlabel("Steps in env")
    plt.ylabel("Rewards")
    plt.title("Steps vs Rewards    (" + str(num_epochs) + " epochs)")
    plt.show()

envs.close()

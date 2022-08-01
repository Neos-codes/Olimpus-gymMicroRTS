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

from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder


torch.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)


# --- Parsear argumentos
# Crear parser
parser = argparse.ArgumentParser(description="PPO Version")

# Capturar video
parser.add_argument("--capture_video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Set this flag \"True\" if you want to capture videos of the epochs")

# Procesar argumentos
args = parser.parse_args()



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AC_Batches:
    def __init__(self, batch_size, n_minibatches):
        self.batch_size = batch_size
        self.n_minibatches = n_minibatches

        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []
    
    def clean_batch(self):
        # Limpiar todas las listas del batch
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []
    

   
   
    # Guardar una observacion y una action mask
    def save_in_batch(self, obs, action, logprob, reward, value, done, action_masks):
        # Añadir al batch todos los componentes en sus respectivas listas
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_masks)



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
        
        action_mask = torch.Tensor(self.envs.get_action_mask().reshape((24, -1))).to(device)    # shape (24, 64, 78)  sin reshape
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

        #print("action:\n", action)
        #print("logprob size:", logprob.size())
        #print("entropy size:", entropy.size())
        #print("logprob sum:", logprob.sum())
        #print("entroy sum:", entropy.sum())
        #print(logprob)
        #print(entropy)
        
        return action, logprob.sum(1).sum(), entropy.sum(1).sum(), action_mask
    

    def get_value(self, x):
        return self.critic(self.forward(x))


# Obtener el gae
def compute_gae(next_value, rewards, dones, values, gamma=0.99, lam_bda=0.95):
    values = values + [next_value]
    gae = 0
    returns = []

    for step in reverse(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * dones[step] - values[step]
        gae = delta + gamma * lam_bda * dones[step] * gae
        returns.instert(0, gae + values[step])
    
    return returns



# Hiperparametros
num_epochs = 100
lr = 2.5e-4
#steps_per_episode = 100000
num_bot_envs = 24
num_steps = 256
gamma = 0.99
epsilon = 0.2
gae = 0.95

batch_size = 1024
minibatch_size = batch_size // 4
update_epochs = 4   # Por cada epoch, se actualizara 4 veces, creando 4 nuevos minibatches cada vez

steps_to_update = 100   # Cada cuantos pasos hace el update con el minibatch

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Miscelaneo
random.seed(1)
np.random.seed(1)

envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=24,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(24)],
    map_paths=["maps/8x8/basesWorkers8x8.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
# envs = VecVideoRecorder(envs, 'videos', record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)


if args.capture_video:
    envs = VecVideoRecorder(envs, "videos/AC_batches", record_video_trigger=lambda x: x == 0, video_length = 2000)


agent = Agent(envs).to(device)
batch = AC_Batches(batch_size, minibatch_size)
optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
rewards_per_episode = []
time_alive = []

envs.action_space.seed(0)
#obs = torch.Tensor(envs.reset()).to(device)

#agent.get_action(obs, envs=envs)

#print("Obs size:", obs.shape)    # (24, 8, 8, 27)
nvec = envs.action_space.nvec
#print("nvec:", nvec)   # [6, 4, 4, 4, 4, 7, 49, .....]
global_step = 0
for epoch in range(num_epochs):
    # Guardamos la observacion en numpy array para uno usar tensores
    next_obs = envs.reset()


    # Para usar la observacion en la red neuronal, la usamos como tensor
    #obs = next_obs = torch.Tensor(obs).to(device)      # Obtener observacion inicial
    total_epoch_reward_mean = 0.
    print("Epoch", epoch)
    
    start = perf_counter()
    for step in range(steps_per_episode):
        envs.render()

        obs = next_obs
        global_step += 1

        # Guardar observaciones
        action_mask = envs.get_action_mask()
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])
        print("action mask reshape:", action_mask.shape)
        action_mask[action_mask == 0] = -9e8

        # Guardar en batch observacion y action mask
        batch.save_step_data(obs, action_mask)

        # Obtenemos acciones y valores sin usar autograd
        with torch.zero_grad():
            # Convertir observacion a tensor
            obs_tensor = torch.Tensor(obs).to(device)

            # Get new action
            action, logprob, entropy, masks = agent.get_action(obs_tensor, action_mask)
            #print("Entropy:", entropy, "  logprob:", logprob)
            state_value = agent.get_value(obs).reshape(-1)
            
            next_obs, reward, done, info = envs.step(action.cpu())
            total_epoch_reward_mean += reward.sum()

            # Aqui debe hacerse el save in batch
            batch.save_in_batch(obs, action, logprob, entropy, sate_value, done, action_mask)

        

    # Cuando se llena el batch, dividir en minibatches y actualizar
    # Antes de obtener el minibatch, hay que obtener los valores de Ventaja con el GAE y todo eso
    # No necesitamos tener el grafico de autograd
    with torch.no_grad():
        last_value = agent.get_value(torch.Tensor(next_obs).to(device)).reshape(-1)   # Tensor del ultimo estado al terminar la epoch
        print("Last value shape:", last_value.size())
        # Obtener los retornos
        batch.values = batch.values + [last_value]   # Añadir el ultimo valor al batch
        gae = 0
        returns = []

        for ret_step in reversed(range(len(batch.rewards))):
            delta = batch.rewards[ret_step] + gamma*batch.values[ret_step + 1] * batch.dones[ret_step] - batch.values[ret_step]
            print("Rewards type:", type(batch.rewards[ret_step]))
            print("Values type:", type(values[ret_step]))
            print("Dones type:", type(batch.dones[ret_step]))
            print("Delta shape:", delta.shape)

            gae = delta + gamma*batch.dones*gae
            returns.insert(0, gae + batch.values[ret_step])   # Con el insert vamos agregando al inicio
                                                              # asi no es necesario hacer reverse al return


        
    # Actualizar red si se cumple con los pasos
    # Obtener minibatches
    # Aqui si necesitamos el grafico de autograd, por lo que debemos usar todo en tensores!
    # TO DO: MODIFICAR BATCH PARA QUE LAS LISTAS SEAN TENSORES (con torch.zeros)
    inds = np.arange(batch_size)

    for e in range(update_epochs):
        random.shuffle(inds)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_ind = inds[start:end]
             
            



            # Despues de actualizar, hay que ver si debemos reiniciar los ambientes
            if done.max() == 1:
                #print("Un ambiente termino!")
                #print("Total reward:", total_epoch_reward_mean / num_bot_envs)
                rewards_per_episode.append(total_epoch_reward_mean / num_bot_envs)
                stop = perf_counter()
                time_alive.append(stop-start)
                print("Reward:", total_epoch_reward_mean / num_bot_envs)
                print("Time:", stop - start)
                break

    #print("\n--------")
       
fig = plt.figure()
gs = fig.add_gridspec(2, hspace = 2)
axs = gs.subplots(sharex = True, sharey = False)
fig.suptitle("Recompensa y tiempo")
axs[0].plot(rewards_per_episode)
axs[1].plot(time_alive)
plt.plot(rewards_per_episode)
plt.show()
input()
envs.close()

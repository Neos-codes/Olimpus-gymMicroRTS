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

parser.add_argument("--num_steps", type=int, default=512, help="Num of max steps per epoch")

parser.add_argument("--epochs", type=int, default=10, help="Num of epochs to train")

parser.add_argument("--envs", type=int, default=12, help="Num of envs to train")

parser.add_argument("--step_graph", type=int, default=100, help="Cada cuantas epocas muestras el grafico \"epocas vs recompensas\"")

# Procesar argumentos
args = parser.parse_args()



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AC_Batches:
    def __init__(self, n_envs, h, w, batch_size, n_minibatches):
        self.batch_size = batch_size
        self.n_minibatches = n_minibatches
        self.n_envs = n_envs

        self.obs = torch.zeros((batch_size, n_envs, h, w, 27)).to(device)          # shape: [batch_size, n_envs, h, w, 27]
        self.actions = torch.zeros((batch_size, n_envs, 7*h*w)).to(device)         # shape: [batch_size, n_evns, 7*h*w]
        self.logprobs = torch.zeros((batch_size, n_envs)).to(device)               # shape: [batch_size, n_envs]
        self.rewards = torch.zeros((batch_size, n_envs)).to(device)                # shape: [batch_size, n_envs]
        self.values = torch.zeros((batch_size, n_envs)).to(device)                 # shape: [batch_size, n_envs]
        self.dones = torch.zeros((batch_size, n_envs)).to(device)                  # shape: [batch_size, n_envs]
        self.action_masks = np.zeros((batch_size, n_envs*h*w, 78))                 # shape: [batch_size, n_envs*h*w, 78]  (en numpy array)
    
      
    # Guardar una observacion y una action mask
    def save_in_batch(self, step, obs, action, logprob, reward, value, done, action_masks):
        # Añadir al batch todos los componentes en sus respectivas listas
        self.obs[step] = obs
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.rewards[step] = reward
        self.values[step] = value
        self.dones[step] = done
        self.action_masks[step] = action_masks
        #print("Action masks shape:", action_masks.shape)
        #print("Action mask saved:", self.action_masks[step])

        #print("In batch:")
        #print("obs shape:", obs.size())
        #print("action shape:", action.size())
        #print("logprob shape:", logprob.size())
        #print("reward shape:", reward.size())
        #print("value shape:", value.size())
        #print("dones shape:", done.size())
        #print("action_mask batch shape:", self.action_masks.shape)
        #print("action mask shape:", action_masks.shape)



class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], sw=None, show_mask=False):
        self.masks = masks
        #print("Mask recibida en numpy:", masks)
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            if sw:
                print(self.masks)
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
        # Siempre entra x (obs) de shape [n_envs, h, w, 27]
        logits = self.actor(self.forward(x))    # [n_envs, 78hw]
        split_logits = torch.split(logits, self.envs.action_space.nvec.tolist(), dim=1)   #  (24, 7 * hw)   (7 actions * grid_size)  7 * 64 = 448
        # Si se da la action mask, siempre es de shape [12*hw, 78]

        # La mascara cuando la accion es None, debe tener shape  [num_envs, 4992]
        if action == None:
            action_mask = torch.Tensor(self.envs.get_action_mask().reshape((num_envs, -1))).to(device)    # shape (24, hw, 78)  sin reshape

            split_action_mask = torch.split(action_mask, self.envs.action_space.nvec.tolist(), dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=ams) for (logits, ams) in zip(split_logits, split_action_mask)]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])

        # La mascara cuando la accion es dada, debe tener shape [minibatch_size, 4992]
        else:
            action = action.view(-1, action.size()[-1]).T    # Trasponer accion para calzar con action mask
            # NO DEBO CAMBIAR LOS 0'S POR -9E8!!!!!
            action_mask = torch.Tensor(action_masks).reshape((-1, self.envs.action_space.nvec.sum())).to(device)  # [num_envs, 4992]
            #print("Action shape:", action.size())
            #print("Action mask shape:", action_mask.size())
            split_action_mask = torch.split(action_mask, self.envs.action_space.nvec.tolist(), dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_action_mask)]
            # LA MASCARA SE ESTA APLICANDO MAL EN  MULTI_CATEGORICALS!!!


        # Retornar logpobs, entropia, accion y action_mask
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])

        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])

        num_predicted_parameters = len(self.envs.action_space.nvec)
        logprob = logprob.T.view(-1, num_predicted_parameters)
        entropy = entropy.T.view(-1, num_predicted_parameters)
        action = action.T.view(-1, num_predicted_parameters)
        #print("nvec.sum:", self.envs.action_space.nvec.sum())
        action_mask = action_mask.view(-1, self.envs.action_space.nvec.sum())
        #print("action mask shape:", action_mask.size())
        #print("action size:", action.size())  # same num_predicted_par 448


        #print("logprob shape:", logprob.sum(1).size())     # [12]  logprob por ambiente
        #print("entropy shape:", entropy.sum(1).size())
        #print("logprob of this action:", logprob.sum(1))    # logprob.sum(1).sum()
        
        return action, logprob.sum(1), entropy.sum(1), action_mask
    

    def get_value(self, x):
        return self.critic(self.forward(x))



# Hiperparametros
num_epochs = args.epochs
lr = 2.5e-4
#steps_per_episode = 100000
num_envs = args.envs
num_steps = args.num_steps    # 512 por defecto
gamma = 0.99
epsilon = 0.2
gae = 0.95

batch_size = num_steps   # De momento el batch_size y num_steps son lo mismo
n_minibatches = 4
minibatch_size = batch_size // n_minibatches     # El batch se divide en "n_minibatches" minibatches
update_epochs = 4   # Por cada epoch, se actualizara 4 veces, creando 4 nuevos minibatches cada vez


# Respecto al grafico
rewards_per_episode = []



# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Miscelaneo
random.seed(1)
np.random.seed(1)

envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=num_envs,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(num_envs)],
    map_paths=["maps/8x8/basesWorkers8x8.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
# envs = VecVideoRecorder(envs, 'videos', record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)


if args.capture_video:
    envs = VecVideoRecorder(envs, "videos/AC_batches", record_video_trigger=lambda x: x == 0, video_length = 2000)


agent = Agent(envs).to(device)
batch = AC_Batches(num_envs, 8, 8, batch_size, n_minibatches)    # n_envs, h, w, batch_size, n_minibatches

# Shape del action space
action_space_shape = (envs.action_space.shape[0],)
action_mask_shape = (envs.action_space.nvec.sum(),)

optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
rewards_per_episode = []
time_alive = []

envs.action_space.seed(0)
#obs = torch.Tensor(envs.reset()).to(device)

#agent.get_action(obs, envs=envs)

#print("Obs size:", obs.shape)    # (24, 8, 8, 27)
nvec = envs.action_space.nvec
#print("nvec:", nvec)   # [6, 4, 4, 4, 4, 7, 49, .....]

# Cuantas epocas vamos a entrenar
for epoch in range(num_epochs):

    # Guardamos la observacion en numpy array para uno usar tensores
    next_obs = envs.reset()
    next_dones = torch.zeros(num_envs).to(device)
    ep_reward = 0
    ep_done = [0] * num_envs

    # Para usar la observacion en la red neuronal, la usamos como tensor
    #obs = next_obs = torch.Tensor(obs).to(device)      # Obtener observacion inicial
    total_epoch_reward_mean = 0.
    print("Epoch", epoch)
    
    start = perf_counter()
    for step in range(num_steps):
        envs.render()

        obs = torch.Tensor(next_obs).to(device)
        #print("obs shape:", obs.size())

        # Guardar observaciones
        action_mask = envs.get_action_mask()
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])   # Este reshape es para establecer todos los 0 como -9e8  [n_envs*hw, 78]
        #action_mask[action_mask == 0] = -9e8

        #print("Action mask main shape:", action_mask.shape)
        #print("Action maks reshaped:", action_mask.reshape((num_envs, -1)).shape)
        #print("Action mask main:", action_mask)

        # Obtenemos acciones y valores sin usar autograd
        with torch.no_grad():
            # Get new action
            action, logprob, entropy, masks = agent.get_action(obs, action_masks=action_mask)
            #print("Entropy:", entropy, "  logprob:", logprob)
            state_value = agent.get_value(obs).reshape(-1)
            

        next_obs, reward, done, info = envs.step(action.cpu())
        dones = torch.Tensor(done).to(device)
        rs = torch.Tensor(reward).to(device)
        total_epoch_reward_mean += reward.sum()

        print("Reward type:", type(reward))
        ep_reward += reward

        # Aqui debe hacerse el save in batch
        # Action mask before save shape: [n_envs*hw, 78]
        batch.save_in_batch(step, obs, action, logprob, rs, state_value, next_dones, action_mask)

        

    # AQUI ACABO EL EPISODIO!
    rewards_per_episode.append(ep_reward)
    # Cuando se ejecuten la cantidad de pasos maxima por epoch, es decir, num_steps pasos...
    # el batch se llena y por lo tanto, dividir en minibatches y actualizar
    # Antes de obtener el minibatch, hay que obtener los valores de Ventaja con el GAE y todo eso

    # No necesitamos tener el grafico de autograd para calcular el ultimo valor ni los valores de ventaja y retorno
    with torch.no_grad():
        last_value = agent.get_value(torch.Tensor(next_obs).to(device)).reshape(-1)   # Tensor del ultimo estado al terminar la epoch
        advantages = torch.zeros_like(batch.rewards).to(device)
        print("Last value shape:", last_value.size())
        # Obtener los retornos
        #batch.values = batch.values + [last_value]   # Añadir el ultimo valor al batch
        last_gae = 0
        #returns = []

        for ret_step in reversed(range(len(batch.rewards))):
            if ret_step == num_steps - 1:
                nextnonterminal = 1.0 - dones
                delta = batch.rewards[ret_step] + gamma * gae * last_value * nextnonterminal - batch.values[ret_step]
            else:
                nextnonterminal = 1.0 - batch.dones[ret_step]
                delta = batch.rewards[ret_step] + gamma * gae * batch.values[ret_step + 1] * nextnonterminal - batch.values[ret_step]
            """print("Rewards type:", type(batch.rewards[ret_step]))
            print("Values type:", type(values[ret_step]))
            print("Dones type:", type(batch.dones[ret_step]))
            print("Delta shape:", delta.shape)"""

            # Obtenemos el valor de ventaja
            advantages[ret_step] = last_gae = delta + gamma * gae * nextnonterminal * last_gae
            #returns.insert(0, gae + batch.values[ret_step])   # Con el insert vamos agregando al inicio
        # Obtenemos los valores de retorno
        returns = advantages + batch.values    # return = advantage + critic_value


    # Cuando se obtienen, actualizar
    #exit()
    print("----- ACTUALIZANDO! -----")
        
    # Copiando aqui lo del batch del paper para ver dimensiones
    b_obs = batch.obs.reshape((-1,)+envs.observation_space.shape)
    b_logprobs = batch.logprobs.reshape(-1)
    b_actions = batch.actions.reshape((-1,)+action_space_shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_action_masks = batch.action_masks.reshape((-1,)+action_mask_shape)   # EL PROBLEMA DE DIMENSIÓN ESTÁ AQUI

    """print("Sobre el batch del paper:")
    print("obs:", b_obs.size())                    # [b_size*n_envs, h, w, 27] 
    print("logprobs:", b_logprobs.size())          # [b_size*n_envs*hw*7]
    print("logprobs:", b_logprobs)
    print("actions:", b_actions.size())            # [b_size*n_envs, hw*7]     [240, 448]
    print("advantages:", b_advantages.size())      # [b_size*envs]
    print("returns:", b_returns.size())            # [b_size*envs]
    print("action_masks:", b_action_masks.shape)   # [b_size*n_envs, hw*78]    [240, 4992]

    print("----------\n\n")"""
    # Actualizar red si se cumple con los pasos
    # Obtener minibatches
    # Aqui si necesitamos el grafico de autograd, por lo que debemos usar todo en tensores!
    inds = np.arange(batch_size)

    for e in range(update_epochs):
        random.shuffle(inds)

        for start in range(0, batch_size, minibatch_size):    # desde 0 hasta batch_size con pasos del tamano del minibatch
            end = start + minibatch_size
            minibatch_ind = inds[start:end]   # Indices correspondientes a los estados del minibatch
            mb_advantages = b_advantages[minibatch_ind]
            #print("mb_advantages shape:", mb_advantages.size())

            #print("b_obs minibatch shape:", b_actions[minibatch_ind].shape)
            #print("b_action_masks minibatch shape:", b_action_masks[minibatch_ind].shape)

            _, newlogprobs, newentropy, _ = agent.get_action(b_obs[minibatch_ind], b_actions[minibatch_ind], b_action_masks[minibatch_ind], envs)

            ratio = (newlogprobs - b_logprobs[minibatch_ind]).exp()
            #print("newlogprobs:", newlogprobs.exp())
            #print("b_logprobs minibatch:", b_logprobs[minibatch_ind].exp())
            #print("Ratio:", ratio)
            
            # Loss
            loss_1 = -mb_advantages * ratio
            loss_2 = -mb_advantages * torch.clamp(ratio, 1 - epsilon, 1 + epsilon)   # Clip el ratio
            pg_loss = torch.max(loss_1, loss_2).mean()
            entropy_loss = newentropy.mean()

            # New values of Critic
            new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
            # Critic loss
            critic_loss = ((new_values - b_returns[minibatch_ind])**2).mean()

            # Total loss
            loss = pg_loss - 0.01 * entropy_loss + 0.5*critic_loss*0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    """if epoch % args.step_graph == 0:
        plt.plot(list(range(epoch+1)), rewards_per_episode)
        plt.xlabel("Epochs")
        plt.ylabel("Rewards")
        plt.title("Epochs vs Rewards      (" + str(num_epochs) + " epochs)")
        plt.show()"""
    #print("\n--------")
       
plt.plot(list(range(num_epochs)), rewards_per_episode)
plt.xlabel("Epochs")
plt.ylabel("Rewards")
plt.title("Epochs vs Rewards      (" + str(num_epochs) + " epochs)")
plt.show()
input()
envs.close()

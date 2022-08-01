import numpy as np
import torch
from torch.distributions import OneHotCategorical


def softmax(x, axis=None):
    """ Recibe un np.array en one-hot encoding y devuelve
    las probabilidades al aplicarle softmax"""
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def sample(logits):
    """ Recibe un np.array obtiene probabilidades softmax y devuelve su posicion"""
    # https://stackoverflow.com/a/40475357/6611317
    p = softmax(logits, axis=None)
    # Escogemos una accion basada en la probabilidad
    m = OneHotCategorical(torch.from_numpy(p))
    m_action = m.sample()
    action =  np.argmax(m_action.numpy())  # int
    action_log = m.log_prob(m_action)      # tensor


    #choice = (u < c).argmax(axis=None)

    return action, action_log

def utype(oh_obs):

    return np.argmax(oh_obs[13:21]) 

def separate_su_mask(su_mask, obs, h, w):      # su_mask shape  (1, hw)
    """ Recieves the source unit mask and returns 2 source unit mask, 1 for military units and another for product units"""
    
    obs_ = obs[0].reshape((h*w, -1))

    milit = np.zeros(h*w)
    prod = np.zeros(h*w)
    
    for i in range(len(su_mask[0])):
        unit_type = utype(obs_[i])

        if unit_type > 4:   # Si es militar
            milit[i] = 1

        elif unit_type > 1: # Si es productora (no recurso)
            prod[i] = 1

    
    #print("prod:\n", prod.reshape((h, w)))
    #print("milit:\n", milit.reshape((h, w)))

    # Retornar 2 mascaras y banderas si hay productoras y/o milicias
    return prod, milit, np.max(prod) > 0, np.max(milit) > 0







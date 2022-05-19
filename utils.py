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


def get_unit_masks(units_mask, observation):
    """
    Args:
    units_mask (ravel): Arreglo de 0s y 1s que indica donde hay una unidad en el mapa.
    observation: Observation reducida (no en one-hot) del mapa, para poder obtener informacion de el
    """
    military = []
    workers = []

    for i in range(len(units_mask)):

        # Si no hay nada en la posicion, ambas mascaras tienen 0
        if units_mask[i] == 0:
            military.append(0)
            workers.append(0)
            continue

        # Si la unidad que hay es milicia, 1 en military 0 en workers
        if observation[i][3] > 4:
            print("Milicia en pos", i, "de tipo", observation[i][3])
            military.append(1)
            workers.append(0)
            continue

        # Si es otra unidad, 1 en workers y 0 en military
        print("Productor en pos", i, "de tipo", observation[i][3])
        military.append(0)
        workers.append(1)
    
    #print("Milicia:\n", military)
    #print("Workers:\n", workers)

    return military, workers


def info_unit(observation: np.array):
    """ Traduce la observacion en one-hot encoding a formato rapido legible"""

    print("Unit Type:", info_unit_type(np.argmax(observation[13:21])))
    print("HP:", np.argmax(observation[0:5]), end="")
    print(" Resources:", np.argmax(observation[5:10]), end="")
    print(" Owner:", np.argmax(observation[10:13]), end="")
    print(" Curr Action:", np.argmax(observation[21:]), end="\n\n")


def info_unit_type(arg :int):
    """ Recibe el numero del tipo de una unidad y retorna el string del tipo de la unidad"""

    if arg == 0:
        return "-"
    elif arg == 1:
        return "Resource"
    elif arg == 2:
        return "Base"
    elif arg == 3:
        return "Barrack"
    elif arg == 4:
        return "Worker"
    elif arg == 5:
        return "Light"
    elif arg == 6:
        return "Heavy"
    else:
        return "Range"

""" Funcion que inicializa una capa de red para evitar
que caiga en gradiente desvaneciente"""
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # Esto sirve para evitar gradiente desvaneciente
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



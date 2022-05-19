import numpy as np
from time import perf_counter

def reduce_dim_observation(observation):

    new_observation = []

    for row in observation[0]:
        for column in row:
            new_observation.append([
                    np.argmax(column[0:5]),
                    np.argmax(column[5:10]),
                    np.argmax(column[10:13]),
                    np.argmax(column[13:21]),
                    np.argmax(column[21:27])
                    ])
    
    return np.array(new_observation)
    

"""Aplica mascara de accion para cada unidad en el mapa
   tomando en cuenta la salida de la red usando el tipo de unidad
   Args:
   - output: Salida de la red neuronal (Ares o Hefesto)
   - units_mask: Mascara con 0 o 1 que contiene las unidades correspondientes a Ares o Hefesto
   output:
   - output_masked: Salida de la red neuronal (Ares o Hefesto) descartando las unidades que no corresponden al tipo de unidad que maneja la red"""
 
def valid_action_mask(actions, units_mask, action_mask):

    # Creamos un arreglo del tamaño ed las acciones solo con 0's
    valid_actions = np.zeros(len(actions))
    # Recorrer la units_mask en busca de 1's
    for i in range(len(units_mask)):
        if units_mask[i] == 1:
            for j in range(78):
                if action_mask[78 * i + j] == 1:
                    valid_actions[78 * i + j] = actions[78 * i + j]

    valid_actions[valid_actions == 0] = 9e-8
    #print(valid_actions)
    return valid_actions













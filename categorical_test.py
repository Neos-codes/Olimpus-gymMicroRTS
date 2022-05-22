import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F



x = np.arange(5, 21, 3, dtype = float)
logits = torch.from_numpy(x.reshape((2, -1)))
mask = torch.from_numpy(np.array([0, 1, 0, 1, 0, 1]).reshape(2, -1))

print("Actions:", x)
print("Mask:", mask)

mask = mask.type(torch.BoolTensor)
print("Mask tensor:", mask)

mask_value = torch.finfo(logits.dtype).min
logits.masked_fill_(~mask, mask_value)


print(logits)





probs = F.softmax(logits, dim = 0)

m = Categorical(probs)

action = m.sample()

print(m)
print(action)


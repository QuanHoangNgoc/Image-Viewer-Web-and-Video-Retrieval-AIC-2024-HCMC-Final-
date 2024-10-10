import numpy as np
import torch


image = np.load("image.npy")
text = np.load("text.npy")

for t in text:
    t = torch.from_numpy(t)

    for i in image:
        i = torch.from_numpy(i)

        # calculate L2 distance
        dist = torch.norm(t - i, p=2)
        print(dist)
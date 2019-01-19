from sklearn import datasets
import numpy as np

x, _ = datasets.make_moons(n_samples=4000, noise=0.1, random_state=42)
np.save('data.npy', x)

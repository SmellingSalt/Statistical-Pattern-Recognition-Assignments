import numpy as np
from scipy.stats import dirichlet
quantiles = np.array([0.1, 0.3, 0.6])  # specify quantiles
alpha = np.array([0.4, 5, 15])  # specify concentration parameters
print(dirichlet.pdf(quantiles, alpha))
import numpy as np
from MF import MF
from preprocessing import Rating_Matrix

import numpy as np

R = Rating_Matrix("matrix-factorization/data/ratings.dat").preprocessing()

mf = MF(R, K = 30, alpha = 0.001, beta = 0.01, iterations = 20)
training_process = mf.train()

print(training_process)
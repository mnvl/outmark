
import numpy as np

def accuracy(a, b):
  return np.mean((a == b).astype(np.float32))

def dice(a, b):
  assert a.shape == b.shape

  a = a.reshape(-1)
  b = b.reshape(-1)

  a_nonzero = (a != 0).astype(np.float32)
  b_nonzero = (b != 0).astype(np.float32)

  nominator = (a == b).astype(np.float32)
  nominator = np.multiply(nominator, a_nonzero)
  nominator = np.multiply(nominator, b_nonzero)
  nominator = 2. * np.sum(nominator)
  denominator = 1.0
  denominator = np.add(denominator, np.sum(a_nonzero))
  denominator = np.add(denominator, np.sum(b_nonzero))

  return nominator / denominator

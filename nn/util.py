
import numpy as np

def accuracy(a, b):
  return np.mean((a.flatten() == b.flatten()).astype(np.float32))

def dice(a, b):
  assert a.shape == b.shape

  a = a.flatten()
  b = b.flatten()

  a_nonzero = (a != 0).astype(np.float32)
  b_nonzero = (b != 0).astype(np.float32)

  nom = (a == b).astype(np.float32)
  nom = np.multiply(nom, a_nonzero)
  nom = np.multiply(nom, b_nonzero)
  nom = 2. * np.sum(nom)

  denom = 1.0
  denom = np.add(denom, np.sum(a_nonzero))
  denom = np.add(denom, np.sum(b_nonzero))

  return nom / denom

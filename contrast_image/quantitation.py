import math
import numpy as np

# Absolute Mean Brightness Error
def AMBE(image_input, image_output):
  return abs(np.mean(image_input) - np.mean(image_output))

# Mean Square Error
def MSE(image_input, image_output):
  err = np.sum((image_input.astype("float") - image_output.astype("float")) ** 2)
  err /= float(image_input.shape[0] * image_input.shape[1])

  return err

# Peak Signal to Noise Ratio
def PSNR(image_input, image_output):
  return 10 * math.log10(255 * 255 / MSE(image_input, image_output))

# Entropy
def Entropy(image_output):
  pdf, _ = np.histogram(image_output, 256, [0, 255])
  pdf = pdf / float(image_output.shape[0] * image_output.shape[1])

  ent = 0
  for probility in pdf:
    ent += probility * math.log2(probility)
  
  return -ent
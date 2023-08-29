## norm function that calculates the norm ##

import numpy as np
import math

# norm function 
def norm(x):
	result = math.sqrt(np.sum(x**2))
	return result

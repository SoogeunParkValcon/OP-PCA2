## H function that performs the thresholding operator for each iterate ##

import numpy as np
import math
from norm_function import norm

def H(Phi, w_old, lambda_value):
	
	fraction = np.matmul(Phi, w_old) / norm(np.matmul(Phi, w_old))

	to_check = (np.matmul(Phi, w_old))**2 / norm(np.matmul(Phi, w_old))

	threshold = to_check <= (lambda_value / 2)

	fraction[threshold] = 0

	return fraction


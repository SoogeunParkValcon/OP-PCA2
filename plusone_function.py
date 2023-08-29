import sys
import pandas as pd
import os
import numpy as np
import math
from sklearn import datasets # to import the iris dataset

from norm_function import norm
from H_function import H


def plus1_sigma (sigma, w_old, lambda_value, maxiter, tol):

	Phi =  sigma - np.diag(np.repeat(1, sigma.shape[1]))

	obj_orig_old = np.transpose(w_old) @ sigma @ w_old - lambda_value * np.sum(w_old != 0)

	# pre-defining the objects to be filled with the iterates
	obj_orig_hist = np.repeat(float(-1), maxiter)

	W_hist = np.ones((sigma.shape[1], maxiter)) * -1

	obj_orig_hist[0] = obj_orig_old

	W_hist[: , 0] = w_old

	diff_norm_hist = np.repeat(float(-1), maxiter)

	iii = 0

	while True:

		w_new = H(Phi, w_old, lambda_value)

		# if zero vector is resulted, stop the iteration
		if np.sum(w_new == 0) != w_new.shape[0]:
			w_new = w_new / norm(w_new)
		else:
			print ("zero vector resulted")
			break

		# if max iteration is reached, stop the iteration
		if iii > 2 and iii > maxiter:
		  	print("max iteration reached")
		  	break

		# comparing the norm between w_old and w_new to see if stop:
		if iii > 2 and norm(w_old - w_new) < tol:
			print ("w norm condition suffices")

			iii = iii + 1

			obj_orig_new = np.transpose(w_new) @ sigma @ w_new - lambda_value * np.sum(w_new != 0)

			obj_orig_hist[iii] = obj_orig_new

			diff_norm_hist[iii] = norm(w_old - w_new)

			W_hist[:, iii] = w_new

			break

		iii = iii + 1

		obj_orig_new = np.transpose(w_new) @ sigma @ w_new - lambda_value * np.sum(w_new != 0)

		obj_orig_hist[iii] = obj_orig_new

		diff_norm_hist[iii] = norm(w_old - w_new)

		W_hist[:, iii] = w_new

		if obj_orig_new - obj_orig_old < 0:
			print ("original objective decrease at " + str(iii))


		# updating the w iterate
		w_old = w_new

		# updating the objectives
		obj_orig_old = obj_orig_new

	# reporting only the relevant results in the history objects
	obj_orig_hist = obj_orig_hist[:(iii+1)]
	diff_norm_hist = diff_norm_hist[:(iii+1)]
	W_hist = W_hist[:, :(iii + 1)]

	results = dict();
	results["w_new"] = w_new    	
	results["W_hist"] = W_hist
	results["obj_hist"] = obj_orig_hist
	results["obj"] = obj_orig_new
	results["diff_norm"] = diff_norm_hist
	results["i"] = iii

	return results


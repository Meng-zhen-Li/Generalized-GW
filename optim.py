# -*- coding: utf-8 -*-
"""
Generic solvers for regularized OT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#		 Titouan Vayer <titouan.vayer@irisa.fr>
#
# License: MIT License

import numpy as np
import ot
from scipy.sparse import csr_matrix, eye
from scipy.optimize import fminbound


def solve_linesearch(cost, G, deltaG, f_val, D1=None, D2=None, Eb=None, constC=None, tau_min=None, tau_max=None):
	
	# dot = np.dot(np.dot(D1, deltaG), D2)
	dot = csr_matrix(D1.dot(deltaG).dot(D2))
	a = -2 * dot.multiply(deltaG).sum() + Eb.multiply(deltaG).multiply(deltaG).sum()
	b = csr_matrix(constC).multiply(deltaG).sum() - 2 * (dot.multiply(G).sum() + D1.dot(G).dot(D2).multiply(deltaG).sum()) + 2 * Eb.multiply(G).multiply(deltaG).sum()
	c = cost(G)

	tau = solve_1d_linesearch_quad(a, b, c)
	if tau_min is not None or tau_max is not None:
		tau = np.clip(tau, tau_min, tau_max)
	f_val = cost(G + tau * deltaG)

	return tau, f_val


def cg(a, b, f, df, G0=None, num_overlap=0, numItermax=250, numItermaxEmd=2000000,
	   stopThr=1e-10, stopThr2=1e-10, verbose=False, log=False, **kwargs):
	loop = 1
	
	if log:
		log = {'loss': []}

	if G0 is None:
		G = csr_matrix(np.outer(a, b))
	else:
		G = G0

	def cost(G):
		return f(G)
	
	f_val = cost(G)
	
	if log:
		log['loss'].append(f_val)

	it = 0
	
	if verbose:
		print('{:5s}|{:12s}|{:8s}|{:8s}'.format('It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
		print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, 0, 0))

	while loop:
		it += 1
		old_fval = f_val

		# problem linearization
		Mi = np.array(df(G).todense())
		# Mi = Mi + np.min(Mi)

		# solve linear program
		Gc, logemd = ot.emd(a, b, Mi, numItermax=numItermaxEmd, log=True)
		
		deltaG = csr_matrix(Gc - G)

		# line search
		tau, f_val = solve_linesearch(cost, G, deltaG, f_val, tau_min=0., tau_max=1., **kwargs)
		G = G + deltaG.multiply(tau)

		# test convergence
		if it >= numItermax:
			loop = 0

		abs_delta_fval = abs(f_val - old_fval)
		relative_delta_fval = abs_delta_fval / abs(f_val)
		if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
			loop = 0

		if log:
			log['loss'].append(f_val)
			
		if verbose:
			if it % 20 == 0:
				print('{:5s}|{:12s}|{:8s}|{:8s}'.format('It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
			print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, relative_delta_fval, abs_delta_fval))

	if log:
		log.update(logemd)
		return G, log
	else:
		return G


def solve_1d_linesearch_quad(a, b, c):

	f0 = c
	df0 = b
	f1 = a + f0 + df0

	if a > 0:  # convex
		minimum = min(1, max(0, np.divide(-b, 2.0 * a)))
		return minimum
	else:  # non convex
		if f0 > f1:
			return 1
		else:
			return 0
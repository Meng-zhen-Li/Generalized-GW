import numpy as np
import ot
from optim import cg
from scipy.sparse import csr_matrix

def init_matrix(D1, D2, p, q):

	def f1(a):
		return (a**2)

	def f2(b):
		return (b**2)

	def h1(a):
		return a

	def h2(b):
		return 2 * b

	constC1 = f1(D1).dot(np.reshape(p, (-1, 1))).dot(np.ones((1, len(q))))
	constC2 = (f2(D2).dot(np.reshape(q, (-1, 1))).dot(np.ones((1, len(p))))).T
	
	constC = constC1 + constC2
	hC1 = h1(D1)
	hC2 = h2(D2)

	return constC, hC1, hC2


def tensor_product(constC, hC1, hC2, T):
	A = - hC1.dot(T).dot(hC2.T)
	tens = constC + A
	return csr_matrix(tens)


def new_loss(B1, B2, T, num_overlap):
	# (B1[i, k] - B2[i, l])^2 = B1[i, k]^2 + 2B1[i, k]B2[i, l] + B2[i, l]^2
	Eb1 = csr_matrix((B1.multiply(B1)).sum(axis=0))
	Eb1 = (csr_matrix(np.ones([T.shape[1],1])) * Eb1).transpose()
	
	Eb2 = -2 * B1.transpose().dot(B2)
	
	Eb3 = csr_matrix((B2.multiply(B2)).sum(axis=0))
	Eb3 = csr_matrix(np.ones([T.shape[0],1])) * Eb3
	
	return Eb1 + Eb2 + Eb3

def gwloss(constC, hC1, hC2, Eb, T, alpha):
	tens = tensor_product(constC, hC1, hC2, T)
	
	square = T.multiply(T)
	
	Eb = Eb.multiply(square)

	return ((1 - alpha) * tens.multiply(T) + alpha * Eb).sum()
	


def gwggrad(constC, hC1, hC2, Eb, T, alpha):
	L = (1 - alpha) * tensor_product(constC, hC1, hC2, T) + alpha * Eb.multiply(T)
	return 2 * L


def gromov_wasserstein(D1, D2, B1, B2, p, q, num_overlap, alpha=0.5, log=False, **kwargs):
	constC, hC1, hC2 = init_matrix(D1, D2, p, q)
	
	G0 = csr_matrix(p[:, None] * q[None, :])
	Eb = new_loss(B1, B2, G0, num_overlap)
	print('initialization finished!')


	def f(G):
		return gwloss(constC, hC1, hC2, Eb, G, alpha)

	def df(G):
		return gwggrad(constC, hC1, hC2, Eb, G, alpha)

	if log:
		res, log = cg(p, q, f, df, G0, num_overlap, log=True, D1=D1, D2=D2, Eb=Eb, constC=constC, alpha=alpha, **kwargs)
		log['gw_dist'] = gwloss(constC, hC1, hC2, Eb, res, alpha)
		log['u'] = log['u']
		log['v'] = log['v']
		return res, log
	else:
		return cg(p, q, f, df, G0, D1=D1, D2=D2, B1=B1, B2=B2, constC=constC, log=False, **kwargs)

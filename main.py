from scipy.io import loadmat, savemat
from preprocess import process_pairs
from itertools import combinations
import gromov
import ot
import numpy as np
import networkx as nx
import argparse
from scipy.sparse import csr_matrix

def main():
	
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('filename', type=str, help='filename of the input data')
	parser.add_argument('--output', type=str, default='data/result.mat', help='filename of the output')
	parser.add_argument('--preprocess', type=bool, default=False, help='whether or not preprocess')
	args = parser.parse_args()
	
	data = loadmat(args.filename)
	
	if args.preprocess:
		data = list(data.values())
		if len(data) == 6:
			pair = process_pairs(data[-3], data[-2], Y=data[-1])
		else:
			pair = process_pairs(data[-2], data[-1])
	else:
		pair = {'C1': data['C1'], 'C2': data['C2'], 'num_overlap': int(data['num_overlap'][0][0])}

	C1 = pair['C1']
	C2 = pair['C2']
	num_overlap = pair['num_overlap']
	p = ot.unif(C1.shape[0] - num_overlap)
	q = ot.unif(C2.shape[0] - num_overlap)
	
	idx1 = np.array(range(num_overlap, C1.shape[0]))
	idx2 = np.array(range(num_overlap, C2.shape[0]))
	idx = np.array(range(num_overlap))
	D1 = C1[idx1, :][:, idx1]
	D2 = C2[idx2, :][:, idx2]
	B1 = C1[idx, :][:, idx1]
	B2 = C2[idx, :][:, idx2]
	
	if 'Y' in pair:
		Y = pair['Y']
		Y1 = Y[:num_overlap]
		Y2 = Y[num_overlap:num_overlap + D1.shape[0]].transpose()
		Y3 = Y[num_overlap + D1.shape[0]:].transpose()
		N1 = np.dot(Y1, Y2)
		N2 = np.dot(Y1, Y3)
	else:
		Y2 = None
		Y3 = None
		N1 = None
		N2 = None
	gw, log = gromov.gromov_wasserstein(D1, D2, B1, B2, p, q, num_overlap, N1=N1, N2=N2, 
		Y2=Y2, Y3=Y3, verbose=True, log=True)	
	savemat(args.output, {'gw': gw})

if __name__ == "__main__":
    main()

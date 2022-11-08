import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.io import loadmat, savemat

def process_pairs(C1, C2, Y=None):
	
	pair = {'C1': C1, 'C2': C2, 'num_overlap': 0}
	num_nodes = C1.shape[0]
	
	degree1 = np.array(C1.sum(axis=1)).flatten()
	degree2 = np.array(C2.sum(axis=1)).flatten()
	overlap = np.intersect1d(np.nonzero(degree1),np.nonzero(degree2))
	pair['num_overlap'] = len(overlap)
	node_mapping1 = dict(zip(overlap,range(pair['num_overlap'])))
	node_mapping2 = dict(zip(overlap,range(pair['num_overlap'])))
	
	zero_degree1 = np.setdiff1d(range(num_nodes), np.nonzero(degree1))
	zero_degree2 = np.setdiff1d(range(num_nodes), np.nonzero(degree2))
	# node_mapping1.update(dict(zip(zero_degree1,range(num_nodes - len(zero_degree1), num_nodes))))
	# node_mapping2.update(dict(zip(zero_degree2,range(num_nodes - len(zero_degree2), num_nodes))))
	
	unique1 = np.setdiff1d(np.nonzero(degree1), overlap)
	unique2 = np.setdiff1d(np.nonzero(degree2), overlap)
	node_mapping1.update(dict(zip(unique1,range(pair['num_overlap'], pair['num_overlap'] + len(unique1)))))
	node_mapping2.update(dict(zip(unique2,range(pair['num_overlap'], pair['num_overlap'] + len(unique2)))))
	
	edge1 = np.array(np.nonzero(C1))
	edge2 = np.array(np.nonzero(C2))
	edge1[0] = np.array([node_mapping1[x] for x in edge1[0]])
	edge1[1] = np.array([node_mapping1[x] for x in edge1[1]])
	edge2[0] = np.array([node_mapping2[x] for x in edge2[0]])
	edge2[1] = np.array([node_mapping2[x] for x in edge2[1]])
	
	C1 = csr_matrix((np.ones(len(edge1[0])), (edge1[0], edge1[1])), shape=(num_nodes - len(zero_degree1), num_nodes - len(zero_degree1)))
	C2 = csr_matrix((np.ones(len(edge2[0])), (edge2[0], edge2[1])), shape=(num_nodes - len(zero_degree2), num_nodes - len(zero_degree2)))
	
	pair['C1'] = C1
	pair['C2'] = C2
	
	if Y is not None:
		idx = np.array(Y.nonzero())
		Y1 = np.array([[node_mapping1[idx[0, x]], idx[1, x] + 1] for x in range(len(idx[0])) if idx[0, x] in node_mapping1])
		Y2 = np.array([[node_mapping2[idx[0, x]], idx[1, x] + 1] for x in range(len(idx[0])) if idx[0, x] in node_mapping2])
		
		Y = np.append(Y1, Y2, axis=0)
		Y = csr_matrix((np.ones(len(Y[:, 0])), (Y[:, 0], Y[:, 1] - 1)), shape=(C1.shape[0] + C2.shape[0] - len(overlap), np.max(Y[:, 1])))
		pair['Y'] = Y
		
	print('preprocess finished!')
	
	return pair
	
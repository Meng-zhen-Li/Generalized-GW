# Generalized-GW
This is an implementation of the generalized Gromov-Wasserstein algorithm. The generalized GW is a supervised network alignment method, and it introduces a loss function based on the topological similarity as well as their relations with the known matching nodes.

## Input Data
The input data of generalized GW should be the adjacency matrices of the pair of networks to be aligned, and the known overlaps between them. The data(adjacency matrices and num_overlap) should be saved in one .mat file, with the two adjacency matrices and the number of overlaps. Note that if num_overlap=min(m,n), then there will be an error because there is no more nodes to align. Two types of input files can be accepted:

- *Processed matrices and number of overlaps:* In the two adjacency matrices, the indices of the overlapping nodes are integers from 1 to number of overlaps, and the non-overlapping nodes afterwards. The indices of the overlapping nodes should be consistent in the two matrices. The name of the matrices 
- *Non-preprocessed matrices:* The indices of two adjacency matrices do not need to be rearranged, but they should reveal the overlapping nodes. The two adjacency matrices have the same size(*m+n-num_overlap*). The indices of the overlapping nodes are consistent in the two matrices. The non-overlapping nodes only appear in one matrix, and the corresponding rows and columns are all zeros in the other matrix. If the input data is in this version, the `--preprocess` option should be set to true.

## Run Generalized GW
The implementation is in python, and you can run it with the following commands.
- An input file is required to run generalized GW:
```
python main.py data/input.mat
```
- The output files are optional, but can be indicated:
```
python main.py data/input.mat --output data/output.mat
```
in which `--output` is the optimal transport of generalized GW.
- Non-preprocessed input data(second type of input) should be preprocessed:
```
python main.py data/input.mat --preprocess true
```
## Output Data
The output data are .mat files with the optimal transport matrix. The matrices can be used as network alignment scores. The matlab function `greedy_matching.m` file can be used to align the two networks based on the optimal transport.

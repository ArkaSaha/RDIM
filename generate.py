import networkx as nx
import numpy as np
import sys

if len(sys.argv) != 4:
	print("Usage : python <script> <n> <m> <d>")
	assert False

g = nx.gnm_random_graph(int(sys.argv[1]), int(sys.argv[2]), directed=True)
with open('data_{}/graph.txt'.format(sys.argv[1]), 'w') as f:
	for u, v in g.edges:
		f.write("{} {}\n".format(u, v))
with open('data_{}/features.txt'.format(sys.argv[1]), 'w') as f:
	for v in g.nodes:
		f.write("{}\t".format(v))
		for n in np.random.uniform(low=-1, high=1, size=int(sys.argv[3])/2):
			f.write("{} ".format(n))
		f.write("\n")

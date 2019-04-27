import matplotlib.pyplot as plt
import networkx as nx
import math
import time

from graphs import Vertex, UnitDistanceGraph, H, V, W, M

"""
M = H().minkowskiVertices(W())
M.update()

tol = 0.05

nodes = list(M.graph.nodes)

with open('debug.out', 'w') as f:
	for i in range(M.n):
		for j in range(i + 1, M.n):
			v = nodes[i]
			w = nodes[j]
#			if M.dist(v, w) < tol:
			if v == w:
				f.write('{}-{}. Distancia = {}\n'.format(v, w, M.dist(v, w)))
print('n = {}\tm = {}'.format(M.n, M.m))
#M = W().minkowskiVertices(H())
"""
myW = W()
print(myW.n)
nodes = list(myW.graph.nodes)
with open('debugW.out', 'w') as f:
	for i in range(myW.n):
		for j in range(i + 1, myW.n):
			v = nodes[i]
			w = nodes[j]
			if v == w:
				f.write('{}-{}. Distancia = {}\n'.format(v, w, myW.dist(v, w)))
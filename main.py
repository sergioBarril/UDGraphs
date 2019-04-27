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
def testMv():

	M = W().minkowskiVertices(H())
	M.update()
	print('n = {}\tm = {}'.format(M.n, M.m))
	nodes = list(M.graph.nodes)
	with open('debugM.out', 'w') as f:
		for i in range(M.n):
			for j in range(i + 1, M.n):
				v = nodes[i]
				w = nodes[j]
				if v.true_eq(w):
					f.write('{}-{}. Distancia = {}\n'.format(v, w, M.dist(v, w)))					



def testM():
	G = M()

	G.update()
	print('n = {}\tm = {}'.format(G.n, G.m))
	print('Connected components: {}'.format(len(list(nx.connected_components(G.graph)))))
	nodes = list(G.graph.nodes)
	minDeg = 99999
	deg12 = 0
	with open('debugM.out', 'w') as f:
		for i in range(G.n):
			v = nodes[i]
			for j in range(i + 1, G.n):
				w = nodes[j]
				if v.true_eq(w):
					f.write('{}-{}. Distancia = {}\n'.format(v, w, G.dist(v, w)))
			vDeg = G.graph.degree[v]

			if vDeg == 12:
				deg12 += 1
				G.remove_node(v)
			else:
				minDeg = min(minDeg, G.graph.degree[v])

	print('Deg 12 deleted: {}'.format(deg12))
	print('Min degree = {}'.format(minDeg))
	print('Cleaned.')
	print('n = {}\tm = {}'.format(G.n, G.m))


def testW():
	myW = W()
	print('n = {}\tm = {}'.format(myW.n, myW.m))
	minDeg = 99999
	nodes = list(myW.graph.nodes)
	with open('debugW.out', 'w') as f:
		for i in range(myW.n):
			v = nodes[i]
			for j in range(i + 1, myW.n):
				w = nodes[j]
				if v.true_eq(w):
					f.write('{}-{}. Distancia = {}\n'.format(v, w, myW.dist(v, w)))
			minDeg = min(minDeg, myW.graph.degree[v])

	print('Min degree = {}'.format(minDeg))




def testV():
	G = V()
	print('n = {}\tm = {}'.format(G.n, G.m))
	nodes = list(G.graph.nodes)
	with open('debugG.out', 'w') as f:
		for i in range(G.n):
			for j in range(i + 1, G.n):
				v = nodes[i]
				w = nodes[j]
				if v.true_eq(w):
					f.write('{}-{}. Distancia = {}\n'.format(v, w, G.dist(v, w)))

testM()
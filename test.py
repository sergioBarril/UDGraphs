import pytest

import networkx as nx
import math
import time

from vertex import Vertex

from graphs import UnitDistanceGraph, H, V, W, M, MoserSpindle
from graphs import J, K, L, T, U, G

class TestG():
	def testG(self):
		myG = G()
		myG.update()

		for v, w in myG.graph.edges:
			assert v.isUnitDist(w)


class TestColoring():
	def check_coloring(self, G):
		for v in G.graph.nodes:
			assert v.isColored()
			for w in G.graph[v]:
				z = G.search_vertex(w)
				assert z.isColored()
				assert z.color != v.color
	def check_unit(self, G):
		for v, w in G.graph.edges:
			assert v.isUnitDist(w)
	
	def test_G(self):
		myG = UnitDistanceGraph()
		myG.load_graph('G')
		myG.sat_color(colors = 5)
		self.check_coloring(myG)
		self.check_unit(myG)

	def test_610(self):
		G = UnitDistanceGraph()
		G.load_graph('610')
		G.sat_color(colors = 5)		
		self.check_coloring(G)


	def test_H(self):
		G = H()
		G.update_and_sort()
		G.color_graph()
		self.check_coloring(G)

	def test_J(self):
		G = J()
		G.update_and_sort()
		G.color_graph()
		self.check_coloring(G)

	def test_K(self):
		G = K()
		G.update_and_sort()
		G.color_graph()
		self.check_coloring(G)
	
	def test_L(self):
		G = L()
		G.update_and_sort()
		G.color_graph()
		self.check_coloring(G)

	def test_W_5(self):
		G = W()
		G.update_and_sort()
		G.color_graph(5)
		self.check_coloring(G)

class TestTriangles():
	def test_H(self):
		G = H()
		assert G.num_triangles(Vertex(0,0)) == 6
		assert G.num_triangles(Vertex(-1, 0)) == 2

class TestVertex():
	def test_add(self):
		v = Vertex(1,2)
		w = Vertex(3,4)

		assert v + w == Vertex(4,6)

	def test_substract(self):
		v = Vertex(1,2)
		w = Vertex(3,4)

		assert v - w == Vertex(-2, -2)

	def test_divide(self):
		v = Vertex(4,2)

		assert v/2 == Vertex(2, 1)

	def test_rotate(self):
		v = Vertex(1,0)

		assert v.rotate(math.pi/2) == Vertex(0,1)



class TestSpindles():
	def test_moser(self):
		G = MoserSpindle()
		for v in G.graph.nodes:
			assert G.num_spindles(v) == 1
		assert G.spindles() == 1

	def test_2_mosers(self):
		G = MoserSpindle()
		H = MoserSpindle().rotate(math.pi)

		F = G.union(H)

		assert F.num_spindles(Vertex(0,0)) == 2
		for v in F.graph.nodes:
			if v != Vertex(0,0):
				assert F.num_spindles(v) == 1
		assert F.spindles() == 2

	def test_V(self):
		G = V()
		spindles = 0
		for v in G.graph.nodes:
			spindles += G.num_spindles(v)

		assert spindles == 0

	def test_T(self):
		G = T()
		assert G.num_spindles(Vertex(0,0)) == 1
		assert G.num_spindles(Vertex(1.8333333, 0.552770798)) == 0
		assert G.num_spindles(Vertex(1.5, 0.866025)) == 1
		assert G.spindles() == 1

class TestRotate():
	def test_A(self):
		A = UnitDistanceGraph()
		A.add_edge(Vertex(1,0), Vertex(0,0))
		AR = A.rotate(2*math.pi/3)
		ARR = AR.rotate(2*math.pi/3)
		ARRR = ARR.rotate(2*math.pi/3)

		assert Vertex(0,0) in ARRR.graph.nodes
		assert Vertex(1,0) in ARRR.graph.nodes

class TestGraphs():
	def test_H(self):
		G = H()
		assert G.n == 7
		assert G.m == 12

	def test_H2(self):
		G1 = H()
		G2 = H(Vertex(1,0))

		nodes1 = list(G1.graph.nodes)
		nodes2 = list(G2.graph.nodes)

		for i in range(G1.n):
			assert nodes1[i].x + 1 == nodes2[i].x

	def test_J(self):
		G = J()
		assert G.n == 31
		assert G.m == 72

	def test_K(self):
		G = K()
		assert G.n == 61
		assert G.m == 150

	def test_L(self):
		G = L()
		assert G.n == 121
		assert G.m == 301

		alpha = 2*math.asin(0.125)
		
		# Check that both linking diagonals are neighbours
		A = Vertex(-2,0)
		B = Vertex(2,0)
		B2 = B.rotate(alpha, center = A)
		assert B2.isUnitDist(B)

	def test_T(self):
		G = T()
		assert G.n  == 9
		assert G.m == 15

	# def test_U(self):
	# 	G = U()
	# 	assert G.n == 15
	# 	assert G.m == 33

	def test_V(self):
		G = V()
		assert G.n == 31
		assert G.m == 60

	def test_W(self):
		G = W()
		assert G.n == 301
		assert G.m == 1230

	def test_M(self):
		G = M()
		assert G.n == 1345
		assert G.m == 8268

		assert Vertex(0,0) in G.graph.nodes
		assert Vertex(1,0) in G.graph.nodes
		assert Vertex(-1, 0) in G.graph.nodes
		assert Vertex(-0.5, math.sqrt(3)/2) in G.graph.nodes
		assert Vertex(-0.5, -math.sqrt(3)/2) in G.graph.nodes
		assert Vertex(0.5, math.sqrt(3)/2) in G.graph.nodes
		assert Vertex(0.5, -math.sqrt(3)/2) in G.graph.nodes

	def test_moser(self):
		G = MoserSpindle()
		assert G.n == 7
		assert G.m == 11


import pytest

import networkx as nx
import math
import time

from graphs import Vertex, UnitDistanceGraph, H, V, W, M, MoserSpindle
from graphs import J, K, L, T


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
		assert G.num_spindles(Vertex(1,0)) == 1

	def test_2_mosers(self):
		G = MoserSpindle()
		H = MoserSpindle().rotate(math.pi)

		F = G.union(H)
		assert F.num_spindles(Vertex(0,0)) == 2
		assert F.num_spindles(Vertex(1,0)) == 1

	def test_V(self):
		G = V()
		spindles = 0
		for v in G.graph.nodes:
			spindles += G.num_spindles(v)

		assert spindles == 0

	def test_T(self):
		G = T()
		assert G.num_spindles(Vertex(0,0)) == 1
		assert G.spindles() == 1


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

	def test_moser(self):
		G = MoserSpindle()
		assert G.n == 7
		assert G.m == 11
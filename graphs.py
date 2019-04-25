import matplotlib.pyplot as plt
import networkx as nx
import copy

from math import isclose, sqrt, sin, cos, acos

class Vertex:
	"""Vertex of the graph"""
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __str__(self):
		return '(' + str(self.x) + ', ' + str(self.y) + ')'

	def __add__(self, v):
		return Vertex(self.x + v.x, self.y + v.y)


	def __eq__(self, other):
		if not isinstance(other, Vertex):
			return NotImplemented
		return isclose(self.x, other.x, abs_tol=1.e-6) and isclose(self.y, other.y, abs_tol=1.e-6)

	def __hash__(self):
		round_x = round(self.x)
		round_y = round(self.y)

		hash1 = hash(round_x)
		hash2 = hash(round_y)

		return hash((hash1, hash2))


	def rotate(self, i, k):
		alpha = (2 * i - 1)/(2 * i)
		alpha = acos(alpha)
		alpha *= k

		x = self.x * cos(alpha) - self.y * sin(alpha)
		y = self.x * sin(alpha) + self.y * cos(alpha)

		x = round(x, 8)
		y = round(y, 8)

		return Vertex(x,y)

	def getR(self):
		x2 = self.x * self.x
		y2 = self.y * self.y

		return sqrt(x2 + y2)




class UnitDistanceGraph:
	"""Graph with edges of distance 1"""

	def __init__(self):
		self.n = 0
		self.m = 0
		self.graph = nx.Graph()

	def update(self):
		self.n = self.graph.number_of_nodes()
		self.m = self.graph.number_of_edges()

	def add_node(self, v):
		self.graph.add_node(v)
		self.update()

	def add_edge(self, v, w):
		if isUnitDist(v, w):
			self.graph.add_edge(v, w)
			self.update()
		else:
			print("Not Unit Distance Edge. Distance: {}".format(dist(v, w)))

	def remove_node(self, v):
		self.graph.remove_node(v)
		self.update()


	def trim(self, d):
		abs_tol = 1.e-6
		for node in list(self.graph.nodes):
			if node.getR() > d + abs_tol:
				self.remove_node(node)



	def print_graph(self):
		print("#################")
		print("n = {}\t m = {}".format(self.n, self.m))
		print("#################")

		for node in self.graph.nodes:
			print(node)

class V(UnitDistanceGraph):
	def __init__(self):
		UnitDistanceGraph.__init__(self)
		
		v0 = Vertex(0,0)
		self.add_node(v0)

		v = Vertex(1,0)

		for k in range(5):
			for j in range(6):
				w1 = v.rotate(3, k/2).rotate(1, j)
				w2 = v.rotate(3, k/2).rotate(1, j + 1)

				self.add_edge(w1, v0)
				self.add_edge(w1, w2)

class W(UnitDistanceGraph):
	def __init__(self):
		UnitDistanceGraph.__init__(self)

		V1 = V()
		V2 = V()
		self.graph = trimMinkowski(V1, V2, sqrt(3))
		self.update()


class H(UnitDistanceGraph):
	def __init__(self):
		UnitDistanceGraph.__init__(self)

		v0 = Vertex(0,0)
		self.add_node(v0)

		v = Vertex(1, 0)
		for j in range(6):
			w1 = v.rotate(1, j)
			w2 = v.rotate(1, j + 1)

			self.add_edge(w1, w2)
			self.add_edge(w1, v0)


def dist(v, w):
	x = v.x - w.x
	x = x * x

	y = v.y - w.y
	y = y * y

	return sqrt(x + y)

def isUnitDist(v, w):
		return isclose(1, dist(v, w), abs_tol=1.e-6)

def trimMinkowski(G, H, d):
	M = UnitDistanceGraph()

	for v in G.graph.nodes:
		for w in H.graph.nodes:
			M.add_node(v)
			M.add_node(w)
			M.add_node(v + w)

	M.trim(d)

	for v in M.graph.nodes:
		for w in M.graph.nodes:
			if isUnitDist(v, w):
				M.add_edge(v, w)

	return M.graph


def minkowskiSum(G, H):
	M = copy.deepcopy(G)

	for v in G.graph.nodes:
		for w in H.graph.nodes:
			x = v + w

			M.add_node(x)
			M.add_node(w)

			for z in M.graph.nodes:
				if isUnitDist(x,z):
					M.add_edge(x, z)

			if isUnitDist(v, w):
				M.add_edge(v, w)
				M.add_edge(x, w)

	return M
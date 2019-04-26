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
		""" (x, y) """
		return '(' + str(self.x) + ', ' + str(self.y) + ')'

	def __add__(self, v):
		""" Sum of 2 vertices """
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

	def save_graph(self, fname):
		with open(fname + '.v', 'w') as f:
			f.write('{}\n'.format(self.n))

			for v in self.graph.nodes:
				f.write("{} {}\n".format(v.x, v.y))

		with open(fname + '.e', 'w') as f:
			f.write('{}\n'.format(self.m))

			for v, w in self.graph.edges:
				f.write('{} {} {} {}\n'.format(v.x, v.y, w.x, w.y))

	def load_vertices(self, fname):
		with open(fname + '.v', 'r') as f:
			self.n = int(f.readline())
			
			for line in f:
				vx, vy = line.split()
				self.add_node(Vertex(float(vx), float(vy)))

	def load_edges(self, fname):
		with open(fname + '.e', 'r') as f:
			self.m = int(f.readline())

			for line in f:
				vx, vy, wx, wy = line.split()
				v = Vertex(float(vx), float(vy))
				w = Vertex(float(wx), float(wy))
				self.add_edge(v, w)

	def load_graph(self, fname):
		self.__init__()
		self.load_edges(fname)
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

		self.graph = trimMinkowski(V(), V(), sqrt(3))
		self.update()

class M(UnitDistanceGraph):
	def __init__(self):
		UnitDistanceGraph.__init__(self)
		myW = UnitDistanceGraph()
		myW.load_graph('W')

		self.graph = minkowskiSum(H(), myW)
		self.update()


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
	M = UnitDistanceGraph()

	for v in G.graph.nodes:
		for w in H.graph.nodes:
			M.add_node(v)
			M.add_node(w)
			M.add_node(v + w)

	for v in M.graph.nodes:
		for w in M.graph.nodes:
			if isUnitDist(v, w):
				M.add_edge(v, w)

	return M.graph
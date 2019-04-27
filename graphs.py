import matplotlib.pyplot as plt
import networkx as nx
import copy
import os
import decimal

from math import isclose, sqrt, sin, cos, acos, fabs

abs_tol = 0.02
rel_tol = 1.e-4

class Vertex:
	"""Vertex of the graph. Contains coordinates and color"""
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.r = self.getR()
		self.color = -1

	def __str__(self):
		""" (x, y)[color] """

		vertex = '({}, {})'.format(round(self.x, 5), round(self.y, 5), self.color)
		color = '[{}]'.format(self.color)

		if self.color != -1:
			return vertex + color
		else:
			return vertex

	def __hash__(self):
		"""
		Hashes with the first two decimal places, rounded
		"""
		round_x = round(self.x, 2)
		round_y = round(self.y, 2)

		hash1 = hash(round_x)
		hash2 = hash(round_y)

		return hash((hash1, hash2))

	def __add__(self, v):
		""" Sum of 2 vertices """
		return Vertex(self.x + v.x, self.y + v.y)


	def __eq__(self, other):
		if not isinstance(other, Vertex):
			return NotImplemented
		return round(self.x, 2) == round(other.x, 2) and round(self.y, 2) == round(other.y, 2)

	def dist(self, v):
		"""
		Returns its Euclidean distance to another vertex
		"""
		x = self.x - v.x
		x = x * x

		y = self.y - v.y
		y = y * y

		return sqrt(x + y)

	def isUnitDist(self, v):
		"""
		Given two vertices, check whether they're at unit distance from each other.
		"""
		return isclose(1, self.dist(v), rel_tol= 1.e-9, abs_tol = 0)

	def rotate(self, i, k):
		"""
		Returns a vertex rotated with respect the origin.
		i changes the angle, where angle = arccos(2i-1 / 2i)
		k changes how many times is the rotation applied
		"""
		alpha = (2 * i - 1)/(2 * i)
		alpha = acos(alpha)
		alpha *= k

		x = self.x * cos(alpha) - self.y * sin(alpha)
		y = self.x * sin(alpha) + self.y * cos(alpha)

		return Vertex(x,y)

	def getR(self):
		"""
		Returns the distance to (0,0)
		"""
		x2 = self.x * self.x
		y2 = self.y * self.y

		return sqrt(x2 + y2)

	def true_eq(self, other):
		"""
		Checks if this vertex is close enough to another one to be considered the same.
		I'll probably change all of this, it's probably unnecessary
		"""
		caseA = isclose(self.x, other.x, rel_tol= rel_tol, abs_tol= 5*abs_tol) and isclose(self.y, other.y, rel_tol= rel_tol, abs_tol=abs_tol)
		caseB = isclose(self.x, other.x, rel_tol= rel_tol, abs_tol= abs_tol) and isclose(self.y, other.y, rel_tol= rel_tol, abs_tol=5*abs_tol)

		same = self.x - other.x == 0 and self.y - other.y == 0
		return caseA or caseB or same

class UnitDistanceGraph:
	"""Graph with edges of distance 1"""

	def __init__(self):
		"""
		"""
		self.n = 0 # Vertices
		self.m = 0 # Edges	
		self.graph = nx.Graph()

	def update(self):
		"""
		Refreshes the number of vertices and edges
		"""
		self.n = self.graph.number_of_nodes()
		self.m = self.graph.number_of_edges()

	#	**********************************************************************
	#								BASIC OPERATIONS
	#	**********************************************************************

	def add_node(self, v):
		"""
		Adds the node v to the graph
		"""
		self.graph.add_node(v)
		self.update()

	def add_edge(self, v, w):
		"""
		This checks if both vertices are at unit distance. If they are, it adds the edge
		to the graph. Else, it prints the error, specifying the distance they're at.
		"""
		if v.isUnitDist(w):
			self.graph.add_edge(v, w)
			self.update()
		else:
			print("Not Unit Distance Edge. Distance: {}".format(v.dist(w)))

	def remove_node(self, v):
		"""
		Removes node v from the graph
		"""
		self.graph.remove_node(v)
		self.update()


	#	**********************************************************************
	#								ADVANCED OPERATIONS
	#	**********************************************************************

	def trim(self, d):
		"""
		Removes all vertices (and edges adjacent to those) that are at greater distance
		than d from (0,0)
		"""
		for node in list(self.graph.nodes):
			if node.r > d + abs_tol:
				self.remove_node(node)
		self.update()

	def union(self, G):
		"""
		Returns the union of this graph with G
		"""
		M = UnitDistanceGraph()
		for v in self.graph.nodes:
			for w in G.graph.nodes:
				M.add_node(v)
				M.add_node(w)

		for v in M.graph.nodes:
			for w in M.graph.nodes:
				if v.isUnitDist(w):
					M.add_edge(v, w)

		return M.graph

	def minkowskiSum(self, G):
		"""
		Minkowski sum of this graph and G.
		"""

		# Add all edges from self
		M = copy.deepcopy(self)

		# Add all edges from G
		for e in G.graph.edges:
			M.add_edge(e[0], e[1])

		# Set of sum nodes
		new_nodes = { v + w for v in self.graph.nodes for w in G.graph.nodes if v + w not in M.graph.nodes}

		# Add edges on them
		for x in list(new_nodes):
			for v in self.graph.nodes:
				if v.isUnitDist(x) and not v.true_eq(x):
					M.add_edge(v, x)

			for w in G.graph.nodes:
				if w.isUnitDist(x) and not w.true_eq(x):
					M.add_edge(w, x)

			for z in new_nodes:
				if z.isUnitDist(x) and not z.true_eq(x):
					M.add_edge(x, z)

			new_nodes.remove(x)

		return M.graph

	def trimMinkowski(self, G, d):
		""" Minkowski sum, trimming vertices at distance greater than d
			This expects 2 graphs already trimmed to distance d, and the distance
		"""
		# Add all edges from self
		M = copy.deepcopy(self)

		# Add all edges from G
		for e in G.graph.edges:
			M.add_edge(e[0], e[1])

		# Set of sum nodes
		new_nodes = { v + w for v in self.graph.nodes for w in G.graph.nodes if (v+w).r < d + abs_tol}

		# Add edges on them
		for x in list(new_nodes):
			for v in self.graph.nodes:
				if v.isUnitDist(x) and not v.true_eq(x):
					M.add_edge(v, x)

			for w in G.graph.nodes:
				if w.isUnitDist(x) and not w.true_eq(x):
					M.add_edge(w, x)

			for z in new_nodes:
				if z.isUnitDist(x) and not z.true_eq(x):
					M.add_edge(x, z)

			new_nodes.remove(x)

		return M.graph



	#	**********************************************************************
	#								VERTEX SORTING
	#	**********************************************************************

	#def num_triangles(self, v):
		"""
		Return the number of unit distance triangles that contain v
		"""
	#	neighbours = self.graph[v]




	#	**********************************************************************
	#								READ/WRITE
	#	**********************************************************************


	def print_graph(self):
		"""
		Prints all vertices of the graph
		"""
		print("#################")
		print("n = {}\t m = {}".format(self.n, self.m))
		print("#################")

		for node in self.graph.nodes:
			print(node)

	def save_graph(self, fname):
		"""
		Given a filename, this saves two files in the 'graphs/' folder.

		* myFile.v -- number of vertices, and all the vertices.
		* myFile.e -- number of edges, and all the edges.

		"""
		with open(os.path.join('graphs', fname + '.v'), 'w') as f:
			f.write('{}\n'.format(self.n))

			for v in self.graph.nodes:
				f.write("{} {}\n".format(v.x, v.y))

		with open(os.path.join('graphs', fname + '.e'), 'w') as f:
			f.write('{}\n'.format(self.m))

			for v, w in self.graph.edges:
				f.write('{} {} {} {}\n'.format(v.x, v.y, w.x, w.y))

	def load_vertices(self, fname):
		"""
		Given a filename, this loads the vertices found in 'graphs/myFile.v'
		"""
		with open(os.path.join('graphs', fname + '.v'), 'r') as f:
			self.n = int(f.readline())
			
			for line in f:
				vx, vy = line.split()
				self.add_node(Vertex(float(vx), float(vy)))

	def load_edges(self, fname):
		"""
		Given a filename, this loads the edges found in 'graphs/myFile.e'
		"""
		with open(os.path.join('graphs', fname + '.e'), 'r') as f:
			self.m = int(f.readline())

			for line in f:
				vx, vy, wx, wy = line.split()
				v = Vertex(float(vx), float(vy))
				w = Vertex(float(wx), float(wy))
				self.add_edge(v, w)

	def load_graph(self, fname):
		"""
		Given a filename, this loads the graph with fname in 'graphs/'
		"""
		self.__init__()
		self.load_edges(fname)
		self.update()


class H(UnitDistanceGraph):
	"""
	Unit distance graph with 7 vertices and 12 edges. It's a regular hexagon with its center.
	"""
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
		self.update()

class V(UnitDistanceGraph):
	"""
	Unit distance graph with 31 vertices and 60 (?) edges. It's 5 regular hexagons slightly rotated.
	"""
	def __init__(self):
		UnitDistanceGraph.__init__(self)
		
		v0 = Vertex(0,0)
		self.add_node(v0)

		v = Vertex(1,0)

		for k in range(-2, 3, 1):
			for j in range(6):
				w1 = v.rotate(3, k/2).rotate(1, j)
				w2 = v.rotate(3, k/2).rotate(1, j + 1)

				self.add_edge(w1, v0)
				self.add_edge(w1, w2)
		self.update()

class W(UnitDistanceGraph):
	"""
	Unit distance graph with 301 vertices and 1230 edges. It's the Minkowski Sum of V with itself, and trimmed up
	to sqrt(3).
	"""
	def __init__(self):
		UnitDistanceGraph.__init__(self)

		self.graph = V().trimMinkowski(V(), sqrt(3))
		self.update()



class M(UnitDistanceGraph):
	"""
	Unit distance graph with 1345 vertices. It's the Minkowski Sum of W with H.
	"""
	def __init__(self):
		UnitDistanceGraph.__init__(self)

		self.graph = W().minkowskiSum(H())
		self.update()
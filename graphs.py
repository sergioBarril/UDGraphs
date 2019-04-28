import matplotlib.pyplot as plt
import networkx as nx
import copy
import os
import decimal
import itertools as it

import math
from math import isclose, sqrt, sin, cos, acos, asin, fabs, pi

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

		vertex = '({}, {})'.format(round(self.x, 3), round(self.y, 3), self.color)
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

	def __sub__(self, v):
		""" Subtraction of 2 vertices """
		return Vertex(self.x - v.x, self.y - v.y)

	def __truediv__(self, num):
		""" Division of the Vertex by a number """
		if isinstance(num, Vertex):
			return NotImplemented
		return Vertex(self.x/num, self.y/num)


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

	def isUnitTriangle(self, v, w):
		"""
		Given two vertices, check whether this vertex forms a
		unit distance triangle with those two
		"""
		return isUnitDist(v, w)

	def rotate(self, i, k = None, center = None):
		"""
		Returns a vertex rotated with respect to the given center, or
		(0,0) as a default center.
		
		If k is given:
			i changes the angle, where angle = arccos(2i-1 / 2i)
			k changes how many times is the rotation applied
		Else:
			i is the angle, in radians
		"""
		if center is None:
			center = Vertex(0,0)

		if k is None:
			alpha = i
		else:
			alpha = (2 * i - 1)/(2 * i)
			alpha = acos(alpha)
			alpha *= k

		c = center
		x = (self.x - c.x) * cos(alpha) - (self.y - c.y) * sin(alpha) + c.x
		y = (self.x - c.x) * sin(alpha) + (self.y - c.y) * cos(alpha) + c.y

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
			print("Not Unit Distance Edge:")
			print("v = {}\tw = {}\tDistance: {}".format(v, w, v.dist(w)))

	def copy_edge(self, v, w):
		"""
		This adds an edge to the graph, even if it isn't unit distance.
		Use it with caution.
		"""
		self.graph.add_edge(v, w)
		self.update()

	def remove_node(self, v):
		"""
		Removes node v from the graph
		"""
		self.graph.remove_node(v)
		self.update()


	#	**********************************************************************
	#								ADVANCED OPERATIONS
	#	**********************************************************************

	def rotate(self, i, k=None, center=None):
		"""
		Rotates the graph around the given center, or (0,0) as default

		If k is given:
			i changes the angle, where angle = arccos(2i-1 / 2i)
			k changes how many times is the rotation applied
		Else:
			i is the angle, in radians
		"""

		if center is None:
			center = Vertex(0,0)

		if k is None:
			alpha = i
		else:
			alpha = (2 * i - 1)/(2 * i)
			alpha = acos(alpha)
			alpha *= k

		R = UnitDistanceGraph()

		for v in self.graph.nodes:
			vr = v.rotate(alpha, center = center)
			for w in self.graph[v]:
				wr = w.rotate(alpha, center = center)
				R.copy_edge(vr, wr)

		return R


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
		Returns the union of this graph with G.
		* The vertices are the union of both vertex sets
		* The edges are the edges of both, plus those vertices at
			unit distance.
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

		return M

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

		return M

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

		return M



	#	**********************************************************************
	#								VERTEX SORTING
	#	**********************************************************************

	def num_triangles(self, v):
		"""
		Return the number of unit distance triangles that contain v
		"""
		neighbours = self.graph[v]

		triangles = 0

		for pair in it.combinations(neighbours, 2):
			if pair[0].isUnitDist(pair[1]):
				triangles += 1

		return triangles

	def spindles(self):
		spindles = 0
		for v in self.graph.nodes:
			spindles = max(spindles, self.num_spindles(v))

		return spindles


	def num_spindles(self, A):
		"""
		Return the number of Moser spindles that contain A
		"""

		neighbours = self.graph[A]

		spindles = 0

		for pair in it.combinations(neighbours, 2): # For each triangle
			v, w = pair
			if v.isUnitDist(w):
				print("Triangle: A = {}\tv = {}\tw = {}".format(A,v,w))
				rhombuses = self.is_rhombus(A, v, w) # Get its rhombuses
				for tip1, tip2 in rhombuses.items():
					if A == tip1: # If the vertex at study is not at the tip of the rhombus
						spindles += self.spindles_in_rhombus(tip1, tip2, v, w) # (Tip mode)
					elif v == tip1:
						spindles += self.spindles_in_rhombus(tip1, tip2, A, w, False) # (A is not in the tip mode)
					else: # w == tip1
						spindles += self.spindles_in_rhombus(tip1, tip2, v, A, False)

		return spindles

	def spindles_in_rhombus(self, tip1, tip2, v, w, tip_mode = True):
		"""
		Returns the number of spindles that contain the rhombus.

		"""
		def is_valid_rhombus(self, sign, tip1, tip2, v, w):
			"""
			Rotates the given rhombus, given the sign of rotation
			around the first tip. Returns true if it's a valid rhombus.
			"""

			# Nodes of the graph
			G = self.graph.nodes

			# Rotation of the rest of points around tip1
			tip2R = tip2.rotate(3, sign, center = tip1)
			vR = v.rotate(3, sign, center = tip1)
			wR = w.rotate(3, sign, center = tip1)

			if tip2R not in G or vR not in G or wR not in G: # if they're not vertices
				return False  # they won't make a valid rhombus

			tip1N = self.graph[tip1] # Neighbours of tip1
			tip2RN = self.graph[tip2R] # Neighbours of tip2R

			# There's no need i think, because it's unit distance already, they should be
			# neighbours no matter what. i'll leave this here tho.
			valid = vR in tip1N and wR and wR in self.graph[vR]
			valid = valid and vR in tip2RN and wR in tip2RN
			valid = valid and tip2 in tip2RN

			return valid

		print("Rhombus: Tip1 = {}\tTip2 ={}\tv = {}\tw = {}".format(tip1, tip2, v, w))
		spindles = 0
		
		if is_valid_rhombus(self, 1, tip1, tip2, v, w):
			spindles += 1
		if not tip_mode:
			if is_valid_rhombus(self, -1, tip1, tip2, v, w):
				spindles += 1
		if is_valid_rhombus(self, 1, tip2, tip1, v, w):
			spindles += 1
		if is_valid_rhombus(self, -1, tip2, tip1, v, w):
			spindles += 1

		if not tip_mode:
			spindles /= 2
		print('Spindles: {}'.format(spindles))
		return spindles

	def is_rhombus(self, u, v, w):
		"""
		Given a unit distance triangle, return all their unit rhombuses
		Returns a dict with the fourth vertex of all rhombuses with
		u, v and w.
		"""

		rhombuses = dict()

		z = v.rotate(math.pi/3, center = u)
		if z == w:
			v, w  = w, v
			z = v.rotate(math.pi/3, center = u)

		if z in self.graph[u] and z in self.graph[v]:
			rhombuses[w] = z

		z = w.rotate(-math.pi/3, center = u)
		if z in self.graph[u] and z in self.graph[w]:
			rhombuses[v] = z

		z = w.rotate(math.pi/3, center = v)
		if z in self.graph[v] and z in self.graph[w]:
			rhombuses[u] = z

		return rhombuses


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
	def __init__(self, v0 = None):
		UnitDistanceGraph.__init__(self)

		if v0 is None:
			v0 = Vertex(0, 0)

		self.add_node(v0)

		v = v0 + Vertex(1, 0)

		for j in range(6):
			w1 = v.rotate(1, k = j, center = v0)
			w2 = v.rotate(1, k = j + 1, center = v0)

			self.add_edge(w1, w2)
			self.add_edge(w1, v0)
		self.update()

class J(UnitDistanceGraph):
	"""
	Unit distance graph with 31 vertices, 72 edges and made with 13 copies of H
	"""
	def __init__(self):
		A = H().minkowskiSum(H())
		for v in list(A.graph.nodes):
			if isclose(v.r, sqrt(3)):
				A = A.union(H(v))

		self.graph = A.graph
		self.update()


class K(UnitDistanceGraph):
	"""
	Unit distance graph with 61 vertices, 150 edges, and made with 2 copies of J
	"""
	def __init__(self):
		J1 = J()
		alpha = 2*asin(0.25)
		J2 = J1.rotate(alpha)

		self.graph = J1.union(J2).graph
		self.update()


class L(UnitDistanceGraph):
	"""
	Unit distance graph with 121 vertices, 301 edges, and made with 2 copies of K (52 copies of H)
	"""
	def __init__(self):
		K1 = K()
		alpha = 2*asin(0.125)
		K2 = K().rotate(alpha, center = Vertex(-2,0))

		self.graph = K1.union(K2).graph
		self.update()

class T(UnitDistanceGraph):
	"""
	Unit distance graph T, with 9 vertices. It's made by adding two vertices to the Moser's spindle.
	"""
	def __init__(self):
		moser = MoserSpindle()
		A = Vertex(1/2, sqrt(3)/2)
		B = Vertex(1,0).rotate(3, 1)

		Z = Vertex(3/2, sqrt(3)/2)
		ZR = Z.rotate(3,1)

		P = (A - B) + ZR
		Q = (B - A) + Z

		moser.add_edge(Q, B)
		moser.add_edge(Q, Vertex(1,0))

		moser.add_edge(P, A)
		moser.add_edge(P, A.rotate(3,1))

		self.graph = moser.graph
		self.update()

class U(UnitDistanceGraph):
	"""
	Unit distance graph U, with 15 vertices and 33 edges. Made of 3 copies of T, at 120deg rotations.
	"""
	def __init__(self):
		UnitDistanceGraph.__init__(self)

		T1 = T()
		T2 = T1.rotate((2*math.pi)/3, center = Vertex(0,0))
		T3 = T2.rotate((2*math.pi)/3, center = Vertex(0,0))

		myU = T1.union(T2)
		myU = myU.union(T3)

		self.graph = myU.graph
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

		self.graph = V().trimMinkowski(V(), sqrt(3)).graph
		self.update()

class M(UnitDistanceGraph):
	"""
	Unit distance graph with 1345 vertices. It's the Minkowski Sum of W with H.
	"""
	def __init__(self):
		UnitDistanceGraph.__init__(self)

		self.graph = W().minkowskiSum(H()).graph
		self.update()

class MoserSpindle(UnitDistanceGraph):
	"""
	Unit distance graph with 7 vertices and 8 edges
	"""
	def __init__(self):
		UnitDistanceGraph.__init__(self)

		A = UnitDistanceGraph()
		B = UnitDistanceGraph()

		A.add_edge(Vertex(0,0), Vertex(1,0))
		B.add_edge(Vertex(0,0), Vertex(1/2, sqrt(3)/2))

		C = A.minkowskiSum(B)
		CR = C.rotate(3,1)

		self.graph = C.union(CR).graph
		self.update()
import networkx as nx
import copy
import os
import itertools as it

import math
from math import isclose, sqrt, sin, cos, acos, asin, fabs, pi


from vertex import Vertex
from color import ColoringGraph
from sat import UDGSat
from tikz import TikzDocument


abs_tol = 0.02
rel_tol = 1.e-4

class UnitDistanceGraph:
	"""Graph with edges of distance 1"""

	def __init__(self):
		"""
		"""
		self.n = 0 # Vertices
		self.m = 0 # Edges	
		self.graph = nx.Graph()
		self.sorted_nodes = []

	def update(self):
		"""
		Refreshes the number of vertices and edges
		"""
		self.n = self.graph.number_of_nodes()
		self.m = self.graph.number_of_edges()

	def update_and_sort(self):
		"""
		Refreshes the number of vertices and edges, and sorts the vertices
		"""
		self.update()
		self.sorted_nodes = self.sort_nodes()

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
			R.add_node(vr)
			for w in self.graph[v]:
				wr = w.rotate(alpha, center = center)
				R.copy_edge(vr, wr)

		R.update()
		return R

	def translate(self, diff):
		"""
		Translation of the graph
		"""	
		T = UnitDistanceGraph()
		for v in self.graph.nodes:
			T.add_node(v + diff)

		for v in self.graph.nodes:
			for w in self.graph[v]:
				T.add_edge(v + diff, w + diff)

		T.update()
		return T

	def reflection(self, c):
		"""
		Reflection over a vertical axis
		"""
		NG = UnitDistanceGraph()

		for v in self.graph.nodes:
			x = v - Vertex(c, 0) 
			NG.add_node(Vertex(c - x.x, v.y))


		return NG.union(UnitDistanceGraph())

	def trim(self, d):
		"""
		Removes all vertices (and edges adjacent to those) that are at greater distance
		than d from (0,0)
		"""


		for node in list(self.graph.nodes):
			if node.r > d + abs_tol:
				self.remove_node(node)
		self.update()

		return self

	def union(self, G):
		"""
		Returns the union of this graph with G.
		* The vertices are the union of both vertex sets
		* The edges are the edges of both, plus those vertices at
			unit distance.
		"""
		M = UnitDistanceGraph()
		
		for v in self.graph.nodes:
			M.add_node(v)
		
		for w in G.graph.nodes:		
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

	def negative_y(self):
		"""
		Returns the reflection of this graph with respect
		to the x-axis
		"""
		G = UnitDistanceGraph()
		for v in set(self.graph.nodes):
			G.add_node(Vertex(v.x, -v.y))

		return G

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
			spindles /= 2		# revisar
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


	def node_score(self, v):
		"""
		Returns the score to sort nodes
		"""
		v_spindles = self.num_spindles(v)
		v_degree = self.graph.degree[v]
		v_triangles = self.num_triangles(v)

		return v_spindles * 100 + v_degree * 10 + v_triangles

	def sort_nodes(self):
		"""
		Returns a list with all the nodes, sorted
		"""
		return sorted(list(self.graph.nodes), reverse=True, key=self.node_score)


	#	**********************************************************************
	#								COLORING
	#	**********************************************************************

	def color_graph(self, colors = 4, new = True):
		cg = ColoringGraph(self, colors = colors, new = new)
		return cg.color()

	def sat_color(self, colors = 4):
		sat = UDGSat(self, colors = colors)
		return sat.solve(color = True)

	def uncolor_graph(self):
		for v in self.sorted_nodes:
			v.color = -1
		self.update()

	def search_vertex(self, v):
		"""
		Given a Vertex, finds that vertex in the graph.
		
		This is relevant for coloring problems, since 'Vertex(x,y)' would
		create a new vertex, thus not coloring the vertex we need.
		"""
		for w in self.sorted_nodes:
			if w == v:
				return w

		return False

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

	def save_dimacs(self, fname):
		with open(os.path.join('dimacs', fname + '.col'), 'w') as f:
			f.write('c FILE: {}\n'.format(fname + '.col'))
			f.write('c\n')
			f.write('p edge {} {}\n'.format(self.n, self.m))

			id_nodes_dict = dict()
			self.update_and_sort()

			i = 0
			for v in self.sorted_nodes:
				id_nodes_dict[v] = i
				i += 1

			for v, w in self.graph.edges:
				f.write("e {} {}\n".format(id_nodes_dict[v], id_nodes_dict[w]))

		with open(os.path.join('dimacs', fname + '.dict'), 'w') as f:
			for v, v_id in id_nodes_dict.items():
				f.write("{} {} {}\n".format(v_id, v.x, v.y))

	def save_cnf(self, fname, colors = 4, random = False):
		sat = UDGSat(self, colors)
		sat.save_cnf(fname, randomMode = random)

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


	def cnf_trimming(self, cnfFile, dictFile):
		"""
		Given a filename, this loads the graph with the fname in 'cnf/'
		"""
		vtoid = dict()
		idtov = dict()
		
		with open(os.path.join('cnf', dictFile), 'r') as f:
			for line in f:
				currLine = line.split()
				currLine = [int(currLine[0]), float(currLine[1]), float(currLine[2])]

				idtov[currLine[0]] = Vertex(currLine[1], currLine[2])
				vtoid[Vertex(currLine[1], currLine[2])] = currLine[0]

		n = len(vtoid)

		if n != self.n:
			raise Exception('Wrong file')
		
		vertices = [False] * n
		vertices = [None] + vertices

		with open(os.path.join('cnf', cnfFile), 'r') as f:
			for line in f:
				if line[0] == 'c':
					continue
				elif line[0] == 'p':
					maxColors = int(line.split()[2])//n
				else:					
					clause = line.split()[:-1]
					clause = [int(literal) for literal in clause]

					if len(clause) == 1:
						literal = int(fabs(clause[0] % n))
						vertices[literal] = True

					elif len(clause) == maxColors:
						literal = clause[0] % n
						if literal == 0:
							literal = n
						vertices[literal] = True

		for v in list(self.graph.nodes):
			if not vertices[vtoid[v]]:
				print("REMOVING VERTEX")
				self.remove_node(v)

		self.update()




	def draw_graph(self, fname, hard=False):
		"""
		Given a filename, this function draws this graph using LaTeX
		and opens its PDF
		"""
		tkz = TikzDocument(fname, self)
		tkz.draw(hard)

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
		UnitDistanceGraph.__init__(self)

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
		UnitDistanceGraph.__init__(self)

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
		UnitDistanceGraph.__init__(self)

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
		UnitDistanceGraph.__init__(self)
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

	def check_property(self, mode = 1):
		def colorH(self, mode):
			self.update_and_sort()
			centralH = []

			centralH.append(self.search_vertex(Vertex(0,0)))
			centralH.append(self.search_vertex(Vertex(1,0)))
			centralH.append(self.search_vertex(Vertex(0.5, sqrt(3)/2)))
			centralH.append(self.search_vertex(Vertex(-0.5, sqrt(3)/2)))
			centralH.append(self.search_vertex(Vertex(-1,0)))
			centralH.append(self.search_vertex(Vertex(-0.5, -sqrt(3)/2)))
			centralH.append(self.search_vertex(Vertex(0.5, -sqrt(3)/2)))

			print('Central H:')
			for v in centralH:
				print(v)
			
			colors = [1, 2, 3]
			
			if mode == 1: # 3 colored H
				colors += [2, 3, 2, 3]
			else:
				colors.append(4)
				if mode == 0: # No triple vertex				
					colors += [2, 4, 3]
				elif mode == 2: # 4 coloring with monochromatic triple
					colors += [3, 4, 3]

			for v in centralH:
				v.color = colors.pop(0)
				self.sorted_nodes.remove(v)

			new_nodes = centralH + self.sorted_nodes
			self.sorted_nodes = new_nodes


		self.update_and_sort()

		colorH(self, mode)
		Mproperty = ColoringGraph(self, new = False)
		return Mproperty.color()



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

class RegularPentagon(UnitDistanceGraph):
	"""
	Unit Distance regular pentagon
	"""
	def __init__(self):
		UnitDistanceGraph.__init__(self)

		# Build the pentagon

		P = UnitDistanceGraph()
		v0 = Vertex(0,0)
		v1 = Vertex(1,0)

		P.add_edge(v0, v1)

		pentagon_angle = -0.6*math.pi
		
		for i in range(4):
			v2 = v0.rotate(pentagon_angle, center = v1)
			P.add_edge(v1, v2)
			v0 = v1
			v1 = v2


		self.graph = P.graph
		self.update()


class FStar(UnitDistanceGraph):
	"""
	Unit distance graph, that is a five point star
	"""

	def __init__(self):
		UnitDistanceGraph.__init__(self)

		FS = UnitDistanceGraph()
		v0 = Vertex(1,0)
		v1 = Vertex(0,0)
		
		FS.add_edge(v0, v1)

		star_angle = -math.pi/5

		for i in range(4):
			v2 = v0.rotate(star_angle, center = v1)
			FS.add_edge(v1, v2)
			v0 = v1
			v1 = v2

		self.graph = FS.graph
		self.update()


class PetersenGraph(UnitDistanceGraph):
	"""
	Unit distance representation of the petersen graph
	"""	
	def find_angle(self):
		x = 1.49606
		
		pentagon_angle = 0.6*math.pi

		v0 = Vertex(0,0)
		v1 = Vertex(1,0)
		v2 = v1.rotate(pentagon_angle, center = v0)

		z1 = Vertex(10,10)
		z2 = Vertex(0,0)

		while not z1.isUnitDist(z2):
			x -= 0.000000001
			z1 = v0.rotate(-x, center = v1)
			z2 = v0.rotate(pentagon_angle - x, center = v2)
			if z1.dist(z2) < 1:
				input('Angle x: {} dist: {}'.format(x, z1.dist(z2)))
			if x < 1.4960521500057606:
				return False
		return x

	def __init__(self):
		UnitDistanceGraph.__init__(self)
		P = RegularPentagon()
		S = UnitDistanceGraph()
		pentagon_angle = 0.6 * math.pi

		x = self.find_angle()

		v0 = Vertex(0,0)
		v1 = Vertex(1,0)	
		
		z_list = []

		for i in range(5):
			z_list.append(v0.rotate(-x, center = v1))
			v2 = v0.rotate(-pentagon_angle, center = v1)
			v0, v1 = v1, v2


		for z in z_list:
			S.add_node(z)

		PG = P.union(S)
		PG = PG.reflection(0.5)

		self.graph = PG.graph
		self.update()




class SA(UnitDistanceGraph):
	def __init__(self, S):
		UnitDistanceGraph.__init__(self)
		# Sa = S.union(S.negative_y())
		Sa = S

		for k in range(1, 7):
			Sr = S.rotate(1, k, center = Vertex(0,0))
			Sa = Sa.union(Sr)

		Sa = Sa.union(Sa.negative_y())
		self.graph = Sa.graph
		self.update()

class G(UnitDistanceGraph):
	def __init__(self):
		UnitDistanceGraph.__init__(self)

		S = UnitDistanceGraph()
		self.fill_nodes(S)
		S.update()

		Sa = SA(S)

		Sb = Sa.rotate(4, 1)

		Y = Sa.union(Sb)

		Y.remove_node(Vertex(1/3, 0))
		Y.remove_node(Vertex(-1/3, 0))
		
		Ya = Y.rotate(16, 0.5, center = Vertex(-2,0))
		# Ya = Ya.rotate(math.pi/2, center = Vertex(-2,0))

		Yb = Y.rotate(16, -0.5, center = Vertex(-2, 0))
		# Yb = Yb.rotate(math.pi/2, center = Vertex(-2, 0))		
		
		self.graph = Ya.union(Yb).graph
		
		self.update()

	def fill_nodes(self, UDGraph):
		UDGraph.add_node(Vertex(0,0))
		UDGraph.add_node(Vertex(1/3, 0))
		UDGraph.add_node(Vertex(1, 0))
		UDGraph.add_node(Vertex(2, 0))
		UDGraph.add_node(Vertex((sqrt(33) - 3)/6, 0))
		UDGraph.add_node(Vertex(1/2, 1/sqrt(12)))
		UDGraph.add_node(Vertex(1, 1/sqrt(3)))
		UDGraph.add_node(Vertex(3/2, sqrt(3)/2))

		UDGraph.add_node(Vertex(7/6, sqrt(11)/6))
		UDGraph.add_node(Vertex(1/6, (sqrt(12) - sqrt(11))/6))
		UDGraph.add_node(Vertex(5/6, (sqrt(12) - sqrt(11))/6))

		UDGraph.add_node(Vertex(2/3, (sqrt(11) - sqrt(3))/6))
		UDGraph.add_node(Vertex(2/3, (3*sqrt(3) - sqrt(11))/6))
		UDGraph.add_node(Vertex(sqrt(33)/6, 1/sqrt(12)))

		UDGraph.add_node(Vertex((sqrt(33) + 3)/6, 1/sqrt(3)))
		UDGraph.add_node(Vertex((sqrt(33) + 1)/6, (3*sqrt(3) - sqrt(11))/6))
		UDGraph.add_node(Vertex((sqrt(33) - 1)/6, (3*sqrt(3) - sqrt(11))/6))

		UDGraph.add_node(Vertex((sqrt(33) + 1)/6, (sqrt(11) - sqrt(3))/6))
		UDGraph.add_node(Vertex((sqrt(33) - 1)/6, (sqrt(11) - sqrt(3))/6))

		UDGraph.add_node(Vertex((sqrt(33) - 2)/6, (2*sqrt(3) - sqrt(11))/6))
		UDGraph.add_node(Vertex((sqrt(33) - 4)/6, (2*sqrt(3) - sqrt(11))/6))

		UDGraph.add_node(Vertex((sqrt(33) + 13)/12, (sqrt(11) - sqrt(3))/12))
		UDGraph.add_node(Vertex((sqrt(33) + 11)/12, (sqrt(3) + sqrt(11))/12))

		UDGraph.add_node(Vertex((sqrt(33) + 9)/12, (sqrt(11) - sqrt(3))/4))
		UDGraph.add_node(Vertex((sqrt(33) + 9)/12, (3*sqrt(3) + sqrt(11))/12))

		UDGraph.add_node(Vertex((sqrt(33) + 7)/12, (sqrt(3) + sqrt(11))/12))
		UDGraph.add_node(Vertex((sqrt(33) + 7)/12, (3*sqrt(3) - sqrt(11))/12))

		UDGraph.add_node(Vertex((sqrt(33) + 5)/12, (5*sqrt(3) - sqrt(11))/12))
		UDGraph.add_node(Vertex((sqrt(33) + 5)/12, (sqrt(11) - sqrt(3))/12))

		UDGraph.add_node(Vertex((sqrt(33) + 3)/12, (3*sqrt(11) - 5*sqrt(3))/12))
		UDGraph.add_node(Vertex((sqrt(33) + 3)/12, (sqrt(3) + sqrt(11))/12))

		UDGraph.add_node(Vertex((sqrt(33) + 3)/12, (3*sqrt(3) - sqrt(11))/12))
		UDGraph.add_node(Vertex((sqrt(33) + 1)/12, (sqrt(11) - sqrt(3))/12))

		UDGraph.add_node(Vertex((sqrt(33) - 1)/12, (3*sqrt(3) - sqrt(11))/12))
		UDGraph.add_node(Vertex((sqrt(33) - 3)/12, (sqrt(11) - sqrt(3))/12))

		UDGraph.add_node(Vertex((15 - sqrt(33))/12, (sqrt(11) - sqrt(3))/4))
		UDGraph.add_node(Vertex((15 - sqrt(33))/12, (7*sqrt(3) - 3*sqrt(11))/12))

		UDGraph.add_node(Vertex((13 - sqrt(33))/12, (3*sqrt(3) - sqrt(11))/12))
		UDGraph.add_node(Vertex((11 - sqrt(33))/12, (sqrt(11) - sqrt(3))/12))
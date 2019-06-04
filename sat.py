import pycosat
import random

from graphs import *


class UDGSat():
	def __init__(self, UDG, colors):
		self.UDG = UDG
		self.toId, self.toNode = self.translate()
		self.k = colors
		self.cnf = []

		# self.only_central_clause()
		self.triangle_clause()
		self.hasColor_clause()		
		self.edgeColor_clause()

	
	def only_central_clause(self):
		v0 = self.UDG.search_vertex(Vertex(0,0))		
		self.cnf.append([self.toId[v0]])

		for color in range(1, self.k):
			self.cnf.append([-(self.toId[v0] + color * self.UDG.n)])

		for v in self.UDG.sorted_nodes:
			if v != Vertex(0,0):
				self.cnf.append([-(self.toId[v])])
	
	def save_cnf(self, fname, randomMode = False):
		with open(os.path.join('cnf', fname + '.cnf'), 'w') as f:
			f.write('c FILE: {}\n'.format(fname + '.cnf'))
			f.write('c\n')
			f.write('p cnf {} {}\n'.format(self.UDG.n * self.k, len(self.cnf)))

			if randomMode:
				random.shuffle(self.cnf)

			for clause in self.cnf:
				for literal in clause:
					f.write('{} '.format(literal))
				f.write('0\n')
		
		with open(os.path.join('cnf', fname + '.dict'), 'w') as f:			
			for i, v in self.toNode.items():
				f.write('{} {} {}\n'.format(i, v.x, v.y))

	def solve(self, color = False):
		print("Solving...")
		random.shuffle(self.cnf)
		solution = pycosat.solve(self.cnf)

		if solution == 'UNKNOWN':			
			return False

		if solution == 'UNSAT':
			print("Unsatisfiable")
			return False

		if color:
			for i in solution:
				if i > 0:
					vid = i % self.UDG.n
					if vid == 0:
						vid = self.UDG.n
						k = (i // self.UDG.n)
					else:
						k = (i // self.UDG.n) + 1
					self.toNode[vid].color = k
		
		print("Solved")
		return True

	def translate(self):
		"""
		Given a UDG, this creates dictionaries to translate
		from Vertex to index, and from index to Vertex.			
		"""
		toId = dict() # Node to id
		toNode = dict() # Id to node

		self.UDG.update_and_sort()

		i = 1
		for v in self.UDG.sorted_nodes: # Assigns an id to every vertex
			toId[v] = i
			toNode[i] = v
			i += 1

		return toId, toNode

	def triangle_score(self, triangle):
			v, w = triangle
			return self.UDG.node_score(v) + self.UDG.node_score(w)
	
	def triangle_clause(self):
		"""
		The nodes are sorted based on spindles and triangles. We look at all the triangles that contain
		vertex 1, and color the greatest one.
		"""

		u = self.UDG.sorted_nodes[0]

		neighbours = self.UDG.graph[u]

		triangles = []
		
		# Find all triangles
		for v, w in it.combinations(neighbours, 2):
			if v.isUnitDist(w):
				triangles.append((v, w))
		
		# Sort them, and take the best
		triangles.sort(reverse=True, key=self.triangle_score)

		if triangles:
			v, w = triangles[0]
			good_nodes = (u, v, w)
		elif self.UDG.graph[u]:
			v = list(self.UDG.graph.adj[u])[0]
			good_nodes = (u, v)
		else:
			good_nodes = tuple((u))

		# Add the clauses		
		for node in good_nodes:
			for color in range(self.k):
				clause = []
				if good_nodes.index(node) == color:
					clause.append(self.toId[node] + self.UDG.n * color)
				else:
					clause.append(-(self.toId[node] + self.UDG.n * color))
				self.cnf.append(clause)



	def hasColor_clause(self):
		"""
		Adds the clauses to make sure every vertex is colored
		"""
		for v in self.UDG.sorted_nodes:
			clause = []
			for color in range(self.k): 
				clause.append(self.toId[v] + self.UDG.n * color)
			self.cnf.append(clause)

	def onlyOneColor_clause(self):
		"""
		Adds the clauses to make sure every vertex has just one color.
		This is unnecessary: it's redundant
		"""
		for v in self.UDG.sorted_nodes:
			vid = self.toId[v]

			for c1 in range(self.k):				
				for c2 in range(c1 + 1, self.k):
					clause = []
					clause.append(-(vid + self.UDG.n * c1))
					clause.append(-(vid + self.UDG.n * c2))
					self.cnf.append(clause)				

	def edgeColor_clause(self):
		"""
		Adds the clauses to make sure no two neighbours have the same color
		"""
		for v, w in self.UDG.graph.edges:
			for color in range(self.k):
				clause = []
				clause.append(-(self.toId[v] + self.UDG.n*color))
				clause.append(-(self.toId[w] + self.UDG.n*color))
				self.cnf.append(clause)
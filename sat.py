import pycosat

from graphs import *


class UDGSat():
	def __init__(self, UDG, colors):
		self.UDG = UDG
		self.toId, self.toNode = self.translate()
		self.k = colors
		self.cnf = []
		
		self.hasColor_clause()
		self.onlyOneColor_clause()
		self.edgeColor_clause()

	def solve(self, color = False):
		print("Solving...")
		solution = pycosat.solve(self.cnf)

		if solution == 'UNKNOWN':
			return False

		if solution == 'UNSAT':
			return False

		if color:
			for i in solution:
				if i > 0:
					vid = i % self.UDG.n
					if vid == 0:
						vid = self.UDG.n

					k = (i // self.UDG.n) + 1
					self.toNode[vid].color = k
		
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
		Adds the clauses to make sure every vertex has just one color
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
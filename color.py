import networkx as nx
import collections

class ColoringGraph():
	"""
	Implements the backtracking coloring algorithm 
	described in de Grey's proofs
	"""
	def __init__(self, UDGraph, colors = 4, new = True):
		self.graph = nx.Graph()
		self.vtoid = dict()
		self.idtov = dict()

		self.colors = colors
		self.new = new

		self.banned_colors = collections.defaultdict(list)
		self.decolorable_nodes = collections.defaultdict(list)

		if new:
			UDGraph.update_and_sort()		
		
		self.translate_graph(UDGraph)
		self.copy_graph(UDGraph)


	def translate_graph(self, UDGraph):
		"""
		Given a UDGraph, maps every vertex to an incremental
		index, filling both vtoid and idtov dictionaries.		
		"""
		i = 0
		for v in UDGraph.sorted_nodes:
			self.vtoid[v] = i
			self.idtov[i] = v
			if self.new:
				self.graph.add_node(i, color=-1)
			else:
				self.graph.add_node(i, color=v.color)
			i +=1

	def copy_graph(self, UDGraph):
		"""
		Using self.vtoid, rewrites UDGraph in self.graph using indices.
		"""
		for (v,w) in UDGraph.graph.edges:		
			self.graph.add_edge(self.vtoid[v], self.vtoid[w])

	def get_color(self, i):
		"""
		Returns the color of the node with index i
		"""
		return self.graph.nodes[i]['color']

	def set_color(self, i, color):
		"""
		Changes the color of node with index i to color.
		"""
		self.graph.nodes[i]['color'] = color

	def isColored(self, i):
		"""
		Returns True if the node has been colored, False otherwise
		"""
		return self.get_color(i) > 0

	def available_colors(self, v):
		"""
		Returns a list with the available colors for vertex v
		"""

		available = [color for color in range(1, self.colors + 1) if color not in self.banned_colors[v]]

		for w in self.graph[v]:
			if self.get_color(w) in available:
				available.remove(self.get_color(w))
		return available

	def color_node(self, v):
		"""
		Colors node v. Then check all of its neighbours.
		If a neighbour only has one available color, color it.

		Else, decolors v and all its neighbours colored in this iteration.
		"""
		remaining_colors = self.available_colors(v)		
		if not remaining_colors:
			return False

		backtrack = False

		for color in remaining_colors:
			self.set_color(v, color)
			for w in self.graph[v]:
				if not self.isColored(w):
					if not self.available_colors(w): # If w doesn't have legal colors
						backtrack = True
						break
					elif len(self.available_colors(w)) == 1:
						colored = self.color_node(w)
						if not colored:
							backtrack = True
							break
						else:
							self.decolorable_nodes[v].append(w)
			if not backtrack:
				return True
			self.decolor_node(v)
			backtrack = False
		return False

	def decolor_node(self, v):
		"""
		Sets the color of v back to -1 (decoloring it)

		Every node colored after it gets decolored as well.
		"""
		self.set_color(v, -1)

		for w in self.decolorable_nodes[v]:
			self.decolor_node(w)
		
		self.decolorable_nodes[v] = []


	def color_graph(self):
		"""
		Colors the graph. colored_nodes is a list with all 'root' colored vertices
		(that is, a vertex for which 'color_node' method was called from this function)

		If there's a need to backtrack, the current vertex's banned_colors is cleared,
		and the previous vertex color is banned -- we then try with another color.

		This goes on until there's no vertex to backtrack -- in which case we stop,
		or when the graph is colored.
		"""
		i = 0
		n = self.graph.number_of_nodes()

		colored_nodes = []		
		while i < n:
			if not self.isColored(i):
				colored = self.color_node(i)
				if not colored: # If it couldn't be colored:	
					if colored_nodes: # If there's some v to backtrack to						
						j = colored_nodes.pop()						
						self.banned_colors[j].append(self.get_color(j))
						self.banned_colors[i] = []
						self.decolor_node(j)
						i = j
					else:
						return False
				else:					
					colored_nodes.append(i)
					i += 1
			else:				
				i += 1
		return True

	def color(self):
		"""
		Tries coloring the graph. If successful, it colors the original
		graph
		"""
		
		colored = self.color_graph()
		if colored: # Coloring was successful -> color original graph
			for i in self.graph.nodes:
				self.idtov[i].color = self.get_color(i)
			return True
		else:
			return False
import networkx as nx
import collections

class ColoringGraph():
	def __init__(self, UDGraph, colors = 4, new = True, verbose=False):
		self.graph = nx.Graph()
		self.vtoid = dict()
		self.idtov = dict()

		self.colors = colors
		self.verbose = verbose

		self.banned_colors = collections.defaultdict(list)
		self.uncolorable_nodes = collections.defaultdict(list)

		if not UDGraph.sorted_nodes:
			UDGraph.update_and_sort()

		if new:
			UDGraph.uncolor_graph()
		
		self.translate_graph(UDGraph)
		self.copy_graph(UDGraph)


	def translate_graph(self, UDGraph):		
		i = 0
		for v in UDGraph.sorted_nodes:
			self.vtoid[v] = i
			self.idtov[i] = v
			self.graph.add_node(i, color=v.color)
			i +=1

	def copy_graph(self, UDGraph):
		for (v,w) in UDGraph.graph.edges:		
			self.graph.add_edge(self.vtoid[v], self.vtoid[w])

	def get_color(self, i):
		return self.graph.nodes[i]['color']

	def set_color(self, i, color):
		self.graph.nodes[i]['color'] = color

	def isColored(self, i):
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
							self.uncolorable_nodes[v].append(w)
			if not backtrack:
				return True
			self.uncolor_node(v)
			backtrack = False
		return False

	def uncolor_node(self, v):
		self.set_color(v, -1)

		for w in self.uncolorable_nodes[v]:
			self.uncolor_node(w)
		
		self.uncolorable_nodes[v] = []


	def color_graph(self):
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
						self.uncolor_node(j)
						i = j
					else:
						if self.verbose:
							print("This graph can't be colored with {} colors".format(self.colors))
						return False
				else:					
					colored_nodes.append(i)
					i += 1
			else:				
				i += 1
		return True

	def color_original(self):
		for i in self.graph.nodes:
			self.idtov[i].color = self.get_color(i)

	def color(self):
		colored = self.color_graph()
		if colored:
			self.color_original()
			return True
		else:
			return False
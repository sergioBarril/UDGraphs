import networkx as nx
import math
import time

from graphs import Vertex, UnitDistanceGraph, H, V, W, M
from graphs import MoserSpindle, J, K, L, T, U
from graphs import RegularPentagon, PetersenGraph
from graphs import FStar

from color import ColoringGraph

from tikz import TikzDocument

import random

class App():

	def __init__(self):
		self.G = None
		self.Gname = None
		self.H = None
		self.Hname = None

	def hasGraph(self):
		return self.G is not None

	def print_options(self, opt_list):
		for opt in opt_list:
			print("{}.\t{}".format(opt_list.index(opt), opt))

	def choose_option(self, opt_funct, opt_names):
		self.print_options(opt_names)
		k = int(input("\nChoose your option:\t"))
		if k < len(opt_funct):
			self.G = opt_funct[k]()
			self.Gname = opt_names[k]
		else:
			print("Going back...")


	def graphs_menu(self):
		print("\t**********************************")
		print("\t\tUnit Distance Graphs")
		print("\t**********************************\n")

		print("U.D. Graphs:")
		graphs = ['H', 'J', 'K', 'L', 'T', 'U', 'V', 'W', 'M', "Moser's spindle", 
		"Regular Pentagon","Petersen Graph", "Pentagram", "Go back"]
		graphs_fun = [H, J, K, L, T, U, V, W, M,
		 MoserSpindle, RegularPentagon, PetersenGraph, FStar]

		self.choose_option(graphs_fun, graphs)


	def draw_graph(self, fname = None, factor = 1, hard=False):
		if fname is None:
			fname = str(random.randint(1, 10000000))

		if self.Gname == 'M':
			hard = True
		tkz = TikzDocument(fname, self.G, factor)
		tkz.draw(hard)

	def print_dimacs(self):
		self.G.save_dimacs(self.Gname)

	def saveG(self):
		self.G.update()
		self.G.save_graph(self.Gname)

	def graph_coloring(self):
		k = input("With how many colors at most? (Between 1 and 5)")
		if not k:
			k = 4
		else:
			k = int(k)

		self.G.uncolor_graph()
		self.G.color_graph(colors = k)


	def amplified_draw(self):
		factor = 2
		hard = False
		if self.Gname == 'M':
			hard = True
		self.draw_graph(factor = factor, hard= hard)

	def check_M_property(self):
		self.G.check_property()
		
	def new_coloring(self):
		self.G.update_and_sort()
		coloring = ColoringGraph(self.G)


	def pic(self):

		self.G = UnitDistanceGraph()
		self.G.add_node(Vertex(-2,0))
		self.G.add_node(Vertex(2,0))
		print(Vertex(2,0).rotate(16, 1, center = Vertex(-2,0)))
		return
		self.G = UnitDistanceGraph()
		self.G.add_node(Vertex(0,0))
		self.G.add_node(Vertex(1,0))

		self.G.add_edge(Vertex(0,0), Vertex(1,0))
		ColoringGraph(self.G).color()

		self.draw_graph(factor = 2)


		self.H = UnitDistanceGraph()
		self.H.add_edge(Vertex(0,0), Vertex(0.5, math.sqrt(3)/2))
		ColoringGraph(self.H).color()
		tkz = TikzDocument('B', self.H, factor = 2)
		tkz.draw()


		self.G = self.G.minkowskiSum(self.H)
		self.G = self.G.union(UnitDistanceGraph())

		ColoringGraph(self.G).color()

		self.draw_graph(factor = 2)

		self.G = MoserSpindle()
		ColoringGraph(self.G).color()

		self.draw_graph(factor = 2)

	def show_menu(self):
		print("\t**********************************")
		print("\t\tUnit Distance Graphs")
		print("\t**********************************\n")

		while True:
			print("Options:")
			options = ['Prebuilt graphs', 'Exit']
			options_fun = [self.graphs_menu, ]

			if self.G is not None:
				options = ['Change G', "Operations on G"]
				options_fun = [self.graphs_menu, self.operations_menu]
				options.extend(("Color Graph", "Draw PDF", "Amplified Draw", "Print coordinates", "Print DIMACS", "New Coloring"))
				options_fun.extend((self.graph_coloring, self.draw_graph, self.amplified_draw, self.saveG, self.print_dimacs, self.new_coloring))
				
				if self.Gname == "M":
					options.append("Check M property")				
					options_fun.append(self.check_M_property)								
				options.append("Exit")

			self.print_options(options)
			k = int(input("\nChoose your option:\t"))
			if k < len(options_fun):
				options_fun[k]()
			else:
				print('Bye')
				break

myApp = App()
myApp.pic()
# myApp.show_menu()
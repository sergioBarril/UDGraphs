from PyQt5 import QtCore, uic

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from GUI.main_ui import *
from GUI.graphs_ui import *
from GUI.calculator_ui import *

from graphs import *
from tikz import TikzDocument
from color import ColoringGraph

import os
import matplotlib.pyplot as plt

import random
import collections


def tex2png(formula, fname, fontsize=12, dpi=300):
	folder = os.path.join('GUI', 'texLabels')
	file = os.path.join(folder, fname + '.png')
	fig = plt.figure(figsize=(0.01, 0.01))
	fig.text(0, 0, r'${}$'.format(formula), fontsize=fontsize)

	fig.savefig(file, dpi=dpi, transparent=True, format='png',
		bbox_inches='tight', pad_inches=0.0, frameon=False)
	return file

class GraphsDialog(QDialog, Ui_GraphsDialog):
	def __init__(self, parent):
		super(GraphsDialog, self).__init__(parent)
		self.setupUi(self)

		self.G = None
		self.Gname = None		
		
		self.btn_H.clicked.connect(lambda ignore, graph='H' : self.setGraph(graph))
		self.btn_J.clicked.connect(lambda ignore, graph='J' : self.setGraph(graph))
		self.btn_K.clicked.connect(lambda ignore, graph='K' : self.setGraph(graph))
		self.btn_L.clicked.connect(lambda ignore, graph='L' : self.setGraph(graph))
		self.btn_M.clicked.connect(lambda ignore, graph='M' : self.setGraph(graph))
		self.btn_T.clicked.connect(lambda ignore, graph='T' : self.setGraph(graph))
		self.btn_U.clicked.connect(lambda ignore, graph='U' : self.setGraph(graph))
		self.btn_V.clicked.connect(lambda ignore, graph='V' : self.setGraph(graph))
		self.btn_W.clicked.connect(lambda ignore, graph='W' : self.setGraph(graph))
		self.btn_moser.clicked.connect(lambda ignore, graph="Moser" : self.setGraph(graph))
		self.btn_pentagon.clicked.connect(lambda ignore, graph='C5' :self.setGraph(graph))
		self.btn_pentagram.clicked.connect(lambda ignore, graph='Pentagram' : self.setGraph(graph))
		self.btn_petersen.clicked.connect(lambda ignore, graph='PG' : self.setGraph(graph))
		
	def setGraph(self, graph):
		graphs = ['H', 'J', 'K', 'L', 'T', 'U', 'V', 'W', 'M', "Moser", 
		"C5","PG", "Pentagram"]

		graphs_fun = [H, J, K, L, T, lambda : print('Not implemented'), V, W, M,
		 MoserSpindle, RegularPentagon, PetersenGraph, FStar]
		
		for i in range(len(graphs)):
			if graph == graphs[i]:
				self.G = graphs_fun[i]()
				self.Gname = graph
		self.close()

	def getGraph(self):
		return self.G, self.Gname

class GraphCalculatorDialog(QDialog, Ui_Graph_Calculator):
	def __init__(self, parent):
		super(GraphCalculatorDialog, self).__init__(parent)
		self.setupUi(self)

		self.G = parent.G
		self.Gname = parent.Gname
		self.Gformula = parent.Gformula

		self.H = None

		self.todo = collections.deque()
		self.todo_dict = {0: self.union, 1: self.minkowski, 2: self.trim,
			3: self.rotate, 4: self.rotate_sp}

		self.binary = None

		self.btn_union.clicked.connect(self.union_mode)
		self.btn_minkowski.clicked.connect(self.minkowski_mode)

		self.connect_graphs()
		self.hide_graphs()
		self.set_names()



	#************** BINARY OPERATIONS *************
	# Toggle graph buttons
	def union_mode(self):
		self.show_graphs()
		self.binary = 0

	def minkowski_mode(self):
		self.show_graphs()
		self.binary = 1

	def union_enqueue(self, graph):
		if self.todo and self.todo[-1][0] == 1:
			self.Gformula = '({})'.format(self.Gformula)
		self.Gformula += ' \\cup {}'.format(graph)
		self.todo.append((0, graph))		
		self.Gname += 'u' + graph
		self.set_names()

	def minkowski_enqueue(self, graph):
		if self.todo and self.todo[-1][0] == 0:
			self.Gformula = '({})'.format(self.Gformula)
		self.Gformula += ' \\oplus {}'.format(graph)
		self.todo.append((1, graph))		
		self.Gname += '+' + graph
		self.set_names()


	def union(self, graph):
		F = self.findGraph(graph)
		return self.G.union(F)

	def minkowski(self, graph):
		F = self.findGraph(graph)
		return self.G.minkowskiSum(F)

	def binary_router(self, graph):
		if self.binary == 0:
			self.union_enqueue(graph)
		elif self.binary == 1:
			self.minkowski_enqueue(graph)		
		self.binary = None
		self.hide_graphs()

	# ************** MONARY OPERATORS ************
	def trim(self, d):
		pass

	def rotate(self, angle):
		pass

	def rotate_sp(self, i, k):
		pass


	# **************** UTILITIES *****************

	def connect_graphs(self):
		self.btn_H.clicked.connect(lambda ignore, graph='H' : self.binary_router(graph))
		self.btn_J.clicked.connect(lambda ignore, graph='J' : self.binary_router(graph))
		self.btn_K.clicked.connect(lambda ignore, graph='K' : self.binary_router(graph))
		self.btn_L.clicked.connect(lambda ignore, graph='L' : self.binary_router(graph))
		self.btn_M.clicked.connect(lambda ignore, graph='M' : self.binary_router(graph))
		self.btn_T.clicked.connect(lambda ignore, graph='T' : self.binary_router(graph))
		self.btn_U.clicked.connect(lambda ignore, graph='U' : self.binary_router(graph))
		self.btn_V.clicked.connect(lambda ignore, graph='V' : self.binary_router(graph))
		self.btn_W.clicked.connect(lambda ignore, graph='W' : self.binary_router(graph))
		self.btn_moser.clicked.connect(lambda ignore, graph="Moser" : self.binary_router(graph))
		self.btn_pentagon.clicked.connect(lambda ignore, graph='C5' :self.binary_router(graph))
		self.btn_pentagram.clicked.connect(lambda ignore, graph='Pentagram' : self.binary_router(graph))
		self.btn_petersen.clicked.connect(lambda ignore, graph='PG' : self.binary_router(graph))


	def show_graphs(self):	
		self.graphs.show()

	def hide_graphs(self):
		self.graphs.hide()

	def set_names(self):
		folder = os.path.join('GUI', 'texLabels')	
		fontsize = 12
		
		formula, name = "G = {}".format(self.Gformula), self.Gname			

		if len(name) > 5:
			fontsize = 8
		
		img = tex2png(formula, name, fontsize)
		lb_pixmap = QPixmap(img)
		self.lb_calc.setPixmap(lb_pixmap)

	def findGraph(self, graph):
		graphs = ['H', 'J', 'K', 'L', 'T', 'U', 'V', 'W', 'M', "Moser", 
		"C5","PG", "Pentagram"]

		graphs_fun = [H, J, K, L, T, lambda : print('Not implemented'), V, W, M,
		 MoserSpindle, RegularPentagon, PetersenGraph, FStar]
		
		for i in range(len(graphs)):
			if graph == graphs[i]:
				return graphs_fun[i]()





# *********************************************
#				MAIN WINDOW
# *********************************************

class MainWindow(QMainWindow, Ui_MainWindow):
	def __init__(self):
		super(MainWindow, self).__init__()
		self.setupUi(self)
		self.setWindowIcon(QIcon(os.path.join('GUI', 'appicon.ico')))

		self.G = None
		self.Gname = None
		self.Gformula = r"\mathrm{None}"

		self.toggle_invisible()
		self.set_names()

		self.btn_graph.clicked.connect(self.constructed_graphs)
		self.btn_operators.clicked.connect(self.operations)
		self.btn_load.clicked.connect(self.load_graph)

		self.btn_draw.clicked.connect(self.draw_graph)
		self.btn_color.clicked.connect(self.color_graph)
		self.btn_print.clicked.connect(self.print_graph)

	# **************** UTILITIES *****************

	def toggle_invisible(self):
		if self.G is None:			
			self.G_options.hide()
			self.btn_operators.hide()
		else:
			self.G_options.show()
			self.btn_operators.show()
			self.btn_graph.setText('Create new graph G')

	def set_names(self):
		folder = os.path.join('GUI', 'texLabels')	
		fontsize = 12
		

		formula, name = "G = {}".format(self.Gformula), self.Gname			

		if name is None:
			name = 'GNone'

		if len(name) > 5:
			fontsize = 8
		
		img = tex2png(formula, name, fontsize)
		lb_G_pixmap = QPixmap(img)
		self.lb_G.setPixmap(lb_G_pixmap)

	# ***************** NEW GRAPHS **********************

	def constructed_graphs(self):	
		graphsDialog = GraphsDialog(self)		
		graphsDialog.show()
		graphsDialog.exec()
		
		self.G, self.Gname = graphsDialog.getGraph()
		self.Gformula = self.Gname
		self.toggle_invisible()
		self.set_names()

	def load_graph(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file, _ = QFileDialog.getOpenFileName(self,"Load Graph", "","Edge files (*.e);;All Files (*)", options=options)
			
		if file:
			filesCorrect = file[-2:] == '.e'

			if not filesCorrect:
				msg = QMessageBox()
				msg.setIcon(QMessageBox.Warning)
				msg.setText('Invalid Files')
				msg.setInformativeText('Please choose a .e file')
				msg.setWindowTitle("Graph Loading Failed")
				msg.setWindowIcon(QIcon(os.path.join('GUI', 'appicon.ico')))
				msg.setStandardButtons(QMessageBox.Ok)
				msg.exec()
			else:
				self.G = UnitDistanceGraph()
				self.G.load_graph(os.path.basename(file)[:-2])
				self.Gname = os.path.basename(file)[:-2]
				self.set_names()


	def operations(self):
		operationsDialog = GraphCalculatorDialog(self)
		operationsDialog.show()
		operationsDialog.exec()


	# ************** OPTIONS ON G ******************
	def draw_graph(self):
		if self.randomName.isChecked():
			fname = str(random.randint(1, 1000000000))
		else:
			fname = self.Gname
		factor = self.drawScaling.value()			
		tkz = TikzDocument(fname, self.G, factor)
		tkz.draw()


	def color_graph(self):
		maxColors = self.maxColors.value()
		cg = ColoringGraph(self.G, colors = maxColors, new = True)

		msg = QMessageBox()
		msg.setWindowIcon(QIcon(os.path.join('GUI', 'appicon.ico')))
		if cg.color():
			msg.setIcon(QMessageBox.Information)			
			msg.setText("Graph successfully colored with at most {} colors.".format(maxColors))
			msg.setWindowTitle("Coloring Successful")
			msg.setStandardButtons(QMessageBox.Ok)

		else:
			msg.setIcon(QMessageBox.Warning)
			msg.setText("This graph can't be colored with {} colors.".format(maxColors))
			msg.setWindowTitle("Coloring Failed")
			msg.setStandardButtons(QMessageBox.Ok)	
		msg.exec()

	def print_graph(self):		
		self.G.save_graph(self.Gname)		


def main():
	app = QApplication([])

	mw = MainWindow()
	mw.show()
	app.exec()

if __name__ == '__main__':
	main()
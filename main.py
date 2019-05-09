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

import re


def tex2png(formula, fname, fontsize=12, dpi=300):
	folder = os.path.join('GUI', 'texLabels')
	file = os.path.join(folder, fname + '.png')
	fig = plt.figure(figsize=(0.01, 0.01))
	fig.text(0, 0, r'${}$'.format(formula), fontsize=fontsize)

	fig.savefig(file, dpi=dpi, transparent=True, format='png',
		bbox_inches='tight', pad_inches=0.0, frameon=False)

	plt.close(fig)
	return file

class GraphsDialog(QDialog, Ui_GraphsDialog):
	def __init__(self, parent):
		super(GraphsDialog, self).__init__(parent)
		self.setupUi(self)

		self.G = None
		self.Gname = None

		self.graph_built = False
		
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
		
		self.graph_built = True
		self.close()

	def getGraph(self):
		return self.G, self.Gname



# **********************************************************
#	 					GRAPH  OPERATIONS
# **********************************************************

class GraphCalculatorDialog(QDialog, Ui_Graph_Calculator):
	def __init__(self, parent):
		super(GraphCalculatorDialog, self).__init__(parent)
		self.setupUi(self)

		self.G = parent.G
		self.Gname = parent.Gname
		self.Gformula = parent.Gformula
		
		self.todo = collections.deque()
		self.todo_dict = {0: self.union, 1: self.minkowski, 2: self.trim,
			3: self.rotate, 4: self.rotate}

		
		self.graph_built = False

		self.btn_union.clicked.connect(self.union_mode)
		self.btn_minkowski.clicked.connect(self.minkowski_mode)

		self.btn_trim.clicked.connect(self.trim_enqueue)
		self.btn_rotate.clicked.connect(self.rotate_enqueue)
		self.btn_rotate_sp.clicked.connect(self.rotate_sp_enqueue)

		self.btn_undo.clicked.connect(self.undo)

		self.btn_ok.clicked.connect(self.build_graph)
		self.btn_cancel.clicked.connect(self.cancel)

		self.connect_graphs()
		self.hide_graphs()
		self.set_names()


	def build_graph(self):
		self.run()
		self.graph_built = True
		self.close()		

	def cancel(self):
		self.close()		

	def getData(self):
		if self.graph_built:
			return self.G, self.Gname, self.Gformula

	def undo(self):
		if self.todo:
			undone = self.todo.pop()
			if undone[0] < 2: # UNION OR MINKOWSKI
				new_formula = self.Gformula
				new_name = self.Gname

				new_formula = new_formula[:-len(undone[2])]
				new_name = new_name[:-len(undone[3])]
				if new_formula[0] == '(' and new_formula[-1] == ')':
					new_formula = new_formula[1:-1]
					new_name = new_name[1:-1]
									
			elif undone[0] == 2: # TRIM
				first_parenth = self.Gformula.find('(')

				new_formula = self.Gformula[first_parenth + 1:]
				new_name = self.Gname[5:-1]

				last_comma = new_formula.rfind(',')
				new_formula = new_formula[:last_comma]


			elif undone[0] == 3: # ROTATE
				first_parenth = self.Gformula.find('(')
				new_formula = self.Gformula[first_parenth + 1:]
				new_name = self.Gname[4:-1]

				last_comma = new_formula.rfind(',')
				new_formula = new_formula[:last_comma]

			elif undone[0] == 4: # ROTATE ESP
				new_formula = self.Gformula[len(undone[2]):-1]
				new_name = self.Gname[4:-1]


			self.Gformula = new_formula
			self.Gname = new_name


		self.set_names()


	def run(self):
		while self.todo:
			action = self.todo.popleft()
			self.G = self.todo_dict[action[0]](action[1])		

	

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
		added_formula = ' \\cup {}'.format(graph)
		added_name = 'u' + graph
		self.todo.append((0, graph, added_formula, added_name))

		self.Gformula += added_formula
		self.Gname += added_name
		self.set_names()

	def minkowski_enqueue(self, graph):
		if self.todo and self.todo[-1][0] == 0:
			self.Gformula = '({})'.format(self.Gformula)
			self.Gname = '({})'.format(self.Gname)
		
		added_formula = ' \\oplus {}'.format(graph)
		added_name = '+' + graph

		self.todo.append((1, graph, added_formula, added_name))		
		self.Gformula += added_formula
		self.Gname += added_name
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

	def sanitize_input(self, plain_text):		
		# This handles sqrts:
		formula_ready = re.sub(r'sqrt\((.+)\)', r'\\sqrt{\1} ', plain_text)
		eval_ready = re.sub(r'sqrt\(', r'math.sqrt(', plain_text)

		# This handles pi:
		formula_ready = re.sub(r'pi', r'\\pi ', formula_ready)
		eval_ready = re.sub(r'pi', r'math.pi', eval_ready)

		# This handles exponentiation:
		eval_ready = re.sub(r'\^', r'**', eval_ready)

		# This handles franctions:
		formula_ready = re.sub(r'{(.+)}/{(.+)}', r'\\frac{\1}{\2} ', formula_ready)

		return formula_ready, eval_ready
	
	def trim_enqueue(self):

		self.Gformula = '\\mathrm{{trim}}({}, '.format(self.Gformula)

		plain_text = self.input_trim_dist.text()
		formula_ready, eval_ready = self.sanitize_input(plain_text)
		
		self.Gformula += formula_ready + ')'
		self.Gname = 'trim({})'.format(self.Gname)

		self.todo.append((2, eval(eval_ready)))
		self.set_names()

	def rotate_enqueue(self):

		self.Gformula = '\\mathrm{{rot}}({}, '.format(self.Gformula)

		plain_text = self.input_rotate_angle.text()
		formula_ready, eval_ready = self.sanitize_input(plain_text)

		self.Gformula += formula_ready + ')'
		self.Gname = 'rot({})'.format(self.Gname)

		self.todo.append((3, eval(eval_ready)))
		self.set_names()

	def rotate_sp_enqueue(self):
		i = self.rotate_i.value()
		k = self.rotate_k.value()

		new_formula = '\\theta_{{{}}}'.format(i)

		if k != 1:
			new_formula += '^{{{}}}'.format(k)
		new_formula += '('
		
		self.Gformula = new_formula + '{})'.format(self.Gformula)
		self.Gname = 'rot({})'.format(self.Gname)

		angle = math.acos((2 * i - 1) / (2 * i))
		angle *= k

		self.todo.append((4, angle, new_formula))
		self.set_names()

	def trim(self, r):		
		return self.G.trim(r)

	def rotate(self, angle):
		return self.G.rotate(angle)

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
		self.graphs.setEnabled(True)

	def hide_graphs(self):
		self.graphs.setEnabled(False)

	def set_names(self):
		folder = os.path.join('GUI', 'texLabels')	
		fontsize = 12
		
		formula, name = "G = {}".format(self.Gformula), self.Gname			

		if len(name) > 5:
			fontsize = 8
		
		img = tex2png(formula, name, fontsize)
		lb_pixmap = QPixmap(img)
		self.lb_calc.setPixmap(lb_pixmap)

		if not self.todo:
			self.btn_ok.setEnabled(False)
		else:
			self.btn_ok.setEnabled(True)

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
		self.btn_clear.clicked.connect(self.clear)		

		self.btn_M_property.clicked.connect(self.check_M_property)

	# **************** UTILITIES *****************

	def clear(self):
		self.G = None
		self.Gname = None
		self.Gformula = r"\mathrm{None}"
		
		self.set_names()
		self.toggle_invisible()

	def toggle_invisible(self):
		self.btn_M_property.hide()

		if self.G is None:			
			self.G_options.hide()
			self.btn_operators.hide()
		else:
			if self.Gname == 'M':
				self.btn_M_property.show()
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

		if self.G is None:
			self.lb_n.setText('n = 0')
			self.lb_m.setText('m = 0')
		else:
			self.lb_n.setText('n = {}'.format(self.G.n))
			self.lb_m.setText('m = {}'.format(self.G.m))

	# ***************** NEW GRAPHS **********************

	def constructed_graphs(self):	
		graphsDialog = GraphsDialog(self)		
		graphsDialog.show()
		graphsDialog.exec()
		
		if graphsDialog.graph_built:
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
				self.Gformula = self.Gname
				self.set_names()


	def operations(self):
		operationsDialog = GraphCalculatorDialog(self)
		operationsDialog.show()
		operationsDialog.exec()
		if operationsDialog.graph_built:
			self.G, self.Gname, self.Gformula = operationsDialog.getData()
			self.set_names()


	# ************** OPTIONS ON G ******************
	def draw_graph(self):
		if self.randomName.isChecked():
			fname = str(random.randint(1, 1000000000))
		else:
			fname = self.Gname

		hard = self.G.n >= 800		
		factor = self.drawScaling.value()			
		tkz = TikzDocument(fname, self.G, factor)
		tkz.draw(hard)


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

		if self.dimacs.isChecked():
			self.G.save_dimacs(self.Gname)

	def check_M_property(self):
		msg = QMessageBox()
		msg.setWindowIcon(QIcon(os.path.join('GUI', 'appicon.ico')))
		msg.setWindowTitle("M property check")
		msg.setStandardButtons(QMessageBox.Ok)

		if not self.G.check_property(1):						
			msg.setIcon(QMessageBox.Information)			
			msg.setText("M couldn't be colored.")
			msg.setInformativeText("When the central H is 3-colored, M can't be 4-colored")
		else:
			msg.setIcon(QMessageBox.Critical)
			msg.setText("M has been 4-colored, although it shouldn't have")
			msg.setText("Something must be wrong")			
		msg.exec()

		if not self.G.check_property(2):
			msg.setIcon(QMessageBox.Information)			
			msg.setText("M couldn't be colored.")
			msg.setInformativeText("When the central H has 3 vertices of the same color, M can't be 4-colored")
		else:
			msg.setIcon(QMessageBox.Critical)
			msg.setText("M has been 4-colored, although it shouldn't have")
			msg.setText("Something must be wrong")			
		msg.exec()



def main():
	app = QApplication([])

	mw = MainWindow()
	mw.show()
	app.exec()

if __name__ == '__main__':
	main()
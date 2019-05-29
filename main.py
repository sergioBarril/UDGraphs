from PyQt5 import QtCore, uic

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from GUI.main_ui import *
from GUI.graphs_ui import *
from GUI.calculator_ui import *

from graphs import *
from tikz import TikzDocument
from color import ColoringGraph
from sat import UDGSat

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


# **********************************************************
#	 					NEW GRAPH
# **********************************************************
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
		self.btn_G.clicked.connect(lambda ignore, graph='G' : self.setGraph(graph))
		self.btn_moser.clicked.connect(lambda ignore, graph="Moser" : self.setGraph(graph))
		self.btn_pentagon.clicked.connect(lambda ignore, graph='C5' :self.setGraph(graph))
		self.btn_pentagram.clicked.connect(lambda ignore, graph='Pentagram' : self.setGraph(graph))
		self.btn_petersen.clicked.connect(lambda ignore, graph='PG' : self.setGraph(graph))
		
	def setGraph(self, graph):
		graphs = ['H', 'J', 'K', 'L', 'T', 'U', 'V', 'W', 'M', "Moser", 
		"C5","PG", "Pentagram", 'G']

		graphs_fun = [H, J, K, L, T, lambda : print('Not implemented'), V, W, M,
		 MoserSpindle, RegularPentagon, PetersenGraph, FStar, G]
		
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
	"""
	Dialog that given the previous graph, builds a new graph
	There are several operators:
		· Rotation
		· Special Rotation
		· Trim
		· Union
		· Minkowski Sum
	"""
	def __init__(self, parent):
		super(GraphCalculatorDialog, self).__init__(parent)
		self.setupUi(self)


		self.G = parent.G
		self.Gname = parent.Gname
		self.Gformula = parent.Gformula


		self.isLoading = False

		self.binary = None
		self.memMode = False


		self.mainTodo = None

		self.todo = collections.deque()
		self.memTodo = collections.deque() # Each memory save will be a new deque.

		self.todo_dict = {0: self.union, 1: self.minkowski, 2: self.trim,
			3: self.rotate, 4: self.rotate, 6: self.new_graph}
		
		self.graph_built = False


		self.btn_merge.hide()
		self.lb_mem.hide()
		self.btn_mem.setEnabled(False)

		self.connect_buttons()		
		self.update()


	def build_graph(self):
		"""
		Executes all commands stored in the stack
		It then closes the dialog.
		"""
		self.run()
		self.graph_built = True
		self.close()

	def cancel(self):
		"""
		Closes the dialog without building any graph
		"""
		self.close()		

	def getData(self):
		"""
		Returns the resulting graph
		"""
		if self.graph_built:
			return self.G, self.Gname, self.Gformula

	def undo(self):
		"""
		Undoes last set action. That is, it removes last
		added command on the stack
		"""
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

			elif undone[0] == 5: # STORE IN MEMORY
				old_G = self.mem.pop()
				self.G, new_name, new_formula = old_G

			elif undone[0] == 6: # NEW GRAPH
				# Reinitialize the graph
				new_name = 'None'
				new_formula = r"\mathrm{None}"

				self.binary = None
			
			elif undone[0] == 7: # MERGE
				self.memMode = True
				self.mainTodo = self.todo #Store main deque
				self.todo = self.memTodo.pop() #Focus on memory deque
				
				new_name, new_formula = undone[1]

		
		elif self.mainTodo:
			self.todo = self.mainTodo
			self.Gname, self.Gformula = self.todo.pop()[1]
			self.memMode = False
			self.undo()	

			new_formula = self.Gformula
			new_name = self.Gname
		else:
			return

		self.Gformula = new_formula
		self.Gname = new_name

		self.update()

	def run(self):
		"""
		Runs all operations
		"""
		while self.todo:
			action = self.todo.popleft()
			if not(action[0] == 5 or action[0] == 7):
				self.G = self.todo_dict[action[0]](action[1])

	def run_mem(self):
		"""
		Builds next graph in memory, and returns it.
		"""

		# Save current graph
		oldG = self.G

		if not self.memTodo:
			raise Exception("Empty memory")

		# Take the instructions to build the next graph.
		newTodo = self.memTodo.popleft()

		# Iterate to build the new graph.
		self.G = UnitDistanceGraph()		
		
		while newTodo:
			action = newTodo.popleft()
			self.G = self.todo_dict[action[0]](action[1])

		# Let G be its former self.
		F, self.G = self.G, oldG
		
		self.todo.popleft() # (5, ...)
		self.todo.popleft() # (7, ...)
		return F	


		

	# ******************* ENQUEUE **************

	def union_enqueue(self, graph):
		# If the graph comes from a file:
		if graph is None and self.isLoading:
			try:
				file, filename = self.load_graph()
				if filename is not None:
					graph = filename
				else:
					return 1
			except Exception as e:				
				msg = QMessageBox()
				msg.setIcon(QMessageBox.Warning)
				msg.setText('Invalid Files')
				msg.setInformativeText('Please choosbe a .e file')
				msg.setWindowTitle("Graph Loading Failed")
				msg.setWindowIcon(QIcon(os.path.join('GUI', 'appicon.ico')))
				msg.setStandardButtons(QMessageBox.Ok)
				msg.exec()
				return
			finally:
				self.isLoading = False
		else:
			filename = None

		# If something has been done and it's not a union, put parentheses
		if self.todo and self.todo[-1][0] == 1:
			self.Gformula = '({})'.format(self.Gformula)
			self.Gname = '({})'.format(self.Gname)
		
		# Add the \cup to the formula, and the name
		added_formula = ' \\cup '
		added_name = 'u'

		if graph is not None:
			added_formula += graph
			added_name += graph
			
		# Append to the todolist
		self.todo.append((0, (graph, self.Gformula, filename), added_formula, added_name))

		# Update attributes
		self.Gformula += added_formula
		self.Gname += added_name
		self.update()

	def minkowski_enqueue(self, graph):
		# If the graph comes from a file:
		if graph is None and self.isLoading:
			try:
				file, filename = self.load_graph()
				if filename is not None:
					graph = filename
				else:
					return
			except Exception as e:				
				msg = QMessageBox()
				msg.setIcon(QMessageBox.Warning)
				msg.setText('Invalid Files')
				msg.setInformativeText('Please choose a .e file')
				msg.setWindowTitle("Graph Loading Failed")
				msg.setWindowIcon(QIcon(os.path.join('GUI', 'appicon.ico')))
				msg.setStandardButtons(QMessageBox.Ok)
				msg.exec()
				return
			finally:
				self.isLoading = False
		else:
			filename = None
		
		# If something has been done and it's not a sum, put parentheses
		if self.todo and self.todo[-1][0] == 0:
			self.Gformula = '({})'.format(self.Gformula)
			self.Gname = '({})'.format(self.Gname)
		
		# Add the \oplus to the formula and the name
		added_formula = ' \\oplus '
		added_name = '+ {}'

		if graph is not None:
			added_formula += graph
			added_name += graph

		# Append to the todolist
		self.todo.append((1, (graph, self.Gformula, filename), added_formula, added_name))

		# Update the attributes
		self.Gformula += added_formula
		self.Gname += added_name
		self.update()

	def trim_enqueue(self):
		# Add the trim statement, opening a parentheses
		self.Gformula = '\\mathrm{{trim}}({}, '.format(self.Gformula)

		# Get input, and translate the sqrts and pis
		plain_text = self.input_trim_dist.text()
		formula_ready, eval_ready = self.sanitize_input(plain_text)
		
		# Update formula, closing the parentheses
		self.Gformula += formula_ready + ')'
		self.Gname = 'trim({})'.format(self.Gname)

		# Append to the todolist and update
		self.todo.append((2, eval(eval_ready)))
		self.update()

	def rotate_enqueue(self):
		# Add the rot statement, opening a parentheses
		self.Gformula = '\\mathrm{{rot}}({}, '.format(self.Gformula)

		# Get input, and translate the sqrts and pis
		plain_text = self.input_rotate_angle.text()
		formula_ready, eval_ready = self.sanitize_input(plain_text)

		# Update formula, closing the parentheses
		self.Gformula += formula_ready + ')'
		self.Gname = 'rot({})'.format(self.Gname)

		# Append to the todolist and update
		self.todo.append((3, eval(eval_ready)))
		self.update()

	def rotate_sp_enqueue(self):
		# Get the i and k values
		i = self.rotate_i.value()
		k = self.rotate_k.value()

		# Formatting theta
		new_formula = '\\theta_{{{}}}'.format(i)

		if k != 1:
			new_formula += '^{{{}}}'.format(k)
		new_formula += '('
		
		# Append formula and name
		self.Gformula = new_formula + '{})'.format(self.Gformula)
		self.Gname = 'rot({})'.format(self.Gname)

		# Calculate the angle
		angle = math.acos((2 * i - 1) / (2 * i))
		angle *= k

		# Update attributes and append to todolist
		self.todo.append((4, angle, new_formula))
		self.update()


	def store_mem_enqueue(self):
		# Append either union or minkowski to old list
		self.binary_router(None)

		# Change focused deque, and append name and formula:
		self.todo.append((5, (self.Gname, self.Gformula)))
		self.mainTodo = self.todo	
		self.todo = collections.deque()

		# Reinitialize the graph
		self.Gname = 'None'
		self.Gformula = r"\mathrm{None}"

		self.binary = None

		# Enable elements
		self.memMode = True				

		self.update()

	def merge(self):
		# Add to memTodo, and restore main todo
		self.memTodo.append(self.todo)
		self.todo = self.mainTodo
		self.mainTodo = None

		# Merge names and formulas
		oldname, oldformula = self.todo[-1][1]

		self.todo.append((7, (self.Gname, self.Gformula)))

		self.Gname = oldname + '[{}]'.format(self.Gname)
		self.Gformula = oldformula + '[{}]'.format(self.Gformula)

		# Change mode:
		self.memMode = False		
		self.update()


	def new_graph_enqueue(self, graph):
		# If the graph comes from a file:
		if self.isLoading:
			try:
				file, filename = self.load_graph()
				if filename is not None:
					self.todo.append((6, (filename, filename, filename)))
					self.Gformula = filename
					self.Gname = filename
				self.update()
			except Exception as e:				
				msg = QMessageBox()
				msg.setIcon(QMessageBox.Warning)
				msg.setText('Invalid Files')
				msg.setInformativeText('Please choose a .e file')
				msg.setWindowTitle("Graph Loading Failed")
				msg.setWindowIcon(QIcon(os.path.join('GUI', 'appicon.ico')))
				msg.setStandardButtons(QMessageBox.Ok)
				msg.exec()
				return
			finally: 
				self.isLoading = False

		# If the graph comes from a graph button:
		if graph is not None:
			self.Gformula = graph
			self.Gname = graph
			self.todo.append((6, (graph, graph, None)))
			self.update()		

	#************** BINARY OPERATIONS *************
	# Toggle graph buttons
	def union_mode(self):				
		self.binary = 0
		self.update()

	def minkowski_mode(self):				
		self.binary = 1
		self.update()


	def union(self, graph):
		if graph[2] is not None:
			F = UnitDistanceGraph()
			F.load_graph(graph[2])
			if F is None:
				print(graph[2])
		elif graph[0] is None:
			F = self.run_mem()
		else:
			F = self.findGraph(graph[0])
		return self.G.union(F)

	def minkowski(self, graph):
		if graph[2] is not None:
			F = UnitDistanceGraph()
			F.load_graph(graph[2])
		elif graph[0] is None:
			F = self.run_mem()
		else:
			F = self.findGraph(graph[0])
		return self.G.minkowskiSum(F)

	def binary_router(self, graph):
		if self.binary is None:
			self.new_graph_enqueue(graph)
		elif self.binary == 0:
			self.union_enqueue(graph)
		elif self.binary == 1:
			self.minkowski_enqueue(graph)
		self.binary = None
		self.update()

	# ************** MONARY OPERATORS ************

	def new_graph(self, graph):
		if graph[2] is None:
			return self.findGraph(graph[0])
		else:
			F = UnitDistanceGraph()
			F.load_graph(graph[2])
			return F

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

	def trim(self, r):		
		return self.G.trim(r)

	def rotate(self, angle):
		return self.G.rotate(angle)

	
	# **************** UTILITIES *****************

	def load_button(self):
		self.isLoading = True
		self.binary_router(None)


	def connect_buttons(self):
		# Utilities
		self.btn_undo.clicked.connect(self.undo)
		self.btn_mem.clicked.connect(self.store_mem_enqueue)

		self.btn_merge.clicked.connect(self.merge)		

		self.btn_ok.clicked.connect(self.build_graph)
		self.btn_cancel.clicked.connect(self.cancel)


		# Operators
		self.btn_union.clicked.connect(self.union_mode)
		self.btn_minkowski.clicked.connect(self.minkowski_mode)

		self.btn_trim.clicked.connect(self.trim_enqueue)
		self.btn_rotate.clicked.connect(self.rotate_enqueue)
		self.btn_rotate_sp.clicked.connect(self.rotate_sp_enqueue)

		# Graphs
		self.btn_load.clicked.connect(self.load_button)

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


	def toggle_memMode(self, state):
		if state:
			self.lb_mem.show()
			self.btn_merge.show()
		else:
			self.lb_mem.hide()
			self.btn_merge.hide()		

	def update(self):
		self.set_names()

		# Show or hide merge button and stored formula.
		self.toggle_memMode(self.memMode)

		# Enable or disable undo buttons
		undoBool = bool(self.todo or self.mainTodo)
		okBool = bool(self.todo and not self.memMode)
		self.btn_undo.setEnabled(undoBool)
		self.btn_ok.setEnabled(okBool)

		# Enable operators
		operatorsBool = not bool(self.memMode and not self.todo)
		self.grid_operators.setEnabled(operatorsBool)

		# Enable graph buttons
		graphsBool = self.binary == 1 or self.binary == 0 or (self.memMode and not self.todo)
		self.graphs.setEnabled(graphsBool)
		self.btn_load.setEnabled(graphsBool)
		self.btn_merge.setEnabled(bool(self.todo))
		self.btn_mem.setEnabled(graphsBool and self.binary is not None)


	def set_names(self):
		folder = os.path.join('GUI', 'texLabels')	
		
		# The beginning of the formula
		graph = "G"
		if self.memMode:
			graph = "M"

		# MAIN FOCUS
		fontsize = 12	
		formula, name = "{} = {}".format(graph, self.Gformula), self.Gname
		
		if len(name) > 5:
			fontsize = 8

		img = tex2png(formula, name, fontsize)
		lb_pixmap = QPixmap(img)
		self.lb_calc.setPixmap(lb_pixmap)

		# SECONDARY FOCUS
		if self.memMode:
			self.lb_mem.show()
			memFormula, memName = "G = {}".format(self.mainTodo[-1][1][1]), self.mainTodo[-1][1][0]
			
			memFontsize = 10
			if len(memName) > 5:
				memFontsize = 6
			
			img = tex2png(memFormula, memName, memFontsize)
			
			lb_pixmap = QPixmap(img)
			self.lb_mem.setPixmap(lb_pixmap)

	def findGraph(self, graph):
		graphs = ['H', 'J', 'K', 'L', 'T', 'U', 'V', 'W', 'M', "Moser", 
		"C5","PG", "Pentagram"]

		graphs_fun = [H, J, K, L, T, lambda : print('Not implemented'), V, W, M,
		 MoserSpindle, RegularPentagon, PetersenGraph, FStar]
		
		for i in range(len(graphs)):
			if graph == graphs[i]:
				return graphs_fun[i]()

	def load_graph(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file, _ = QFileDialog.getOpenFileName(self,"Load Graph", "","Edge files (*.e);;All Files (*)", options=options)
			
		if file:
			filesCorrect = file[-2:] == '.e'

			if not filesCorrect:
				raise Exception('Invalid File')				
			else:				
				filename = os.path.basename(file)[:-2]
				self.update()

				return file, filename
		else:
			return None, None











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
			self.btn_clear.hide()
			self.btn_operators.hide()
		else:
			if self.Gname == 'M':
				self.btn_M_property.show()
			self.G_options.show()
			self.btn_clear.show()
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
				self.toggle_invisible()
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

		hard = self.G.n >= 650		
		factor = self.drawScaling.value()			
		tkz = TikzDocument(fname, self.G, factor)
		tkz.draw(hard)


	def color_graph(self):
		maxColors = self.maxColors.value()

		if self.sat_color.isChecked():
			colored = self.G.sat_color(colors = maxColors)					
		else:
			colored = self.G.color_graph(colors = maxColors, new = True)						

		msg = QMessageBox()
		msg.setWindowIcon(QIcon(os.path.join('GUI', 'appicon.ico')))
		if colored:
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
		try:
			printed = True
			self.G.save_graph(self.Gname)

			if self.dimacs.isChecked():
				self.G.save_dimacs(self.Gname)

			if self.cnf.isChecked():
				maxColors = self.maxColors.value()
				self.G.save_cnf(self.Gname, maxColors)
		except Exception as e:
			printed = False
			print(e)
		finally:
			msg = QMessageBox()
			msg.setWindowIcon(QIcon(os.path.join('GUI', 'appicon.ico')))
			msg.setWindowTitle("Print Graph")
			msg.setStandardButtons(QMessageBox.Ok)

			if printed:
				msg.setIcon(QMessageBox.Information)
				msg.setText("Graph printed successfully.")				
			else:
				msg.setIcon(QMessageBox.Critical)
				msg.setText("Something went wrong.")
				msg.setText("The graph couldn't be printed.")

			msg.exec()

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
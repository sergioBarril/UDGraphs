import os
import math
import subprocess
import platform


class TikzDocument():
	"""
	The TikzDocument class takes care of drawing the resulting graph, using up to 6 colors.

	The output will be drawn in the tikz/ folder, and its texFile  in tikz/texFiles

	#### Parameters ####
		fname = name of output file
		UDGraph = unit distance graph to draw
		factor = edge length, thus effectively making the vertices smaller and the edges longer

	### Observations ###

		This makes use of the'pdflatex' and/or 'lualatex' commands, otherwise it won't compile the pdf.
	"""
	def __init__(self, fname, UDGraph, factor = 1):
		texFolder = os.path.join('tikz', 'texFiles')
		self.fname = os.path.join(texFolder, fname + '.tex')
		self.pdfname = os.path.join('tikz', fname + '.pdf')
		self.G = UDGraph.graph
		self.factor = factor

	def open_new(self):
		"""
		Starts writing the preamble on the .tex file
		"""
		with open(self.fname, 'w') as f:
			f.write(r"""\documentclass[border=2mm, tikz]{standalone}
\usepackage{tkz-graph}

\usetikzlibrary{shapes.geometric}
\usetikzlibrary{shapes, positioning}
\usetikzlibrary{calc}

\begin{document}

\renewcommand*{\VertexSmallMinSize}{2pt}
\renewcommand*{\VertexInnerSep}{1pt}
\renewcommand*{\EdgeLineWidth}{0.2pt}

\begin{tikzpicture}
	\GraphInit[vstyle=Welsh]	
	\SetVertexNoLabel

""")			
	
	def end(self):
		"""
		Closes the .tex file
		"""
		with open(self.fname, 'a') as f:
			f.write(r"""
\end{tikzpicture}
\end{document}""")

	def add_nodes(self):
		"""
		Adds all the nodes to draw in the .tex file
		"""
		with open(self.fname, 'a') as f:
			f.write("\n%%%%%%%%%% ADDING NODES %%%%%%%%%%%%%\n\n")
			i = 0
			for v in self.G.nodes:
				f.write('\t\\Vertex[x={}, y={}]{{{}}}\n'.format(round(self.factor*v.x, 3), round(self.factor*v.y, 3), i))
				v.id = i
				i += 1

	def add_edges(self):
		"""
		Adds all the edgegs to draw in the .tex file
		"""
		with open(self.fname, 'a') as f:
			f.write("%%%%%%%%%% ADDING EDGES %%%%%%%%%%%%%\n\n")
			for v in self.G.nodes:			
				for w in self.G.nodes:
					if (v, w) in self.G.edges:
						f.write('\t\\Edge({})({})\n'.format(v.id, w.id))

	def add_colors(self, colors = 6):
		"""
		Adds all the colors to the vertices
		"""
		col_dict = {0: 'yellow!15', 1: 'yellow!30', 2:'red!30', 3:'blue!30', 4:'green!30', 5:'orange!45', 6:'cyan!30'}

		nodes = [""]*(colors + 1)

		with open(self.fname, 'a') as f:
			f.write("%%%%%%%%%% COLORING NODES %%%%%%%%%%%%%\n\n")
			for v in self.G.nodes:
				color = v.color				
				if v.color == -1:
					color += 1				
				
				nodes[color] += str(v.id) + ', '

			for i in range(colors + 1):
				if nodes[i]:
					nodes[i] = nodes[i][:-2]
					f.write("\t\\AddVertexColor{{{}}}{{{}}}\n".format(col_dict[i], nodes[i]))

	def run(self, hard=False):
		"""
		Compiles the .texFile using pdflatex or lualatex.
		"""
		texFiles = os.path.join('tikz', 'texFiles')
		auxFiles = os.path.join(texFiles, 'auxFiles')

		pdfFile = self.pdfname

		runningOS = platform.system()

		# Linux and Windows compatibility
		if not (runningOS == 'Windows' or runningOS == 'Linux'):
			raise Exception('Not available for MacOS')

		if runningOS == 'Windows':
			auxFiles = '--aux-directory {}'.format(auxFiles)			
		else:
			auxFiles = ''
			pdfFile = 'evince {}'.format(pdfFile)
		
		if not hard:
			command = "pdflatex -interaction=nonstopmode "
		else:
			command = "lualatex "

		command += "--shell-escape --output-directory tikz {} {}".format(auxFiles, self.fname)
		
		os.system(command)
		self.p = subprocess.Popen([pdfFile],shell=True)

	def draw(self, hard=False):
		"""
		Using all of the methods above, draws the graph
		"""
		self.open_new()

		self.add_nodes()
		self.add_edges()
		self.add_colors()

		self.end()
		
		self.run(hard)

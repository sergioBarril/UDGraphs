import os

# from graphs import Vertex, UnitDistanceGraph, H, V, W, M
# from graphs import MoserSpindle, J, K, L, T, U

import subprocess


class TikzDocument():
	def __init__(self, fname, UDGraph):
		texFolder = os.path.join('tikz', 'texFiles')
		self.fname = os.path.join(texFolder, fname + '.tex')
		self.pdfname = os.path.join('tikz', fname + '.pdf')
		self.G = UDGraph.graph
		self.p = None # Process for pdf

	def open_new(self):
		with open(self.fname, 'w') as f:
			f.write(r"""\documentclass[border=2mm, tikz]{standalone}
\usepackage{tkz-graph}

\usetikzlibrary{shapes.geometric}
\usetikzlibrary{shapes, positioning}
\usetikzlibrary{calc}

\begin{document}

\renewcommand*{\VertexSmallMinSize}{2pt}
%\renewcommand*{\VertexInnerSep}{0pt}
\renewcommand*{\EdgeLineWidth}{0.2pt}

\begin{tikzpicture}
	\GraphInit[vstyle=Welsh]	
	\SetVertexNoLabel

""")			
	
	def end(self):
		with open(self.fname, 'a') as f:
			f.write(r"""
\end{tikzpicture}
\end{document}""")

	def add_nodes(self):
		with open(self.fname, 'a') as f:
			f.write("\n%%%%%%%%%% ADDING NODES %%%%%%%%%%%%%\n\n")
			i = 0
			for v in self.G.nodes:
				f.write('\t\\Vertex[x={}, y={}]{{{}}}\n'.format(round(v.x, 3), round(v.y, 3), i))
				v.id = i
				i += 1

	def add_edges(self):
		with open(self.fname, 'a') as f:
			f.write("%%%%%%%%%% ADDING EDGES %%%%%%%%%%%%%\n\n")
			for v in self.G.nodes:
				for w in self.G.nodes:
					if (v, w) in self.G.edges:
						f.write('\t\\Edge({})({})\n'.format(v.id, w.id))

	def add_colors(self):

		col_dict = {0: 'yellow!15', 1: 'yellow!30', 2:'red!30', 3:'blue!30', 4:'green!30'}

		nodes = [""]*5

		with open(self.fname, 'a') as f:
			f.write("%%%%%%%%%% COLORING NODES %%%%%%%%%%%%%\n\n")
			for v in self.G.nodes:
				if v.color == -1:
					continue
				nodes[v.color] += str(v.id) + ', '

			for i in range(5):
				if nodes[i]:
					nodes[i] = nodes[i][:-2]
					f.write("\t\\AddVertexColor{{{}}}{{{}}}\n".format(col_dict[i], nodes[i]))

				# if v.color == -1: # NO COLOR
				# 	continue
				# if v.color == 0: # LIGHT, BASE COLOR
				# 	strength = "!15"
				# else:
				# 	strength = "!30"

				# color = col_dict[v.color] + strength				
				# f.write("\t\\AddVertexColor{{{}}}{{{}}}\n".format(color, v.id ))

	def run(self, hard=False):
		texFiles = os.path.join('tikz', 'texFiles')
		auxFiles = os.path.join(texFiles, 'auxFiles')

		if not hard:
			command = "pdflatex -halt-on-error "
		else:
			command = "lualatex "

		command += '--shell-escape -output-directory tikz -aux-directory {} '.format(auxFiles) + self.fname

		print(command)
		os.system(command)
		self.p = subprocess.Popen([self.pdfname],shell=True)

	def draw(self, hard=False):
		self.open_new()

		self.add_nodes()
		self.add_edges()
		self.add_colors()

		self.end()
		
		self.run(hard)

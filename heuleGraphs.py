import sys

from graphs import UnitDistanceGraph
from vertex import Vertex
import math


import re
import os

def rewriteSqrt(vertex):

	x, y = vertex[0], vertex[1]

	x = re.sub(r'Sqrt\[(.*?)\]', r'math.sqrt(\1)', x)	
	y = re.sub(r'Sqrt\[(.*?)\]', r'math.sqrt(\1)', y)

	return [x,y]

def readGraph(fname):
	"""
	Reads a file with vertices in the format used by Marijn Heule,
	and changes it to the format of the other files.
	"""

	idtov = dict()
	vtoid = dict()

	i = 1

	G = UnitDistanceGraph()

	# with open(os.path.join('heule', 'edited' + fname), w) as fw:
	with open(os.path.join('heule', fname + '.vtx'), 'r') as fr:				
		for line in fr:
			nl = line[1:-2]
			print(nl)
			nl = nl.split(",")
			print(nl)
			nl = rewriteSqrt(nl)
			print(nl)

			v = Vertex(eval(nl[0]), eval(nl[1]))

			G.add_node(v)

			idtov[i] = v
			vtoid[v] = i

			i += 1

	with open(os.path.join('heule', fname + '.edge'), 'r') as fr:
		for line in fr:
			if line[0] == 'p':
				pass
			else:			
				v, w = line.split()[1:]

				G.add_edge(idtov[int(v)], idtov[int(w)])


	G.save_graph(fname)

	with open(os.path.join('heule', fname + '.dict'), 'w') as fw:		
		for v, v_id in vtoid.items():
			fw.write("{} {} {}\n".format(v_id, v.x, v.y))

readGraph(sys.argv[1])
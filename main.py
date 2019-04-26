import matplotlib.pyplot as plt
import networkx as nx
import math
import time

from graphs import Vertex, UnitDistanceGraph, H, V, W, M

"""
myM = UnitDistanceGraph()
myM.load_graph('M')

maxR = 0
for v in myM.graph.nodes:
	maxR = max(v.getR(), maxR)

print("n = {}\tm = {}".format(myM.n, myM.m))
print("maxR = {}".format(maxR))
"""

myW = W()
myW.save_graph('W')
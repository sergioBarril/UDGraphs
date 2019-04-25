import matplotlib.pyplot as plt
import networkx as nx
import math
import time

from graphs import Vertex, UnitDistanceGraph, H, V, W
from graphs import minkowskiSum, trimMinkowski

def testMinkowski():
	A = UnitDistanceGraph()
	A.add_edge(Vertex(0,0), Vertex(1,0))

	B = UnitDistanceGraph()
	B.add_edge(Vertex(0,0), Vertex(0.5, math.sqrt(3)/2))

	M = minkowskiSum(A,B)
	M.print_graph()


def testH():
	myH = H()
	print(myH.n)
	print(myH.m)

W0 = W()
print("W: n = {}\t m = {}".format(W0.n, W0.m))

H0 = H()
print("H: n = {}\t m = {}".format(H0.n, H0.m))

M = minkowskiSum(H0, W0)
print("M: n = {}\t m = {}".format(M.n, M.m))


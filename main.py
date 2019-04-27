import matplotlib.pyplot as plt
import networkx as nx
import math
import time

from graphs import Vertex, UnitDistanceGraph, H, V, W, M


def testH():
	G = H()
	print('n = {}\tm = {}'.format(G.n, G.m))
	# G.save_graph('H')

def testV():
	G = V()
	print('n = {}\tm = {}'.format(G.n, G.m))
	# G.save_graph('V')

def testW():
	G = W()
	print('n = {}\tm = {}'.format(G.n, G.m))
	# G.save_graph('W')

def testM():
	G = M()
	G.update()
	print('n = {}\tm = {}'.format(G.n, G.m))
	# G.save_graph('M')

testM()
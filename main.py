import networkx as nx
import math
import time

from graphs import Vertex, UnitDistanceGraph, H, V, W, M
from graphs import MoserSpindle, J, K, L


def saveH():
	G = H()
	print('n = {}\tm = {}'.format(G.n, G.m))
	G.save_graph('H')

def saveJ():
	G = J()
	G.update()
	print('n = {}\tm = {}'.format(G.n, G.m))
	G.save_graph('J')

def saveK():
	G = K()
	G.update()
	G.save_graph('K')

def saveL():
	G = L()
	G.update()
	G.save_graph('L')

def saveV():
	G = V()
	print('n = {}\tm = {}'.format(G.n, G.m))
	G.save_graph('V')

def saveW():
	G = W()
	G.update()
	print('n = {}\tm = {}'.format(G.n, G.m))
	G.save_graph('W')

def saveM():
	G = M()
	G.update()
	G.save_graph('M')

def saveSpindle():
	G = MoserSpindle()
	G.update()
	G.save_graph('moser_spindle')


saveL()
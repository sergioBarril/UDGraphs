import matplotlib.pyplot as plt
import networkx as nx
import math
import time

from graphs import Vertex, UnitDistanceGraph, H, V, W, M
from graphs import minkowskiSum, trimMinkowski

G = UnitDistanceGraph()
G.graph = minkowskiSum(H(), H())
G.update()
G.print_graph()
import math
from math import isclose, sqrt, sin, cos, acos, asin, fabs, pi

class Vertex:
	"""Vertex of the graph. Contains coordinates and color"""
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.r = self.getR()

		self.color = -1

	def __str__(self):
		""" (x, y)[color] """

		vertex = '({}, {})'.format(round(self.x, 3), round(self.y, 3), self.color)
		color = '[{}]'.format(self.color)

		if self.color != -1:
			return vertex + color
		else:
			return vertex

	def __hash__(self):
		"""
		Hashes with the first three decimal places, rounded
		"""
		round_x = round(self.x, 3)
		round_y = round(self.y, 3)

		hash1 = hash(round_x)
		hash2 = hash(round_y)

		return hash((hash1, hash2))

	def __add__(self, v):
		""" Sum of 2 vertices """
		return Vertex(self.x + v.x, self.y + v.y)

	def __sub__(self, v):
		""" Subtraction of 2 vertices """
		return Vertex(self.x - v.x, self.y - v.y)

	def __truediv__(self, num):
		""" Division of the Vertex by a number """
		if isinstance(num, Vertex):
			return NotImplemented
		return Vertex(self.x/num, self.y/num)


	def __eq__(self, other):
		if not isinstance(other, Vertex):
			return NotImplemented
		return round(self.x, 3) == round(other.x, 3) and round(self.y, 3) == round(other.y, 3)

	def dist(self, v):
		"""
		Returns its Euclidean distance to another vertex
		"""
		x = self.x - v.x
		x = x * x

		y = self.y - v.y
		y = y * y

		return sqrt(x + y)

	def isUnitDist(self, v):
		"""
		Given two vertices, check whether they're at unit distance from each other.
		"""
		return isclose(1, self.dist(v), rel_tol= 1.e-9, abs_tol = 0)

	def isColored(self):
		return self.color > 0
	
	def rotate(self, i, k = None, center = None):
		"""
		Returns a vertex rotated with respect to the given center, or
		(0,0) as a default center.
		
		If k is given:
			i changes the angle, where angle = arccos(2i-1 / 2i)
			k changes how many times is the rotation applied
		Else:
			i is the angle, in radians
		"""
		if center is None:
			center = Vertex(0,0)

		if k is None:
			alpha = i
		else:
			alpha = (2 * i - 1)/(2 * i)
			alpha = acos(alpha)
			alpha *= k

		c = center
		x = (self.x - c.x) * cos(alpha) - (self.y - c.y) * sin(alpha) + c.x
		y = (self.x - c.x) * sin(alpha) + (self.y - c.y) * cos(alpha) + c.y

		return Vertex(x,y)

	def getR(self):
		"""
		Returns the distance to (0,0)
		"""
		x2 = self.x * self.x
		y2 = self.y * self.y

		return sqrt(x2 + y2)
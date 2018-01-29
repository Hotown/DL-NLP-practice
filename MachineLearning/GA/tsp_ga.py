import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from MachineLearning.GA.ga import GA
from pylab import *



class TSP(object):
	def __init__(self, life_count=100):
		self.init_cities()
		self.life_count = life_count
		self.ga = GA(cross_rate=0.7,
								 mutation_rate=0.02,
								 life_count=self.life_count,
								 gene_length=len(self.cities),
								 match_func=self.match_func())

	def init_cities(self):
		self.cities = []
		data = pd.read_csv('./data/city.csv', dtype=object)
		self.cities = [(float(data['1'][i]), float(data['2'][i])) for i in range(len(data))]
		# print(self.cities)

	def distance(self, order):
		distance = 0.0
		for i in range(-1, len(self.cities) - 1):
			index1, index2 = order[i], order[i + 1]
			city1, city2 = self.cities[index1], self.cities[index2]
			distance += math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
		# print(distance)
		return distance

	def match_func(self):
		return lambda life: 1.0 / self.distance(life.gene)

	def run(self, n=0):
		while n > 0:
			self.ga.next()
			distance = self.distance(self.ga.best.gene)
			print("%d : %f" % (self.ga.generation, distance))
			n -= 1

	def show(self, best):
		cities = []
		for i in range(len(best) - 1):
			cities.append(self.cities[best[i]])

		x = [cities[i][0] for i in range(len(cities))]
		x.append(cities[0][0])
		y = [cities[i][1] for i in range(len(cities))]
		y.append(cities[0][1])
		plt.plot(x, y, '.-')
		plt.show()


if __name__ == '__main__':
	tsp = TSP()
	tsp.run(10000)
	tsp.show(tsp.ga.best.gene)
	# print(tsp.ga.best.gene)
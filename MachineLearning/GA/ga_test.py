from MachineLearning.GA import ga
from MachineLearning.GA import ga_object
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

if __name__ == '__main__':
	# ga = ga.GA(0.5, 0.5, 10, 9)
	# parent1 = ga_object.GAObject(gene=[8, 9, 6, 7, 5, 3, 4, 2, 1])
	# parent2 = ga_object.GAObject(gene=[9, 8, 7, 6, 5, 4, 3, 2, 1])
	# print(ga.cross(parent1,parent2))

	data = pd.read_csv('./data/city.csv', dtype=object)
	print(data['1'][1], data['2'][1])

	cities = [(float(data['1'][i]), float(data['2'][i])) for i in range(len(data))]
	# print(cities)
	# print(len(cities))
	# x = np.arange(-5.0, 5.0, 0.02)
	# y1 = np.sin(x)
	#
	# plt.figure(1)
	# plt.subplot(211)
	# plt.plot(x, y1)
	#
	# plt.subplot(212)
	# # 设置x轴范围
	# xlim(-2.5, 2.5)
	# # 设置y轴范围
	# ylim(-1, 1)
	# plt.plot(x, y1)
	# plt.show()
	x = [cities[i][0] for i in range(len(cities))]
	y = [cities[i][1] for i in range(len(cities))]
	plt.plot(x, y, '.-')
	plt.show()
	# print(x)


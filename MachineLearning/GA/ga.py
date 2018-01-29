import random
from MachineLearning.GA.ga_object import GAObject


class GA(object):
	def __init__(self,
							 cross_rate,
							 mutation_rate,
							 life_count,
							 gene_length,
							 match_func=lambda life: 1):
		"""
		遗传算法类
		:param cross_rate: 交叉率
		:param mutation_rate: 变异率
		:param life_count: 种群个体数
		:param gene_length: 基因型长度
		:param match_func: 适应函数
		"""
		self.cross_rate = cross_rate
		self.mutation_rate = mutation_rate
		self.life_count = life_count
		self.gene_length = gene_length
		self.match_func = match_func
		self.lives = []  # 种群
		self.best = None  # 最优个体
		self.generation = 1
		self.cross_count = 0
		self.mutation_count = 0
		self.bounds = 0.0  # 适配值之和

		self.init_live()

	def init_live(self):
		"""初始化种群"""
		self.lives = []
		for i in range(self.life_count):
			gene = [x for x in range(self.gene_length)]
			random.shuffle(gene)
			life = GAObject(gene)
			self.lives.append(life)

	def judge(self):
		"""评估个体适配值"""
		self.bounds = 0.0
		self.best = self.lives[0]
		for life in self.lives:
			life.score = self.match_func(life)
			self.bounds += life.score
			if self.best.score < life.score:
				self.best = life

	def cross(self, parent1, parent2):
		"""交叉"""

		# 选择一个基因片段
		index_front = random.randint(0, self.gene_length - 1)
		index_tail = random.randint(index_front, self.gene_length - 1)
		gene_fragment = parent2.gene[index_front:index_tail]

		# TSP问题中，每个城市只经过一次，所以在交叉过程中应该注意不让碱基重复
		# ep. parent1 = [8,9,6,7,5,3,4,2,1] parent2 = [9,8,7,6,5,4,3,2,1]
		# gene_fragment = [6,5,4] => new_gene = [8,9,7,6,5,4,3,2,1]
		new_genes = []
		p = 0
		for x in parent1.gene:
			if p == index_front:
				new_genes.extend(gene_fragment)
				p += 1
			if x not in gene_fragment:
				new_genes.append(x)
				p += 1
		self.cross_count += 1
		return new_genes

	def mutation(self, gene):
		"""基因突变，选取染色体上随机的两个碱基对交叉互换"""
		index1 = random.randint(0, self.gene_length - 1)
		index2 = random.randint(0, self.gene_length - 1)
		while index1 ^ index2 == 0:
			index2 = random.randint(0, self.gene_length - 1)

		new_gene = gene[:]
		new_gene[index1], new_gene[index2] = new_gene[index2], new_gene[index1]
		self.mutation_count += 1
		return new_gene

	def get_one(self):
		"""选择一个个体"""
		r = random.uniform(0, self.bounds)
		for life in self.lives:
			r -= life.score
			if r <= 0:
				return life

		raise Exception("选择失败", self.bounds)

	def create_child(self):
		"""产生新的后代"""
		parent1 = self.get_one()

		# 交叉
		rate = random.random()
		if rate < self.cross_rate:
			parent2 = self.get_one()
			gene = self.cross(parent1, parent2)
		else:
			gene = parent1.gene

		# 突变
		rate = random.random()
		if rate < self.mutation_rate:
			gene = self.mutation(gene)

		return GAObject(gene)

	def next(self):
		self.judge()
		new_lives = [self.best]
		while len(new_lives) < self.life_count:
			new_lives.append(self.create_child())
		self.lives = new_lives
		self.generation += 1

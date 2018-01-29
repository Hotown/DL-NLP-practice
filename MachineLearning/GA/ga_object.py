SCORE_NONE = -1


class GAObject(object):
	def __init__(self, gene=None):
		self.gene = gene
		self.score = SCORE_NONE

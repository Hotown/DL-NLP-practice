import random as rd


def segment(text, segs):
	words = []
	p = 0
	for i in range(len(segs)):
		if segs[i] == '1':
			words.append(text[p:i+1])
			p = i+1
	words.append(text[p:])
	return words


def evaluate(text, segs):
	words = segment(text, segs)
	text_len = len(words)
	dic_len = len(' '.join(list(set(words))))
	return text_len + dic_len


def flip(segs, pos):
	return segs[:pos] + str(1 - int(segs[pos])) + segs[pos + 1:]


def flip_n(segs, n):
	for i in range(n):
		segs = flip(segs, rd.randint(0, len(segs) - 1))
	return segs


def anneal(text, segs, iterations, cooling_rate):
	init_t = float(len(segs))
	while init_t > 0.5:
		best_seg, best_score = segs, evaluate(text, segs)
		for i in range(iterations):
			guess = flip_n(segs, int(round(init_t)))
			score = evaluate(text, guess)
			if score < best_score:
				best_seg, best_score = guess, score
		score, segs = best_score, best_seg
		init_t = init_t / cooling_rate
		print(evaluate(text, segs), segment(text, segs))
	return segs


if __name__ == '__main__':
	# text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
	# segs1 = "0000000000000001000000000010000000000000000100000000000"
	text = "iloveyouilovedrinkingyoulovedrinking"
	segs1 = "000000001000000000000100000000000000"
	print(anneal(text, segs1, 10000, 1.2))

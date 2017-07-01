# coding: utf-8
import math
import numpy as np

seq_length = 8
max_word_length = 3

def convert_semi_markov_to_markov(semi_markov_prob):
	pass

def convert_markov_to_semi_markov(markov_weights):
	semi_markov_prob = np.zeros((seq_length + 1, max_word_length + 1, max_word_length + 1))
	for t in range(1, seq_length):
		for k in range(1, max_word_length + 1):
			for j in range(1, max_word_length + 1):
				semi_markov_prob[t - 1][k][j] = math.exp(gamma(t - k + 1, t + 1, markov_weights))
	return semi_markov_prob

def gamma(s, t, weights):
	if s == t - 1:
		return weights[t][0][0]
	total = weights[s + 1][1][0]
	for i in range(s + 2, t):
		total += weights[i][1][1]
	total += weights[t][0][1]
	return total

def generate_forward_table_from_markov_weights(markov_weights):
	alpha = np.zeros((seq_length + 1, max_word_length + 1))
	for t in range(1, seq_length):	# eos以外
		for k in range(1, max_word_length + 1):
			sum_alpha = 0
			if t - k == 0:	# bosに接続してる場合
				sum_alpha += math.exp(gamma(t - k + 1, t + 1, markov_weights) + gamma(0, 1, markov_weights))
			for j in range(1, min(t - k + 1, max_word_length + 1)):
				sum_alpha += math.exp(gamma(t - k + 1, t + 1, markov_weights)) * alpha[t - k][j]
			alpha[t][k] = sum_alpha
	return alpha

def generate_forward_table_from_semi_markov_prob(semi_markov_prob):
	alpha = np.zeros((seq_length + 1, max_word_length + 1))
	for t in range(1, seq_length):	# eos以外
		for k in range(1, max_word_length + 1):
			sum_alpha = 0
			if t - k == 0:	# bosに接続してる場合
				sum_alpha += semi_markov_prob[t][k][1]
			for j in range(1, min(t - k + 1, max_word_length + 1)):
				sum_alpha += semi_markov_prob[t][k][j] * alpha[t - k][j]
			alpha[t][k] = sum_alpha
	return alpha

def test_gamma():
	markov_weights = np.ones((seq_length + 1, 2, 2), dtype=float)
	markov_weights[:, 0, 1] = 0.5
	markov_weights[:, 1, 0] = 0.5
	for s in range(1, seq_length - 1):
		for t in range(s + 1, seq_length):
			g = gamma(s, t, markov_weights)
			if s + 1 == t:
				assert g == 1
			else:
				assert t - s - 1 == g

def test_markov_to_semi_markov_conversion():
	markov_weights = np.random.normal(0, 1, size=(seq_length + 1, 2, 2))
	alpha1 = generate_forward_table_from_markov_weights(markov_weights)
	semi_markov_prob = convert_markov_to_semi_markov(markov_weights)
	alpha2 = generate_forward_table_from_semi_markov_prob(semi_markov_prob)
	print(alpha1)
	print(alpha2)

def test_semi_markov_to_markov_conversion():
	semi_markov_prob = np.random.normal(0, 1, size=(seq_length + 1, max_word_length + 1, max_word_length + 1))
	semi_markov_prob = np.exp(semi_markov_prob) / np.sum(np.exp(semi_markov_prob))

if __name__ == "__main__":
	np.random.seed(0)
	test_gamma()
	test_markov_to_semi_markov_conversion()
	test_semi_markov_to_markov_conversion()

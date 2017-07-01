# coding: utf-8
from __future__ import division
from __future__ import print_function
import math
import numpy as np

seq_length = 8
max_word_length = 3

neg_large_value = -1e-3
np.set_printoptions(suppress=True)

# オリジナル版のsemi-Markov->CRFの変換
def convert_semi_markov_to_markov(semi_markov_prob):
	markov_weights = np.full((seq_length + 1, 2, 2), neg_large_value, dtype=float)
	markov_weights[1][1][1] = semi_markov_prob[1][1][1]
	for t in range(2, seq_length - 1):
		# 1->1
		total = 0
		k = 1
		for _t in range(t, min(seq_length - 1, t + max_word_length)):
			total += semi_markov_prob[_t][k][1]
			k += 1
		markov_weights[t][1][1] = total
		# 0->1
		if t > 2:
			total = 0
			for j in range(2, min(t, max_word_length + 1)):
				k = 1
				for _t in range(t, min(seq_length - 1, t + max_word_length)):
					total += semi_markov_prob[_t][k][j]
					k += 1
			markov_weights[t][1][0] = total
		# 1->0
		if t == 2:
			markov_weights[t][0][1] = semi_markov_prob[t][2][1]
		else:
			total = 0
			for j in range(1, min(t - 1, max_word_length + 1)):
				k = 2
				for _t in range(t, min(seq_length - 1, t + max_word_length - 1)):
					total += semi_markov_prob[_t][k][j]
					k += 1
			markov_weights[t][0][1] = total
		# 0->0
		if t > 2:
			total = 0
			for j in range(1, min(t - 1, max_word_length + 1)):
				for k in range(3, max_word_length + 1):
					_k = k
					for _t in range(t, min(seq_length - 1, t + max_word_length - 1)):
						total += semi_markov_prob[_t][_k][j]
						_k += 1
						if _k > max_word_length:
							break
			markov_weights[t][0][0] = total
	markov_weights[seq_length - 1][1][1] = semi_markov_prob[seq_length - 1][1][1]
	markov_weights[seq_length - 1][1][0] = np.sum(semi_markov_prob[seq_length - 1][1][:])
	return markov_weights

# 修正版のCRF->semi-Markovの変換
def convert_markov_to_semi_markov(markov_weights):
	semi_markov_prob = np.zeros((seq_length + 1, max_word_length + 1, max_word_length + 1))
	for t in range(1, seq_length - 1):
		for k in range(1, min(t + 1, max_word_length + 1)):
			if t - k == 0:
				semi_markov_prob[t][k][1] = math.exp(gamma(t - k + 1, t + 1, markov_weights))
			for j in range(1, min(t - k + 1, max_word_length + 1)):
				semi_markov_prob[t][k][j] = math.exp(gamma(t - k + 1, t + 1, markov_weights))
	return semi_markov_prob

# γ(s, t)
def gamma(s, t, weights):
	if s == t - 1:
		return weights[t][1][1]
	total = weights[s + 1][0][1]
	for i in range(s + 2, t):
		total += weights[i][0][0]
	total += weights[t][1][0]
	return total

# CRFの重みからNPYLMの前向き確率を計算
def generate_forward_table_from_markov_weights(markov_weights):
	alpha = np.zeros((seq_length + 1, max_word_length + 1))
	for t in range(1, seq_length - 1):	# eos以外
		for k in range(1, max_word_length + 1):
			sum_alpha = 0
			if t - k == 0:	# bosに接続してる場合
				sum_alpha += math.exp(gamma(t - k + 1, t + 1, markov_weights) + gamma(0, 1, markov_weights))
			for j in range(1, min(t - k + 1, max_word_length + 1)):
				sum_alpha += math.exp(gamma(t - k + 1, t + 1, markov_weights)) * alpha[t - k][j]
			alpha[t][k] = sum_alpha
	return alpha

# semi-Markovモデルのパスの確率からNPYLMの前向き確率を計算
# NPYLMと同じ
def generate_forward_table_from_semi_markov_prob(semi_markov_prob):
	alpha = np.zeros((seq_length + 1, max_word_length + 1))
	for t in range(1, seq_length - 1):	# eos以外
		for k in range(1, max_word_length + 1):
			sum_alpha = 0
			if t - k == 0:	# bosに接続してる場合
				sum_alpha += semi_markov_prob[t][k][1]
			for j in range(1, min(t - k + 1, max_word_length + 1)):
				sum_alpha += semi_markov_prob[t][k][j] * alpha[t - k][j]
			alpha[t][k] = sum_alpha
	return alpha

# CRFの規格化定数の計算
def compute_z_markov(markov_weights, edge, apply_exp=True):
	z = np.zeros((seq_length + 1, 2), dtype=float)
	z[1][1] = math.exp(markov_weights[1][1][1])
	z[2][1] = math.exp(markov_weights[2][1][1]) * z[1][1]
	z[2][0] = math.exp(markov_weights[2][0][1]) * z[1][1]
	def exp(x):
		if apply_exp:
			return math.exp(x)
		return x
	print("compute z")
	if apply_exp:
		print(np.exp(markov_weights))
	else:
		print(markov_weights)
	for t in range(3, edge):
		z[t][1] = z[t - 1][0] * exp(markov_weights[t][1][0]) + z[t - 1][1] * exp(markov_weights[t][1][1])
		z[t][0] = z[t - 1][0] * exp(markov_weights[t][0][0]) + z[t - 1][1] * exp(markov_weights[t][0][1])
	z[edge][1] = z[edge - 1][0] * exp(markov_weights[edge][1][0]) + z[edge - 1][1] * exp(markov_weights[edge][1][1])
	return z[edge][1]

def test_gamma():
	markov_weights = np.ones((seq_length + 1, 2, 2), dtype=float)
	markov_weights[:, 0, 1] = 0.5
	markov_weights[:, 1, 0] = 0.5
	for s in range(0, seq_length - 1):
		for t in range(s + 1, seq_length):
			g = gamma(s, t, markov_weights)
			if s + 1 == t:
				assert g == 1
			else:
				assert t - s - 1 == g

def test_markov_to_semi_markov_conversion():
	markov_weights = np.random.normal(-1, 1, size=(seq_length, 2, 2))
	markov_weights = np.ones((seq_length + 1, 2, 2), dtype=float)
	markov_weights[:, 0, 1] = 2
	markov_weights[:, 1, 0] = 2
	markov_weights[0, :] = neg_large_value
	markov_weights[1, 0, :] = neg_large_value
	markov_weights[1, 1, 0] = neg_large_value
	markov_weights[2, 1, 0] = neg_large_value
	markov_weights[2, 0, 0] = neg_large_value
	markov_weights[-1, 0, :] = neg_large_value
	alpha1 = generate_forward_table_from_markov_weights(markov_weights)
	semi_markov_prob = convert_markov_to_semi_markov(markov_weights)
	alpha2 = generate_forward_table_from_semi_markov_prob(semi_markov_prob)
	_markov_weights = convert_semi_markov_to_markov(semi_markov_prob)
	print("markov")
	print(markov_weights)
	print("semi-markov")
	print(semi_markov_prob)
	print("markov")
	print(_markov_weights)
	raise Exception()
	z_markov = compute_z_markov(markov_weights, seq_length - 2)
	_z_markov = compute_z_markov(_markov_weights, seq_length - 2, False)
	print("z")
	print(z_markov, _z_markov)
	print(alpha1 / (np.sum(alpha1, axis=1, keepdims=True) + 1e-16))
	print(alpha2 / (np.sum(alpha2, axis=1, keepdims=True) + 1e-16))
	print(np.exp(markov_weights) / z_markov)
	print(np.exp(_markov_weights) / _z_markov)

def test_semi_markov_to_markov_conversion():
	semi_markov_prob = np.random.normal(0, 1, size=(seq_length + 1, max_word_length + 1, max_word_length + 1))
	semi_markov_prob = np.exp(semi_markov_prob) / np.sum(np.exp(semi_markov_prob))
	semi_markov_prob[0] = 0
	semi_markov_prob[:, 0] = 0
	semi_markov_prob[:, :, 0] = 0
	markov_weights = convert_semi_markov_to_markov(semi_markov_prob)

if __name__ == "__main__":
	np.random.seed(0)
	test_gamma()
	test_markov_to_semi_markov_conversion()
	test_semi_markov_to_markov_conversion()

import argparse, sys, os, time, codecs, random
from tabulate import tabulate
import MeCab
import npycrf as nlp
from train import printb

def viterbi(npycrf, dictionary, filepath, directory, neologd_path=None):
	assert filepath is not None or directory is not None
	sentence_list = []

	if filepath is not None:
		with codecs.open(filepath, "r", "utf-8") as f:
			for sentence_str in f:
				sentence_str = sentence_str.strip()
				sentence_list.append(sentence_str)

	if directory is not None:
		for filename in os.listdir(directory):
			with codecs.open(os.path.join(directory, filename), "r", "utf-8") as f:
				for sentence_str in f:
					sentence_str = sentence_str.strip()
					sentence_list.append(sentence_str)

	# 教師データはMeCabによる分割
	tagger = MeCab.Tagger() if neologd_path is None else MeCab.Tagger("-d " + neologd_path)
	tagger.parse("")	# バグ回避のため空データを分割
	for sentence_str in sentence_list:
		sentence_str = sentence_str.strip()
		m = tagger.parseToNode(sentence_str)	# 形態素解析
		words_true = []
		while m:
			word = m.surface
			if len(word) > 0:
				words_true.append(word)
			m = m.next
		if len(words_true) > 0:
			words_viterbi = npycrf.parse(sentence_str, dictionary)
			print("MeCab+NEologd")
			printb(" / ".join(words_true))
			print("NPYCRF Viterbi")
			printb(" / ".join(words_viterbi))
			print()

def main():
	assert args.working_directory is not None
	try:
		os.mkdir(args.working_directory)
	except:
		pass

	# 辞書
	dictionary = nlp.dictionary(os.path.join(args.working_directory, "char.dict"))

	# モデル
	crf = nlp.crf(os.path.join(args.working_directory, "crf.model"))
	npylm = nlp.npylm(os.path.join(args.working_directory, "npylm.model"))
	npycrf = nlp.npycrf(npylm=npylm, crf=crf)

	num_features = crf.get_num_features()
	num_character_ids = dictionary.get_num_characters()
	print(tabulate([["#characters", num_character_ids], ["#features", num_features]]))

	# ビタビアルゴリズムによる最尤分解を求める
	viterbi(npycrf, dictionary, args.test_filename, args.test_directory, args.neologd_path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# 以下のどちらかを必ず指定
	parser.add_argument("--test-filename", "-file", type=str, default=None)
	parser.add_argument("--test-directory", "-dir", type=str, default=None)

	parser.add_argument("--working-directory", "-cwd", type=str, default="out", help="ワーキングディレクトリ")
	parser.add_argument("--neologd-path", "-neologd", type=str, default=None)
	args = parser.parse_args()
	main()
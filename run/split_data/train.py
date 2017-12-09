import argparse, sys, os, time, codecs, random
from tabulate import tabulate
import MeCab
import npycrf as nlp

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

def printb(string):
	print(stdout.BOLD + string + stdout.END)

def printr(string):
	sys.stdout.write("\r" + stdout.CLEAR)
	sys.stdout.write(string)
	sys.stdout.flush()

def build_corpus(filepath, directory, semi_supervised_split_ratio):
	assert filepath is not None or directory is not None
	corpus_l = nlp.corpus()	# 教師あり
	corpus_u = nlp.corpus()	# 教師なし
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

	random.shuffle(sentence_list)

	semi_supervised_split = int(len(sentence_list) * semi_supervised_split_ratio)
	sentence_list_l = sentence_list[:semi_supervised_split]
	sentence_list_u = sentence_list[semi_supervised_split:]

	# 教師データはMeCabによる分割
	tagger = MeCab.Tagger() if args.neologd_path is None else MeCab.Tagger("-d " + args.neologd_path)
	tagger.parse("")	# バグ回避のため空データを分割
	for sentence_str in sentence_list_l:
		m = tagger.parseToNode(sentence_str)	# 形態素解析
		words = []
		while m:
			word = m.surface
			if len(word) > 0:
				words.append(word)
			m = m.next
		if len(words) > 0:
			corpus_l.add_words(words)

	for sentence_str in sentence_list_u:
		corpus_u.add_words([sentence_str])	# 教師なしデータは文を単語とみなす

	return corpus_l, corpus_u

def main():
	assert args.working_directory is not None
	try:
		os.mkdir(args.working_directory)
	except:
		pass

	# 学習に用いるテキストデータを準備
	corpus_l, corpus_u = build_corpus(args.train_filename, args.train_directory, args.semi_supervised_split)

	# 辞書
	dictionary = nlp.dictionary()

	# 訓練データ・検証データに分けてデータセットを作成
	# 同時に辞書が更新される
	dataset_l = nlp.dataset(corpus_l, dictionary, args.train_dev_split, args.seed)	# 教師あり
	dataset_u = nlp.dataset(corpus_u, dictionary, args.train_dev_split, args.seed)	# 教師なし

	# 辞書を保存
	dictionary.save(os.path.join(args.working_directory, "npycrf.dict"))

	# 確認
	table = [
		["Labeled", dataset_l.get_size_train(), dataset_l.get_size_dev()],
		["Unlabeled", dataset_u.get_size_train(), dataset_u.get_size_dev()],
		["Total", dataset_u.get_size_train() + dataset_l.get_size_train(), dataset_u.get_size_dev() + dataset_l.get_size_dev()],
	]
	print(tabulate(table, headers=["Data", "Train", "Dev"]))

	num_character_ids = dictionary.get_num_characters()

	# モデル
	crf = nlp.crf(num_character_ids=num_character_ids,
				feature_x_unigram_start=args.crf_feature_x_unigram_start,
				feature_x_unigram_end=args.crf_feature_x_unigram_end,
				feature_x_bigram_start=args.crf_feature_x_bigram_start,
				feature_x_bigram_end=args.crf_feature_x_bigram_end,
				feature_x_identical_1_start=args.crf_feature_x_identical_1_start,
				feature_x_identical_1_end=args.crf_feature_x_identical_1_end,
				feature_x_identical_2_start=args.crf_feature_x_identical_2_start,
				feature_x_identical_2_end=args.crf_feature_x_identical_2_end,
				sigma=args.crf_prior_sigma)

	npylm = nlp.npylm(max_word_length=args.max_word_length,
					g0=1.0 / num_character_ids,
					initial_lambda_a=args.lambda_a,
					initial_lambda_b=args.lambda_b,
					vpylm_beta_stop=args.vpylm_beta_stop,
					vpylm_beta_pass=args.vpylm_beta_pass)

	npycrf = nlp.npycrf(npylm=npylm, crf=crf)

	num_features = crf.get_num_features()
	print(tabulate([["#characters", num_character_ids], ["#features", num_features]]))

	# 学習の準備
	trainer = nlp.trainer(dataset_labeled=dataset_l, 
						dataset_unlabeled=dataset_u,
						dictionary=dictionary,
						npycrf=npycrf,
						crf_regularization_constant=1.0)

	# 文字列の単語IDが衝突しているかどうかをチェック
	# 時間の無駄なので一度したらしなくてよい
	# メモリを大量に消費します
	if False:
		print("ハッシュの衝突を確認中 ...")
		num_checked_words = trainer.detect_hash_collision(args.max_word_length)
		print("衝突はありません (総単語数 {})".format(num_checked_words))

	learning_rate = args.crf_learning_rate
	batchsize = 32

	# 初期化
	trainer.add_labeled_data_to_npylm()					# 教師データをNPYLMに追加
	trainer.sgd(learning_rate, batchsize, pure_crf=True)	# NPYLMを除いてCRF単体を最適化

	for epoch in range(1, args.epochs + 1):
		start = time.time()

		# 学習
		trainer.gibbs(include_labeled_data=True)	# NPYLMのパラメータをギブスサンプリング
													# 教師データも含めた方がいい気がする
		trainer.sgd(learning_rate, batchsize)		# CRFを最適化

		# 各種サンプリング
		trainer.sample_hpylm_vpylm_hyperparameters()	# HPYLMとVPYLMのハイパーパラメータの更新
		trainer.sample_npylm_lambda()					# 単語長のポアソン分布のパラメータλの更新

		if epoch > 3:							# この推定は数イテレーション後にやるほうが精度が良い
			trainer.update_p_k_given_vpylm()	# VPYLMから長さkの単語が生成される確率P(k|VPYLM)を推定
			
		# ログ
		elapsed_time = time.time() - start
		print("Iteration {} / {} - {:.3f} sec".format(epoch, args.epochs, elapsed_time))
		if epoch % 10 == 0:
			trainer.print_segmentation_train(10)	# ランダムに分割を表示

			log_likelihood_l = trainer.compute_log_likelihood_labeled_dev()
			log_likelihood_u = trainer.compute_log_likelihood_unlabeled_dev()
			table = [
				["Labeled", log_likelihood_l],
				["Unlabeled", log_likelihood_u]
			]
			print(tabulate(table, headers=["Log-likelihood", "Dev"]))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# 以下のどちらかを必ず指定
	parser.add_argument("--train-filename", "-file", type=str, default=None, help="訓練用のテキストファイルのパス")
	parser.add_argument("--train-directory", "-dir", type=str, default=None, help="訓練用のテキストファイルが入っているディレクトリ")

	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--epochs", "-e", type=int, default=100000, help="総epoch")
	parser.add_argument("--working-directory", "-cwd", type=str, default="out", help="ワーキングディレクトリ")
	parser.add_argument("--train-dev-split", "-td-split", type=float, default=0.9, help="テキストデータの何割を訓練データにするか")
	parser.add_argument("--semi-supervised-split", "-ssl-split", type=float, default=0.1, help="テキストデータの何割を教師データにするか")
	parser.add_argument("--neologd-path", "-neologd", type=str, default=None)

	# NPYLM
	parser.add_argument("--lambda-a", "-lam-a", type=float, default=4)
	parser.add_argument("--lambda-b", "-lam-b", type=float, default=1)
	parser.add_argument("--vpylm-beta-stop", "-beta-stop", type=float, default=4)
	parser.add_argument("--vpylm-beta-pass", "-beta-pass", type=float, default=1)
	parser.add_argument("--max-word-length", "-l", type=int, default=12, help="可能な単語の最大長.")

	# CRF
	# Input characters/numbers/letters locating at positions i−2, i−1, i, i+1, i+2
	parser.add_argument("--crf-feature-x-unigram-start", type=int, default=-2)
	parser.add_argument("--crf-feature-x-unigram-end", type=int, default=2)
	# The character/number/letter bigrams locating at positions i−2, i−1, i, i+1
	parser.add_argument("--crf-feature-x-bigram-start", type=int, default=-2)
	parser.add_argument("--crf-feature-x-bigram-end", type=int, default=1)
	# Whether xj and xj+1 are identical, for j = (i−2)...(i + 1)
	parser.add_argument("--crf-feature-x-identical-1-start", type=int, default=-2)
	parser.add_argument("--crf-feature-x-identical-1-end", type=int, default=1)
	# Whether xj and xj+2 are identical, for j = (i−3)...(i + 1)
	parser.add_argument("--crf-feature-x-identical-2-start", type=int, default=-3)
	parser.add_argument("--crf-feature-x-identical-2-end", type=int, default=1)
	parser.add_argument("--crf-prior-sigma", type=float, default=1.0)
	parser.add_argument("--crf-learning-rate", type=float, default=0.001)

	args = parser.parse_args()

	main()
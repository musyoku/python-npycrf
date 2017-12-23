#include <boost/python.hpp>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <iostream>
#include "../npycrf/sampler.h"
#include "../npycrf/wordtype.h"
#include "../npycrf/hash.h"
#include "trainer.h"

namespace npycrf {
	namespace python {
		Trainer::Trainer(Dataset* dataset_l, Dataset* dataset_u, Dictionary* dict, NPYCRF* npycrf, double crf_regularization_constant){
			_dataset_l = dataset_l;
			_dataset_u = dataset_u;
			_dict = dict;
			_npycrf = npycrf;
			_sgd = new solver::SGD(npycrf->_crf, crf_regularization_constant);
			_vpylm_sampling_probability_table = array<double>(_dict->get_num_characters() + 1);	// </s>を含む
			_total_gibbs_iterations = 0;

			// 教師なしデータ
			int num_data = dataset_u->get_size_train();
			_added_to_npylm_u = array<bool>(num_data);
			for(int data_index = 0;data_index < num_data;data_index++){
				_rand_indices_train_u.push_back(data_index);
				_added_to_npylm_u[data_index] = false;
			}
			for(int data_index = 0;data_index < dataset_u->_sentences_dev.size();data_index++){
				_rand_indices_dev_u.push_back(data_index);
			}
			
			// 教師ありデータ
			num_data = dataset_l->get_size_train();
			_added_to_npylm_l = array<bool>(num_data);
			for(int data_index = 0;data_index < num_data;data_index++){
				_rand_indices_train_l.push_back(data_index);
				_added_to_npylm_l[data_index] = false;
			}
			for(int data_index = 0;data_index < dataset_l->_sentences_dev.size();data_index++){
				_rand_indices_dev_l.push_back(data_index);
			}

			// 単語の最大長をチェック
			auto check_word_length = [&npycrf](std::vector<Sentence*> &sentences){
				for(auto &sentence: sentences){
					for(int t = 0;t < sentence->get_num_segments();t++){
						int word_length = sentence->get_word_length_at(t);
						if(word_length > npycrf->get_max_word_length()){
							std::cout << "max_word_length must be greater or equal to " << word_length << std::endl;
						}
					}
				}
			};
			check_word_length(dataset_l->_sentences_train);
			check_word_length(dataset_l->_sentences_dev);

			// CRF素性を展開
			// 教師付きデータはCRFの初期化時に展開されているので本来しなくてもいい
			for(Sentence* sentence: dataset_l->_sentences_train){
				if(sentence->_features == NULL){
					sentence->_features = npycrf->_crf->extract_features(sentence, false);
				}
			}
			for(Sentence* sentence: dataset_l->_sentences_dev){
				if(sentence->_features == NULL){
					sentence->_features = npycrf->_crf->extract_features(sentence, false);
				}
			}
			for(Sentence* sentence: dataset_u->_sentences_train){
				assert(sentence->_features == NULL);
				sentence->_features = npycrf->_crf->extract_features(sentence, false);
				assert(sentence->_num_segments == 4);
			}
			for(Sentence* sentence: dataset_u->_sentences_dev){
				assert(sentence->_features == NULL);
				sentence->_features = npycrf->_crf->extract_features(sentence, false);
				assert(sentence->_num_segments == 4);
			}

			// 必要な領域を確保
			int max_word_length = npycrf->_npylm->_max_word_length;
			int max_sentence_length = std::max(dataset_l->get_max_sentence_length(), dataset_u->get_max_sentence_length());
			npycrf->_npylm->reserve(max_sentence_length);
			npycrf->_lattice->reserve(max_word_length, max_sentence_length);
		}
		// HPYLM,VPYLMのdとthetaをサンプリング
		void Trainer::sample_hpylm_vpylm_hyperparameters(){
			_npycrf->_npylm->sample_hpylm_vpylm_hyperparameters();
		}
		// 文字種ごとにNPYLMのλをサンプリング
		void Trainer::sample_npylm_lambda(){
			std::vector<double> a_for_type(WORDTYPE_NUM_TYPES + 1, 0.0);
			std::vector<double> b_for_type(WORDTYPE_NUM_TYPES + 1, 0.0);
			std::unordered_set<id> words;
			npylm::NPYLM* npylm = _npycrf->_npylm;
			for(int type = 1;type <= WORDTYPE_NUM_TYPES;type++){
				a_for_type[type] = npylm->_lambda_a;
				b_for_type[type] = npylm->_lambda_b;
			}
			auto enumerate_words = [&a_for_type, &b_for_type, &words, &npylm](std::vector<Sentence*> &dataset) {
				for(auto sentence: dataset){
					// <bos>と<eos>は除外
					for(int t = 2;t < sentence->get_num_segments() - 1;t++){
						std::wstring word = sentence->get_word_str_at(t);
						id word_id = sentence->get_word_id_at(t);
						int word_length = sentence->get_word_length_at(t);
						if(word_length > npylm->_max_word_length){
							continue;
						}
						if(words.find(word_id) == words.end()){
							std::vector<int> &tables = npylm->_hpylm->_root->_arrangement[word_id];
							int t_w = tables.size();
							int type = wordtype::detect_word_type(word);
							a_for_type[type] += t_w * word_length;
							b_for_type[type] += t_w;
							words.insert(word_id);
						}
					}
				}
			};
			enumerate_words(_dataset_l->_sentences_train);
			enumerate_words(_dataset_u->_sentences_train);
			for(int type = 1;type <= WORDTYPE_NUM_TYPES;type++){
				double lambda = sampler::gamma(a_for_type[type], b_for_type[type]);
				npylm->_lambda_for_type[type] = lambda;
			}
		}
		// VPYLMに文脈を渡し次の文字を生成
		int Trainer::sample_word_from_vpylm_given_context(npycrf::array<int> &context_ids, int sample_t){
			double sum_probs = 0;
			npylm::lm::VPYLM* vpylm = _npycrf->_npylm->_vpylm;
			auto &all_characters = _dict->_map_character_to_id;
			int num_characters = _dict->get_num_characters();
			for(auto elem: all_characters){
				int character_id = elem.second; 
				double pw = vpylm->compute_p_w_given_h(character_id, context_ids, 0, sample_t - 1);
				sum_probs += pw;
				_vpylm_sampling_probability_table[character_id] = pw;
			}

			double normalizer = 1.0 / sum_probs;
			double r = sampler::uniform(0, 1);
			double stack = 0;
			for(int character_id = 0;character_id < num_characters;character_id++){
				stack += _vpylm_sampling_probability_table[character_id] * normalizer;
				if(r <= stack){
					return character_id;
				}
			}
			return SPECIAL_CHARACTER_END;
		}
		// VPYLMから長さkの単語が出現する確率をキャッシュする
		void Trainer::update_p_k_given_vpylm(){
			int num_samples = 20000;
			int early_stopping_threshold = 10;
			int max_word_length = _npycrf->get_max_word_length() + 1; // 最大+1
			npycrf::array<double> &pk_vpylm = _npycrf->_npylm->_pk_vpylm;
			npycrf::array<int> num_words_of_k(max_word_length + 1);
			for(int i = 0;i <= max_word_length;i++){
				pk_vpylm[i] = 0;
				num_words_of_k[i] = 0;
			}
			npylm::lm::VPYLM* vpylm = _npycrf->_npylm->_vpylm;
			npycrf::array<int> character_ids(max_word_length + 1);
			double sum_words = 0;
			auto &all_characters = _dict->_map_character_to_id;
			int num_characters = _dict->get_num_characters();
			// std::cout << "num_characters: " << num_characters << std::endl;
			// std::cout << "all_characters: " << all_characters.size() << std::endl;
			npycrf::array<double> unigram_distribution(num_characters);
			double sum_probs = 0;
			for(auto elem: all_characters){
				int character_id = elem.second; 
				character_ids[0] = character_id;
				if(character_id == SPECIAL_CHARACTER_END || character_id == SPECIAL_CHARACTER_BEGIN){
					unigram_distribution[character_id] = 0;
					continue;
				}
				double pw = vpylm->compute_p_w(character_ids, 0, 0);
				sum_probs += pw;
				unigram_distribution[character_id] = pw;
				// std::cout << character_id << ": " << "character_id=" << character_id << ", pw=" << pw << std::endl;
			}
			// std::cout << "sum_probs: " << sum_probs << std::endl;
			for(int m = 1;m <= num_samples;m++){
				if (PyErr_CheckSignals() != 0) {	// ctrl+cが押されたかチェック
					return;		
				}
				int start_character_id = -1;
				double normalizer = 1.0 / sum_probs;
				double r = sampler::uniform(0, 1);
				double stack = 0;
				for(int character_id = 0;character_id < num_characters;character_id++){
					stack += unigram_distribution[character_id] * normalizer;
					if(r <= stack){
						start_character_id = character_id;
						break;
					}
				}
				assert(start_character_id != -1);
				// wcout << "m = " << m << endl;
				character_ids[0] = start_character_id;
				int word_length = 1;
				for(int k = 1;k < max_word_length;k++){
					int next_character_id = sample_word_from_vpylm_given_context(character_ids, k);
					character_ids[k] = next_character_id;
					if(next_character_id == SPECIAL_CHARACTER_END){
						break;
					}
					word_length += 1;
				}

				// std::cout << "length: " << word_length << std::endl;
				// std::wstring str = L"";
				// for(int u = 0;u < word_length;u++){
				// 	if(character_ids[u] == SPECIAL_CHARACTER_END){
				// 		break;
				// 	}
				// 	str += _dict->_map_id_to_character[character_ids[u]];
				// 	std::cout << character_ids[u] << ", ";
				// }
				// std::cout << std::endl;
				// std::wcout << str << std::endl;

				sum_words += 1;
				if(word_length == 0){	// <bow><eow>
					continue;
				}
				assert(word_length <= max_word_length);
				num_words_of_k[word_length] += 1;

				// すべてのkが生成されていたら早期終了
				if(m % 100 == 0){
					bool stop = true;
					for(int k = 1;k <= max_word_length;k++){
						if(num_words_of_k[k] < early_stopping_threshold){
							stop = false;
							break;
						}
					}
					if(stop){
						break;
					}
				}
			}
			for(int k = 1;k <= max_word_length;k++){
				pk_vpylm[k] = (num_words_of_k[k] + 1) / (sum_words + max_word_length);	// ラプラススムージングを入れておく
				// std::cout << "k = " << k << ", " << pk_vpylm[k] << std::endl;
				assert(pk_vpylm[k] > 0);
			}
		}
		// 単語分割のギブスサンプリング
		void Trainer::gibbs(bool include_labeled_data){
			_npycrf->_npylm->_fix_g0_using_poisson = true;
			if(_total_gibbs_iterations < 3){
				_npycrf->_npylm->_fix_g0_using_poisson = false;
			}
			// 教師なしデータでモデルパラメータを更新
			std::vector<int> segments;		// 分割の一時保存用
			shuffle(_rand_indices_train_u.begin(), _rand_indices_train_u.end(), sampler::mt);		// データをシャッフル
			auto start_time = std::chrono::system_clock::now();
			for(int i = 0;i < _rand_indices_train_u.size();i++){
				if (PyErr_CheckSignals() != 0) {	// ctrl+cが押されたかチェック
					return;		
				}
				// 訓練データを一つ取り出す
				int data_index = _rand_indices_train_u[i];
				assert(data_index < _dataset_u->get_size_train());
				Sentence* sentence = _dataset_u->_sentences_train[data_index];
				assert(sentence->_features != NULL);

				// NPYLMのg0キャッシュを消去（消す理由は単に重くなるから）
				_npycrf->_npylm->clear_g0_cache();

				// モデルに追加されているかチェック
				if(_added_to_npylm_u[data_index] == true){
					// 古い分割をモデルから削除
					for(int t = 2;t < sentence->get_num_segments();t++){
						_npycrf->_npylm->remove_customer_at_time_t(sentence, t);
					}
					
					#ifdef __DEBUG__
						// 正規化しない場合の結果と比較するためシードを合わせる
						int seed = (unsigned int)time(NULL);
						sampler::mt.seed(seed);
					#endif

					// 新しい分割を取得
					_npycrf->_lattice->blocked_gibbs(sentence, segments, true);
					sentence->split(segments);
					
					#ifdef __DEBUG__
						// 正規化しない場合の結果と比較
						// std::cout << sentence->size() << std::endl;
						if(sentence->size() < 150){
							std::vector<int> a = segments;
							sampler::mt.seed(seed);
							_npycrf->_lattice->blocked_gibbs(sentence, segments, false);
							std::vector<int> b = segments;
							if(a.size() != b.size()){
								sentence->dump_words();
							}
							assert(a.size() == b.size());
							for(int i = 0;i < a.size();i++){
								assert(a[i] == b[i]);
							}
						}
					#endif
				}
				// 新しい分割結果をモデルに追加
				for(int t = 2;t < sentence->get_num_segments();t++){
					_npycrf->_npylm->add_customer_at_time_t(sentence, t);
				}
				_added_to_npylm_u[data_index] = true;

				if(i % 100 == 0 || i == _rand_indices_train_u.size() - 1){
					auto diff = std::chrono::system_clock::now() - start_time;
					double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / 1000.0;
					double gibbs_per_sec = (double)(i + 1) / elapsed_time;
					double percent = (double)i / (double)_rand_indices_train_u.size() * 100.0;
					std::cout << "\r\033[2K" << i << "/" << _rand_indices_train_u.size() << " (" << std::fixed << std::setprecision(2) << percent << "%) " << gibbs_per_sec << " gibbs/s" << std::flush;
				}
			}
			std::cout << "\r\033[2K" << std::flush;

			// 客数チェック
			assert(_npycrf->_npylm->_hpylm->_root->_num_tables <= _npycrf->_npylm->_vpylm->get_num_customers());

			if(include_labeled_data){
				_gibbs_labeled();
			}

			_total_gibbs_iterations += 1;
		}
		void Trainer::add_labeled_data_to_npylm(){
			_gibbs_labeled();
		}
		// 教師ありデータでモデルパラメータを更新
		void Trainer::_gibbs_labeled(){
			_npycrf->_npylm->_fix_g0_using_poisson = true;
			if(_total_gibbs_iterations < 3){
				_npycrf->_npylm->_fix_g0_using_poisson = false;
			}
			shuffle(_rand_indices_train_l.begin(), _rand_indices_train_l.end(), sampler::mt);		// データをシャッフル
			for(int i = 0;i < _rand_indices_train_l.size();i++){
				if (PyErr_CheckSignals() != 0) {	// ctrl+cが押されたかチェック
					return;		
				}
				// 訓練データを一つ取り出す
				int data_index = _rand_indices_train_l[i];
				assert(data_index < _dataset_l->get_size_train());
				Sentence* sentence = _dataset_l->_sentences_train[data_index];
				assert(sentence->_features != NULL);

				// NPYLMのg0キャッシュを消去（消す理由は単に重くなるから）
				_npycrf->_npylm->clear_g0_cache();

				// 教師あり
				// モデルに追加されているかチェック
				if(_added_to_npylm_l[data_index] == true){
					// 古い分割をモデルから削除
					for(int t = 2;t < sentence->get_num_segments();t++){
						_npycrf->_npylm->remove_customer_at_time_t(sentence, t);
					}
				}
				// 同じ分割結果を再度モデルに追加
				// 追加と削除を繰り返すことでHPYLMとVPYLMのパラメータ（客の配置）がギブスサンプリングされる
				for(int t = 2;t < sentence->get_num_segments();t++){
					_npycrf->_npylm->add_customer_at_time_t(sentence, t);
				}
				_added_to_npylm_l[data_index] = true;
			}
		}
		void Trainer::sgd(double learning_rate, int batchsize, bool pure_crf){
			shuffle(_rand_indices_train_l.begin(), _rand_indices_train_l.end(), sampler::mt);		// データをシャッフル
			int total_batches = (double)_rand_indices_train_l.size() / (double)batchsize + ((_rand_indices_train_l.size() % batchsize) ? 1 : 0);
			Lattice* lattice = _npycrf->_lattice;
			mat::quad<double> &p_conc_tkji = lattice->_p_conc_tkji;
			mat::tri<double> &pz_s = lattice->_pz_s;
			mat::quad<double> &pw_h_tkji = lattice->_pw_h_tkji;
			lattice->set_pure_crf_mode(pure_crf);
			int num_completed = 0;
			auto start_time = std::chrono::system_clock::now();
			for(int b = 0;b < total_batches;b++){
				_sgd->clear_grads();
				int size = std::min(batchsize, (int)(_rand_indices_train_l.size() - batchsize * b));
				for(int i = 0;i < size;i++){
					assert(lattice->get_pure_crf_mode() == pure_crf);
					if (PyErr_CheckSignals() != 0) {	// ctrl+cが押されたかチェック
						return;		
					}
					int data_index = _rand_indices_train_l[i + batchsize * b];

					// NPYLMのg0キャッシュを消去（消す理由は単に重くなるから）
					_npycrf->_npylm->clear_g0_cache();

					// std::cout << "data_index: " << data_index << std::endl;
					Sentence* sentence = _dataset_l->_sentences_train[data_index];
					assert(sentence->_features != NULL);
					if(pure_crf){
						// 周辺確率を求める
						lattice->enumerate_marginal_p_path_given_sentence(sentence, pz_s);
						// 更新
						_sgd->backward_crf(sentence, pz_s);
					}else{
						lattice->enumerate_marginal_p_path_and_trigram_given_sentence(sentence, p_conc_tkji, pw_h_tkji, pz_s);
						_sgd->backward_crf(sentence, pz_s);
						_sgd->backward_lambda_0(sentence, p_conc_tkji, pw_h_tkji, lattice->_max_word_length);
					}
					// 周辺確率を求める
				}
				_sgd->update(learning_rate / (double)size);	// 勾配の平均をとるため学習率を調整

				num_completed += size;
				auto diff = std::chrono::system_clock::now() - start_time;
				double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / 1000.0;
				double sgd_per_sec = (double)num_completed / elapsed_time;
				double percent = (double)b / (double)total_batches * 100.0;
				std::cout << "\r\033[2K" << num_completed << "/" << _rand_indices_train_l.size() << " (" << std::fixed << std::setprecision(2) << percent << "%) " << sgd_per_sec << " sgd/s" << std::flush;
			}
			std::cout << "\r\033[2K" << std::flush;
			lattice->set_npycrf_mode();
		}
		double Trainer::compute_perplexity_train(){
			return _compute_perplexity(_dataset_u->_sentences_train);
		}
		double Trainer::compute_perplexity_dev(){
			return _compute_perplexity(_dataset_u->_sentences_dev);
		}
		double Trainer::_compute_perplexity(std::vector<Sentence*> &dataset){
			if(dataset.size() == 0){
				return 0;
			}
			double ppl = 0;
			int num_sentences = dataset.size();
			std::vector<int> segments;		// 分割の一時保存用
			for(int data_index = 0;data_index < num_sentences;data_index++){
				if (PyErr_CheckSignals() != 0) {	// ctrl+cが押されたかチェック
					return 0;		
				}
				Sentence* sentence = dataset[data_index]->copy();	// 干渉を防ぐためコピー
				_npycrf->_lattice->viterbi_decode(sentence, segments);
				sentence->split(segments);
				ppl += _npycrf->_npylm->compute_log_p_y_given_sentence(sentence) / ((double)sentence->get_num_segments() - 2);
				delete sentence;
			}
			ppl = exp(-ppl / num_sentences);
			return ppl;
		}
		double Trainer::compute_log_likelihood_labeled_train(){
			return _compute_log_likelihood(_dataset_l->_sentences_train, true);
		}
		double Trainer::compute_log_likelihood_unlabeled_train(){
			return _compute_log_likelihood(_dataset_u->_sentences_train, false);
		}
		double Trainer::compute_log_likelihood_labeled_dev(){
			return _compute_log_likelihood(_dataset_l->_sentences_dev, true);
		}
		double Trainer::compute_log_likelihood_unlabeled_dev(){
			return _compute_log_likelihood(_dataset_u->_sentences_dev, false);
		}
		double Trainer::_compute_log_likelihood(std::vector<Sentence*> &dataset, bool labeled){
			if(dataset.size() == 0){
				return 0;
			}
			std::vector<int> segments;		// 分割の一時保存用
			double sum_log_likelihood = 0;
			int num_sentences = dataset.size();
			for(int data_index = 0;data_index < num_sentences;data_index++){
				if (PyErr_CheckSignals() != 0) {	// ctrl+cが押されたかチェック
					return 0;		
				}
				Sentence* sentence = dataset[data_index]->copy();
				if(labeled == false){	// 教師なしデータの場合は最尤分解を求める
					_npycrf->parse(sentence);
				}
				double log_py_s = _npycrf->compute_log_proportional_p_y_given_sentence(sentence);
				double log_Zs = _npycrf->compute_log_normalizing_constant(sentence);
				sum_log_likelihood += log_py_s - log_Zs;
				delete sentence;
			}
			return sum_log_likelihood;
		}
		// デバッグ用
		void Trainer::remove_all_data(){
			for(int data_index = 0;data_index < _dataset_u->get_size_train();data_index++){
				if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
					return;
				}
				Sentence* sentence = _dataset_u->_sentences_train[data_index];
				// 古い分割をモデルから削除
				if(_added_to_npylm_u[data_index] == true){
					for(int t = 2;t < sentence->get_num_segments();t++){
						_npycrf->_npylm->remove_customer_at_time_t(sentence, t);
					}
				}
			}
			for(int data_index = 0;data_index < _dataset_l->get_size_train();data_index++){
				if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
					return;
				}
				Sentence* sentence = _dataset_l->_sentences_train[data_index];
				// 古い分割をモデルから削除
				if(_added_to_npylm_l[data_index] == true){
					for(int t = 2;t < sentence->get_num_segments();t++){
						_npycrf->_npylm->remove_customer_at_time_t(sentence, t);
					}
				}
			}
		}
		// 教師付き評価データの分割のF値を求める
		boost::python::list Trainer::compute_precision_and_recall_labeled_dev(){
			double mean_precision = 0;
			double mean_recall = 0;
			for(Sentence* sentence_true: _dataset_l->_sentences_dev){
				if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
					break;
				}
				Sentence* sentence_estimated = sentence_true->copy();
				_npycrf->parse(sentence_estimated);
				array<int> &labels_true = sentence_true->_labels;
				array<int> &labels_estimated = sentence_estimated->_labels;
				int sentence_size = sentence_true->size();

				// 一致する個数を調べる
				int num_correct = 0;
				bool continuing_word = true;
				for(int i = 1;i <= sentence_size;i++){
					if(labels_estimated[i] == 1){
						if(labels_true[i] == 1 && continuing_word == true){
							num_correct += 1;
						}
						continuing_word = true;
						if(labels_true[i] != 1){
							continuing_word = false;
						}
						continue;
					}
					if(labels_estimated[i] != labels_true[i]){
						continuing_word = false;
					}
				}

				// 精度を求める
				// 分母はモデルによる分割の単語数になる
				double precision = (double)num_correct / (double)(sentence_estimated->_num_segments - 3);

				// 再現率を求める
				// 分母は正解分割の単語数になる
				double recall = (double)num_correct / (double)(sentence_true->_num_segments - 3);

				mean_precision += precision;
				mean_recall += recall;
			}

			mean_precision /= (double)_dataset_l->_sentences_dev.size();
			mean_recall /= (double)_dataset_l->_sentences_dev.size();

			boost::python::list result;
			result.append(mean_precision);
			result.append(mean_recall);
			return result;
		}
		void Trainer::print_p_k_vpylm(){
			int max_word_length = _npycrf->get_max_word_length();
			npycrf::array<double> &pk_vpylm = _npycrf->_npylm->_pk_vpylm;
			for(int k = 1;k <= max_word_length;k++){
				std::cout << "k = " << k << ", " << pk_vpylm[k] << std::endl;
			}
		}
		void Trainer::print_segmentation_labeled_train(int num_to_print){
			_print_segmentation(num_to_print, _dataset_l->_sentences_train, _rand_indices_train_l);
		}
		void Trainer::print_segmentation_unlabeled_train(int num_to_print){
			_print_segmentation(num_to_print, _dataset_u->_sentences_train, _rand_indices_train_u);
		}
		void Trainer::print_segmentation_labeled_dev(int num_to_print){
			shuffle(_rand_indices_dev_l.begin(), _rand_indices_dev_l.end(), sampler::mt);
			_print_segmentation(num_to_print, _dataset_l->_sentences_dev, _rand_indices_dev_l);
		}
		void Trainer::print_segmentation_unlabeled_dev(int num_to_print){
			shuffle(_rand_indices_dev_u.begin(), _rand_indices_dev_u.end(), sampler::mt);
			_print_segmentation(num_to_print, _dataset_u->_sentences_dev, _rand_indices_dev_u);
		}
		void Trainer::_print_segmentation(int num_to_print, std::vector<Sentence*> &dataset, std::vector<int> &rand_indices){
			num_to_print = std::min((int)dataset.size(), num_to_print);
			std::vector<int> segments;		// 分割の一時保存用
			for(int n = 0;n < num_to_print;n++){
				if (PyErr_CheckSignals() != 0) {	// ctrl+cが押されたかチェック
					return;		
				}
				int data_index = rand_indices[n];
				Sentence* sentence = dataset[data_index]->copy();
				_npycrf->_npylm->reserve(sentence->size());
				_npycrf->_npylm->clear_g0_cache();
				_npycrf->_lattice->viterbi_decode(sentence, segments);
				sentence->split(segments);
				sentence->dump_words();
				delete sentence;
			}
		}
		int Trainer::detect_hash_collision(int max_word_length){
			std::unordered_map<id, std::wstring> pool;
			auto exec = [&pool, &max_word_length](Sentence* sentence){
				for(int t = 1;t <= sentence->size();t++){
					for(int k = 1;k <= std::min(t, max_word_length);k++){
						if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
							return;
						}
						id word_id = sentence->get_substr_word_id(t - k, t - 1);
						std::wstring word = sentence->get_substr_word_str(t - k, t - 1);
						assert(word_id == hash_wstring(word));
						auto itr = pool.find(word_id);
						if(itr == pool.end()){
							pool[word_id] = word;
						}else{
							assert(itr->second == word);
						}
					}
				}
			};
			int step = 0;
			for(Sentence* sentence: _dataset_u->_sentences_train){
				if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
					return 0;
				}
				exec(sentence);
				step++;
			}
			for(Sentence* sentence: _dataset_u->_sentences_dev){
				if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
					return 0;
				}
				exec(sentence);
				step++;
			}
			for(Sentence* sentence: _dataset_l->_sentences_train){
				if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
					return 0;
				}
				exec(sentence);
				step++;
			}
			for(Sentence* sentence: _dataset_l->_sentences_dev){
				if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
					return 0;
				}
				exec(sentence);
				step++;
			}
			return pool.size();
		}
	}
}
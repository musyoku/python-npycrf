#include <iostream>
#include <chrono>
#include <sstream>
#include <utility>
#include "../../src/npycrf/sampler.h"
#include "../../src/npycrf/ctype.h"
#include "../../src/python/npycrf.h"
#include "../../src/python/dataset.h"
#include "../../src/python/dictionary.h"
#include "../../src/python/trainer.h"
#include "../../src/python/model/crf.h"
#include "../../src/python/model/npylm.h"
using namespace npycrf;
using namespace npycrf::python;
using std::cout;
using std::flush;
using std::endl;

std::vector<std::wstring> explode(std::wstring const &s, wchar_t delim){
    std::vector<std::wstring> result;
    std::wistringstream iss(s);

    for (std::wstring token; std::getline(iss, token, delim);){
		std::wstring word = std::move(token);
		if(word.size() > 0){
			result.push_back(word);
		}
    }
    return result;
}

void run_training_loop(){
	setlocale(LC_CTYPE, "ja_JP.UTF-8");
	std::ios_base::sync_with_stdio(false);
	std::locale default_loc("ja_JP.UTF-8");
	std::locale::global(default_loc);
	std::locale ctype_default(std::locale::classic(), default_loc, std::locale::ctype); //※
	std::wcout.imbue(ctype_default);
	std::wcin.imbue(ctype_default);
	
	int max_word_length = 12;

	Corpus* corpus_u = new Corpus();
	std::string filename_u = "../../dataset/aozora/neko.txt";
	std::wifstream ifs_u(filename_u.c_str());
	std::wstring sentence_str;
	assert(ifs_u.fail() == false);
	while(getline(ifs_u, sentence_str)){
		if (sentence_str.empty()){
			continue;
		}
		// std::wcout << sentence_str << std::endl;
		std::vector<std::wstring> words = {sentence_str};
		corpus_u->add_words(words);
		if(corpus_u->_word_sequences.size() > 500){
			break;
		}
	}
	ifs_u.close();
	
	Corpus* corpus_l = new Corpus();
	std::string filename_l = "../../dataset/mecab.txt";
	int num_labelded_data = 1000;
	std::wifstream ifs_l(filename_l.c_str());
	assert(ifs_l.fail() == false);
	int i = 0;
	while(getline(ifs_l, sentence_str)){
		if (sentence_str.empty()){
			continue;
		}
		i++;
		std::vector<std::wstring> words = explode(sentence_str, L' ');
		for(auto word: words){
			std::wcout << word << L" ";
			if(word.size() > max_word_length){
				words.clear();
				break;
			}
		}
		if(words.size() == 0){
			continue;
		}
		std::wcout << std::endl;
		corpus_l->add_words(words);
		if(i > num_labelded_data){
			break;
		}
	}
	
	Dictionary* dict = new Dictionary();

	int seed = 0;
	Dataset* dataset_l = new Dataset(corpus_l, dict, 1, seed);
	Dataset* dataset_u = new Dataset(corpus_u, dict, 1, seed);

	double g0 = 1.0 / (double)dict->get_num_characters();
	double initial_lambda_a = 4;
	double initial_lambda_b = 1;
	double vpylm_beta_stop = 4;
	double vpylm_beta_pass = 1;
	model::NPYLM* py_npylm1 = new model::NPYLM(max_word_length, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);
	model::NPYLM* py_npylm2 = new model::NPYLM(max_word_length, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);

	int num_character_ids = dict->get_num_characters();
	int feature_x_unigram_start = -2;
	int feature_x_unigram_end = 2;
	int feature_x_bigram_start = -2;
	int feature_x_bigram_end = 1;
	int feature_x_identical_1_start = -2;
	int feature_x_identical_1_end = 1;
	int feature_x_identical_2_start = -3;
	int feature_x_identical_2_end = 1;
	double sigma = 1;
	model::CRF* py_crf = new model::CRF(dataset_l,
										num_character_ids,
										feature_x_unigram_start,
										feature_x_unigram_end,
										feature_x_bigram_start,
										feature_x_bigram_end,
										feature_x_identical_1_start,
										feature_x_identical_1_end,
										feature_x_identical_2_start,
										feature_x_identical_2_end,
										1.0,
										sigma);

	NPYCRF* model = new NPYCRF(py_npylm1, py_crf);
	dict->save("npylm.dict");
	double learning_rate = 0.001;
	unsigned int batchsize = 32;
	double crf_regularization_constant = 1;

	std::cout << "initializing ..." << std::endl;
	Trainer* trainer = new Trainer(dataset_l, dataset_u, dict, model, crf_regularization_constant);
	trainer->add_labeled_data_to_npylm();
	trainer->sgd(learning_rate, batchsize, true);
	for(int epoch = 1;epoch < 200;epoch++){
		std::cout << "epoch " << epoch << std::endl;
		auto start_time = std::chrono::system_clock::now();


		std::cout << "gibbs1 ..." << std::endl;
		trainer->gibbs();
		std::cout << "gibbs1 ..." << std::endl;
		trainer->gibbs();
		std::cout << "gibbs1 ..." << std::endl;
		trainer->gibbs();
		std::cout << "gibbs1 ..." << std::endl;
		trainer->gibbs();
		std::cout << "gibbs1 ..." << std::endl;
		trainer->gibbs();
		std::cout << "gibbs1 ..." << std::endl;
		trainer->gibbs();
		std::cout << "gibbs1 ..." << std::endl;
		trainer->gibbs();

		// model->_npylm = py_npylm2->_npylm;
		// model->_lattice->_npylm =  py_npylm2->_npylm;
		// int max_sentence_length = std::max(dataset_l->get_max_sentence_length(), dataset_u->get_max_sentence_length());
		// model->_npylm->reserve(max_sentence_length);
		// trainer->_added_to_npylm_u.fill(false);
		// std::cout << "gibbs2 ..." << std::endl;
		// trainer->gibbs();
		// std::cout << "gibbs2 ..." << std::endl;
		// trainer->gibbs();

		// std::cout << model->_npylm->_hpylm->_root->get_num_nodes() << std::endl;
		// std::cout << model->_npylm->_hpylm->_root->get_num_customers() << std::endl;
		// std::cout << model->_npylm->_hpylm->_root->get_num_tables() << std::endl;
		// std::cout << model->_npylm->_vpylm->_root->get_num_nodes() << std::endl;
		// std::cout << model->_npylm->_vpylm->_root->get_num_customers() << std::endl;
		// std::cout << model->_npylm->_vpylm->_root->get_num_tables() << std::endl;
		// std::cout << model->_npylm->_vpylm->_root->sum_pass_counts() << std::endl;
		// std::cout << model->_npylm->_vpylm->_root->sum_stop_counts() << std::endl;

		{
			model->_npylm->_fix_g0_using_poisson = true;
			if(trainer->_total_gibbs_iterations < 3){
				model->_npylm->_fix_g0_using_poisson = false;
			}
			// 教師なしデータでモデルパラメータを更新
			std::vector<int> segments;		// 分割の一時保存用
			for(int data_index = 0;data_index < dataset_u->_sentences_train.size();data_index++){
				Sentence* sentence = dataset_u->_sentences_train[data_index];
				assert(sentence->_features != NULL);
				// モデルに追加されているかチェック
				if(trainer->_added_to_npylm_u[data_index] == true){
					// 古い分割をモデルから削除
					// for(int t = 2;t < sentence->get_num_segments();t++){
					// 	model->_npylm->remove_customer_at_time_t(sentence, t);
					// }
					// for(int t = 2;t < sentence->get_num_segments();t++){
					// 	model->_npylm->add_customer_at_time_t(sentence, t);
					// }
					model->_npylm->_g0_cache.clear();
					// 新しい分割を取得
					model->_lattice->blocked_gibbs(sentence, segments, true);
					// sentence->split(segments);
				}
				// 新しい分割結果をモデルに追加
				trainer->_added_to_npylm_u[data_index] = true;
			}
		}

		// std::vector<int> segments;		// 分割の一時保存用
		// // model->_lattice->set_pure_crf_mode(false);
		// for(int i = 0;i < std::min(1000, (int)dataset_u->_sentences_train.size());i++){
		// 	Sentence* sentence = dataset_u->_sentences_train[i]->copy();
		// 	// mat::quad<double> &p_conc_tkji = model->_lattice->_p_conc_tkji;
		// 	// mat::quad<double> &p_transition_tkji = model->_lattice->_p_transition_tkji;
		// 	// mat::tri<double> &pz_s = model->_lattice->_pz_s;
		// 	// mat::quad<double> &pw_h_tkji = model->_lattice->_pw_h_tkji;
		// 	// model->_lattice->reserve(model->_lattice->_max_word_length, sentence->size());
		// 	// model->_lattice->_clear_word_id_cache(sentence->size());
		// 	// p_transition_tkji.fill(-1, sentence->size());
		// 	// pw_h_tkji.fill(-1, sentence->size());

		// 	model->_lattice->blocked_gibbs(sentence, segments, true);
		// 	// model->_lattice->_clear_word_id_cache(sentence->size());
		// 	// model->_lattice->_clear_p_tkji(sentence->size());
		// 	// model->_lattice->forward_filtering(sentence, true);
		// 	// model->_lattice->_enumerate_forward_variables(sentence, model->_lattice->_alpha, pw_h_tkji, p_transition_tkji, model->_lattice->_scaling, true);
		// 	// model->_lattice->_enumerate_backward_variables(sentence, model->_lattice->_beta, p_transition_tkji, model->_lattice->_scaling, true);
		// 	// model->_lattice->_enumerate_marginal_p_path_given_sentence(sentence, pz_s, model->_lattice->_alpha, model->_lattice->_beta);
		// 	// model->_lattice->_enumerate_marginal_p_trigram_given_sentence(sentence, p_conc_tkji, model->_lattice->_alpha, model->_lattice->_beta, p_transition_tkji, model->_lattice->_scaling, true);
		// 	// model->_lattice->enumerate_marginal_p_path_and_trigram_given_sentence(sentence, p_conc_tkji, pw_h_tkji, pz_s);
		// 	// trainer->_sgd->clear_grads();
		// 	// trainer->_sgd->backward_crf(sentence, pz_s);
		// 	// trainer->_sgd->backward_lambda_0(sentence, p_conc_tkji, pw_h_tkji, model->_lattice->_max_word_length);
		// 	// trainer->_sgd->update(0.01);
		// }
		// model->_lattice->set_npycrf_mode();

		// std::cout << "sgd ..." << std::endl;
		// trainer->sgd(learning_rate, batchsize, false);

		// std::cout << model->_npylm->_hpylm->_root->get_num_nodes() << std::endl;
		// std::cout << model->_npylm->_hpylm->_root->get_num_customers() << std::endl;
		// std::cout << model->_npylm->_hpylm->_root->get_num_tables() << std::endl;
		// std::cout << model->_npylm->_vpylm->_root->get_num_nodes() << std::endl;
		// std::cout << model->_npylm->_vpylm->_root->get_num_customers() << std::endl;
		// std::cout << model->_npylm->_vpylm->_root->get_num_tables() << std::endl;
		// std::cout << model->_npylm->_vpylm->_root->sum_pass_counts() << std::endl;
		// std::cout << model->_npylm->_vpylm->_root->sum_stop_counts() << std::endl;
		// trainer->_added_to_npylm_u.fill(false);
		// std::cout << "gibbs2 ..." << std::endl;
		// trainer->gibbs();
		// std::cout << "gibbs2 ..." << std::endl;
		// trainer->gibbs();

		// model->_npylm = py_npylm1->_npylm;
		// model->_lattice->_npylm =  py_npylm1->_npylm;

		// trainer->_added_to_npylm_u.fill(false);
		std::cout << "gibbs2 ..." << std::endl;
		trainer->gibbs();
		std::cout << "gibbs2 ..." << std::endl;
		trainer->gibbs();

		exit(0);
	    auto diff = std::chrono::system_clock::now() - start_time;
	    cout << (std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / 1000.0) << endl;
		trainer->sample_hpylm_vpylm_hyperparameters();
		trainer->sample_npylm_lambda();
		if(epoch > 3){
			trainer->update_p_k_given_vpylm();
		}
		trainer->print_p_k_vpylm();
		cout << "lambda_0: " << py_crf->get_lambda_0() << endl;

		py_npylm1->parse(dataset_l->_sentences_train[0]);
		// if(epoch % 10 == 0){
			// trainer->print_segmentation_labeled_dev(10);
			// cout << "ppl: " << trainer->compute_perplexity_train() << endl;
			// trainer->print_segmentation_unlabeled_dev(10);
			// cout << "ppl: " << trainer->compute_perplexity_dev() << endl;
			// cout << "log_likelihood: " << trainer->compute_log_likelihood_train() << endl;
			// cout << "log_likelihood: " << trainer->compute_log_likelihood_dev() << endl;
		// }
	}
	delete dict;
	delete dataset_l;
	delete dataset_u;
	delete trainer;
	delete model;
}

int main(int argc, char *argv[]){
	for(int i = 0;i < 1000;i++){
		run_training_loop();
	}
}
#include  <iostream>
#include <chrono>
#include "../../../src/npycrf/sampler.h"
#include "../../../src/python/model.h"
#include "../../../src/python/dataset.h"
#include "../../../src/python/dictionary.h"
#include "../../../src/python/trainer.h"
using namespace npycrf;
using namespace npycrf::python;
using std::cout;
using std::flush;
using std::endl;

double compute_forward_probability(Lattice* lattice, Sentence* sentence, bool normalize){
	assert(sentence->size() <= lattice->_max_sentence_length);
	int size = sentence->size() + 1;
	lattice->_alpha[0][0][0] = 1;
	lattice->_log_z[0] = 0;
	for(int i = 0;i < size;i++){
		for(int j = 0;j < lattice->_max_word_length + 1;j++){
			lattice->_substring_word_id_cache[i][j] = 0;
		}
	}
	for(int t = 0;t < size;t++){
		lattice->_log_z[t] = 0;
		for(int k = 0;k < lattice->_max_word_length + 1;k++){
			for(int j = 0;j < lattice->_max_word_length + 1;j++){
				lattice->_alpha[t][k][j] = -1;
			}
		}
	}
	lattice->forward_filtering(sentence, normalize);
	double sum_probability = 0;
	int t = sentence->size();
	for(int k = 1;k <= std::min(t, lattice->_max_word_length);k++){
		for(int j = 1;j <= std::min(t - k, lattice->_max_word_length);j++){
			if(normalize){
				sum_probability += lattice->_alpha[t][k][j] * exp(lattice->_log_z[t]);
			}else{
				sum_probability += lattice->_alpha[t][k][j];
			}
		}
	}
	return sum_probability;
}

double compute_backward_probability(Lattice* lattice, Sentence* sentence, bool normalize){
		assert(sentence->size() <= lattice->_max_sentence_length);
		int size = sentence->size() + 1;
		lattice->_beta[0][0][0] = 1;
		lattice->_log_z[0] = 0;
		for(int i = 0;i < size;i++){
			for(int j = 0;j < lattice->_max_word_length + 1;j++){
				lattice->_substring_word_id_cache[i][j] = 0;
			}
		}
		#ifdef __DEBUG__
			for(int t = 0;t < size;t++){
				for(int k = 0;k < lattice->_max_word_length + 1;k++){
					for(int j = 0;j < lattice->_max_word_length + 1;j++){
						lattice->_beta[t][k][j] = -1;
					}
				}
			}
		#endif 
		lattice->backward_filtering(sentence, normalize);
		double sum_probability = 0;
		int t = sentence->size();
		for(int k = 1;k <= std::min(t, lattice->_max_word_length);k++){
			for(int j = 1;j <= std::min(t - k, lattice->_max_word_length);j++){
				if(normalize){
					sum_probability += lattice->_beta[t][k][j] * exp(lattice->_log_z[t]);
				}else{
					sum_probability += lattice->_beta[t][k][j];
				}
			}
		}
		return sum_probability;
}

void test_compute_forward_probability(){
	std::string filename = "../../../dataset/test.txt";
	Corpus* corpus = new Corpus();
	corpus->add_textfile(filename);
	int seed = 0;
	Dataset* dataset = new Dataset(corpus, 1, seed);

	double lambda_0 = 1;
	int max_word_length = 12;
	int max_sentence_length = dataset->get_max_sentence_length();
	double g0 = 1.0 / (double)dataset->_dict->get_num_characters();
	double initial_lambda_a = 4;
	double initial_lambda_b = 1;
	double vpylm_beta_stop = 4;
	double vpylm_beta_pass = 1;
	python::model::NPYLM* py_npylm = new python::model::NPYLM(max_word_length, max_sentence_length, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);

	int num_character_ids = dataset->_dict->get_num_characters();
	int num_character_types = 281;
	int feature_x_unigram_start = -2;
	int feature_x_unigram_end = 2;
	int feature_x_bigram_start = -2;
	int feature_x_bigram_end = 1;
	int feature_x_identical_1_start = -2;
	int feature_x_identical_1_end = 1;
	int feature_x_identical_2_start = -3;
	int feature_x_identical_2_end = 1;
	python::model::CRF* py_crf = new python::model::CRF(num_character_ids,
														num_character_types,
														feature_x_unigram_start,
														feature_x_unigram_end,
														feature_x_bigram_start,
														feature_x_bigram_end,
														feature_x_identical_1_start,
														feature_x_identical_1_end,
														feature_x_identical_2_start,
														feature_x_identical_2_end);

	Model* model = new Model(py_npylm, py_crf, lambda_0, max_word_length, dataset->get_max_sentence_length());
	Dictionary* dictionary = dataset->_dict;

	Trainer* trainer = new Trainer(dataset, model, false);
	Lattice* lattice = model->_lattice;
	npylm::NPYLM* npylm = model->_npylm;

	for(int epoch = 0;epoch < 2;epoch++){
		trainer->gibbs();
		for(Sentence* sentence: dataset->_sentence_sequences_train){
			double prob_n = compute_forward_probability(lattice, sentence, true);
			double prob_u = compute_forward_probability(lattice, sentence, false);
			assert(std::abs(prob_n - prob_u) < 1e-16);
		}
	}
}

void test_compute_backward_probability(){
	std::string filename = "../../../dataset/test.txt";
	Corpus* corpus = new Corpus();
	corpus->add_textfile(filename);
	int seed = 0;
	Dataset* dataset = new Dataset(corpus, 1, seed);

	double lambda_0 = 1;
	int max_word_length = 12;
	int max_sentence_length = dataset->get_max_sentence_length();
	double g0 = 1.0 / (double)dataset->_dict->get_num_characters();
	double initial_lambda_a = 4;
	double initial_lambda_b = 1;
	double vpylm_beta_stop = 4;
	double vpylm_beta_pass = 1;
	python::model::NPYLM* py_npylm = new python::model::NPYLM(max_word_length, max_sentence_length, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);

	int num_character_ids = dataset->_dict->get_num_characters();
	int num_character_types = 281;
	int feature_x_unigram_start = -2;
	int feature_x_unigram_end = 2;
	int feature_x_bigram_start = -2;
	int feature_x_bigram_end = 1;
	int feature_x_identical_1_start = -2;
	int feature_x_identical_1_end = 1;
	int feature_x_identical_2_start = -3;
	int feature_x_identical_2_end = 1;
	python::model::CRF* py_crf = new python::model::CRF(num_character_ids,
														num_character_types,
														feature_x_unigram_start,
														feature_x_unigram_end,
														feature_x_bigram_start,
														feature_x_bigram_end,
														feature_x_identical_1_start,
														feature_x_identical_1_end,
														feature_x_identical_2_start,
														feature_x_identical_2_end);

	Model* model = new Model(py_npylm, py_crf, lambda_0, max_word_length, dataset->get_max_sentence_length());
	Dictionary* dictionary = dataset->_dict;

	Trainer* trainer = new Trainer(dataset, model, false);
	Lattice* lattice = model->_lattice;
	npylm::NPYLM* npylm = model->_npylm;

	for(int epoch = 0;epoch < 2;epoch++){
		trainer->gibbs();
		for(Sentence* sentence: dataset->_sentence_sequences_train){
			double prob_n = compute_backward_probability(lattice, sentence, true);
			double prob_u = compute_backward_probability(lattice, sentence, false);
			assert(std::abs(prob_n - prob_u) < 1e-16);
		}
	}
}

// void test_compute_z_x(){
// 	std::string filename = "../../../dataset/test.txt";
// 	Corpus* corpus = new Corpus();
// 	corpus->add_textfile(filename);
// 	int seed = 0;
// 	Dataset* dataset = new Dataset(corpus, 1, seed);

// 	double lambda_0 = 1;
// 	int max_word_length = 12;
// 	int max_sentence_length = dataset->get_max_sentence_length();
// 	double g0 = 1.0 / (double)dataset->_dict->get_num_characters();
// 	double initial_lambda_a = 4;
// 	double initial_lambda_b = 1;
// 	double vpylm_beta_stop = 4;
// 	double vpylm_beta_pass = 1;
// 	python::model::NPYLM* py_npylm = new python::model::NPYLM(max_word_length, max_sentence_length, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);

// 	int num_character_ids = dataset->_dict->get_num_characters();
// 	int num_character_types = 281;
// 	int feature_x_unigram_start = -2;
// 	int feature_x_unigram_end = 2;
// 	int feature_x_bigram_start = -2;
// 	int feature_x_bigram_end = 1;
// 	int feature_x_identical_1_start = -2;
// 	int feature_x_identical_1_end = 1;
// 	int feature_x_identical_2_start = -3;
// 	int feature_x_identical_2_end = 1;
// 	python::model::CRF* py_crf = new python::model::CRF(num_character_ids,
// 														num_character_types,
// 														feature_x_unigram_start,
// 														feature_x_unigram_end,
// 														feature_x_bigram_start,
// 														feature_x_bigram_end,
// 														feature_x_identical_1_start,
// 														feature_x_identical_1_end,
// 														feature_x_identical_2_start,
// 														feature_x_identical_2_end);

// 	Model* model = new Model(py_npylm, py_crf, lambda_0, max_word_length, dataset->get_max_sentence_length());
// 	Dictionary* dictionary = dataset->_dict;

// 	Trainer* trainer = new Trainer(dataset, model, false);
// 	Lattice* lattice = model->_lattice;
// 	npylm::NPYLM* npylm = model->_npylm;

// 	for(int epoch = 0;epoch < 2;epoch++){
// 		trainer->gibbs();
// 		for(Sentence* sentence: dataset->_sentence_sequences_train){
// 			double prob_f = lattice->compute_z_x(sentence, false);
// 			double prob_b = lattice->compute_z_x_backward(sentence, false);
// 			assert(std::abs(prob_f - prob_b) / prob_b < 1e-14);
// 			// lattice->compute_forward_probability(sentence, false);
// 			// lattice->compute_backward_probability(sentence, false);
// 			// cout << sentence->size() << endl;
// 			// for(int t = 0;t <= sentence->size();t++){
// 			// 	double prob_sum = 0;
// 			// 	for(int k = 1;k <= std::min(sentence->size() - t, max_word_length);k++){
// 			// 		int _t = k + t;
// 			// 		if(_t - k == 0){
// 			// 			cout << "[" << _t << "][" << k << "][" << 0 << "]" << endl;
// 			// 			prob_sum += lattice->_alpha[_t][k][0] * lattice->_beta[_t][k][0];
// 			// 		}
// 			// 		for(int j = 1;j <= std::min(t, max_word_length);j++){
// 			// 			cout << "[" << _t << "][" << k << "][" << j << "]" << endl;
// 			// 			prob_sum += lattice->_alpha[_t][k][j] * lattice->_beta[_t][k][j];
// 			// 		}
// 			// 	}
// 			// 	// assert(std::abs(prob_t - prob_sum) / prob_sum < 1e-16);
// 			// 	cout << "t = " << t << ", " << prob_f << " == " << prob_sum << ", " << (std::abs(prob_f - prob_sum) / prob_sum) << endl;
// 			// }
// 		}
// 	}
// }

void test_sgd(){
	std::string filename = "../../../dataset/test.txt";
	Corpus* corpus = new Corpus();
	corpus->add_textfile(filename);
	int seed = 0;
	Dataset* dataset = new Dataset(corpus, 1, seed);

	double lambda_0 = 1;
	int max_word_length = 12;
	int max_sentence_length = dataset->get_max_sentence_length();
	double g0 = 1.0 / (double)dataset->_dict->get_num_characters();
	double initial_lambda_a = 4;
	double initial_lambda_b = 1;
	double vpylm_beta_stop = 4;
	double vpylm_beta_pass = 1;
	python::model::NPYLM* py_npylm = new python::model::NPYLM(max_word_length, max_sentence_length, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);

	int num_character_ids = dataset->_dict->get_num_characters();
	int num_character_types = 281;
	int feature_x_unigram_start = -2;
	int feature_x_unigram_end = 2;
	int feature_x_bigram_start = -2;
	int feature_x_bigram_end = 1;
	int feature_x_identical_1_start = -2;
	int feature_x_identical_1_end = 1;
	int feature_x_identical_2_start = -3;
	int feature_x_identical_2_end = 1;
	python::model::CRF* py_crf = new python::model::CRF(num_character_ids,
														num_character_types,
														feature_x_unigram_start,
														feature_x_unigram_end,
														feature_x_bigram_start,
														feature_x_bigram_end,
														feature_x_identical_1_start,
														feature_x_identical_1_end,
														feature_x_identical_2_start,
														feature_x_identical_2_end);

	Model* model = new Model(py_npylm, py_crf, lambda_0, max_word_length, dataset->get_max_sentence_length());
	Dictionary* dictionary = dataset->_dict;

	Trainer* trainer = new Trainer(dataset, model, false);
	Lattice* lattice = model->_lattice;
	crf::CRF* crf = model->_crf;
	npylm::NPYLM* npylm = model->_npylm;

	for(int epoch = 0;epoch < 5;epoch++){
		trainer->gibbs();
	}
	for(int i = 0;i < crf->_w_size_label_u + crf->_w_size_label_b;i++){
		crf->_w_label[i] = 1;
	}
	for(int i = 0;i < crf->_w_size_unigram_u + crf->_w_size_unigram_b;i++){
		crf->_w_unigram[i] = 1;
	}
	for(int i = 0;i < crf->_w_size_bigram_u + crf->_w_size_bigram_b;i++){
		crf->_w_bigram[i] = 1;
	}
	for(int i = 0;i < crf->_w_size_identical_1_u + crf->_w_size_identical_1_b;i++){
		crf->_w_identical_1[i] = 1;
	}
	for(int i = 0;i < crf->_w_size_identical_2_u + crf->_w_size_identical_2_b;i++){
		crf->_w_identical_2[i] = 1;
	}
	for(int i = 0;i < crf->_w_size_unigram_type_u + crf->_w_size_unigram_type_b;i++){
		crf->_w_unigram_type[i] = 1;
	}
	trainer->sgd(false, 1);
}

int main(int argc, char *argv[]){
	test_sgd();
	cout << "OK" << endl;
	test_compute_backward_probability();
	cout << "OK" << endl;
	// test_compute_z_x();
	// cout << "OK" << endl;
	test_compute_forward_probability();
	cout << "OK" << endl;
}
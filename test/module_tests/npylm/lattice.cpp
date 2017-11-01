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

	for(int epoch = 0;epoch < 20;epoch++){
		trainer->gibbs();
		for(Sentence* sentence: dataset->_sentence_sequences_train){
			double prob_n = compute_forward_probability(lattice, sentence, true);
			double prob_u = compute_forward_probability(lattice, sentence, false);
			assert(std::abs(prob_n - prob_u) < 1e-16);
		}
	}
}

int main(int argc, char *argv[]){
	test_compute_forward_probability();
	cout << "OK" << endl;
}
#include  <iostream>
#include <chrono>
#include "../../src/npycrf/sampler.h"
#include "../../src/python/model.h"
#include "../../src/python/dataset.h"
#include "../../src/python/dictionary.h"
#include "../../src/python/trainer.h"
using namespace npycrf;
using namespace npycrf::python;
using std::cout;
using std::flush;
using std::endl;

int main(int argc, char *argv[]){
	std::string filename = "../../dataset/test.txt";
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
	dictionary->save("npylm.dict");

	double likelihood_1 = model->python_compute_marginal_p_x(L"こんにちは", dictionary);
	cout << likelihood_1 << endl;
	crf::CRF* crf = py_crf->_crf;
	for(int i = 0;i < crf->_w_size_unigram_type_u;i++){
		crf->_w_unigram_type[i] = 1;
	}
	double likelihood_2 = model->python_compute_marginal_p_x(L"こんにちは", dictionary);
	cout << likelihood_2 << endl;
}
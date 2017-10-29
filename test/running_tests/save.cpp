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

template<typename T>
void compare_node(npylm::lm::Node<T>* a, npylm::lm::Node<T>* b){
	assert(a->_num_tables == b->_num_tables);
	assert(a->_num_customers == b->_num_customers);
	assert(a->_stop_count == b->_stop_count);
	assert(a->_pass_count == b->_pass_count);
	assert(a->_depth == b->_depth);
	assert(a->_token_id == b->_token_id);
	assert(a->_arrangement.size() == b->_arrangement.size());
	for(auto elem: a->_arrangement){
		T key = elem.first;
		std::vector<int> &table_a = elem.second;
		std::vector<int> &table_b = b->_arrangement[key];
		assert(table_a.size() == table_b.size());
	}
	for(auto elem: a->_children){
		T key = elem.first;
		npylm::lm::Node<T>* children_a = elem.second;
		npylm::lm::Node<T>* children_b = b->_children[key];
		compare_node(children_a, children_b);
	}
}

void compare_npylm(npylm::NPYLM* a, npylm::NPYLM* b){
	assert(a != NULL);
	assert(b != NULL);
	compare_node(a->_hpylm->_root, b->_hpylm->_root);
	compare_node(a->_vpylm->_root, b->_vpylm->_root);
}

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
	Trainer* trainer = new Trainer(dataset, model, false);

	for(int epoch = 0;epoch < 1000;epoch++){
		cout << "\r" << epoch << flush;
		trainer->gibbs();
		trainer->sample_hpylm_vpylm_hyperparameters();
		trainer->sample_lambda();
		py_npylm->save("npylm.model");
		py_crf->save("crf.model");
		python::model::NPYLM* _npylm = new python::model::NPYLM("npylm.model");
		python::model::CRF* _crf = new python::model::CRF("crf.model");
		Model* _model = new Model(_npylm, _crf, lambda_0, max_word_length, dataset->get_max_sentence_length());
		compare_npylm(model->_npylm, _model->_npylm);
		delete _model;
	}
}
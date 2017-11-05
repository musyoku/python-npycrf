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

void npylm_add_customers(npylm::NPYLM* npylm, Sentence* sentence){
	for(int t = 2;t < sentence->get_num_segments();t++){
		npylm->add_customer_at_time_t(sentence, t);
	}
}

double compute_forward_probability(Lattice* lattice, Sentence* sentence, bool normalize){
	assert(sentence->size() <= lattice->_max_sentence_length);
	int size = sentence->size() + 1;
	lattice->_alpha[0][0][0] = 1;
	lattice->_log_z_alpha[0] = 0;
	for(int i = 0;i < size;i++){
		for(int j = 0;j < lattice->_max_word_length + 1;j++){
			lattice->_substring_word_id_cache[i][j] = 0;
		}
	}
	for(int t = 0;t < size;t++){
		lattice->_log_z_alpha[t] = 0;
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
				sum_probability += lattice->_alpha[t][k][j] * exp(lattice->_log_z_alpha[t]);
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
		lattice->_log_z_beta[0] = 0;
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
				lattice->_log_z_beta[t] = 0;
			}
		#endif 
		lattice->_enumerate_backward_probabilities(sentence, lattice->_beta, lattice->_pw_h, lattice->_log_z_beta, normalize);
		double sum_probability = 0;
		for(int k = 1;k <= std::min(sentence->size(), lattice->_max_word_length);k++){
			int t = k;
			if(normalize){
				assert(lattice->_log_z_beta[t] < 0);
				sum_probability += lattice->_beta[t][k][0] * exp(lattice->_log_z_beta[t]);
			}else{
				sum_probability += lattice->_beta[t][k][0];
			}
		}
		return sum_probability;
}

void test_compute_forward_probability(){
	std::unordered_map<wchar_t, int> _token_ids;
	std::wstring sentence_str = L"ああいいいううううえええおお";
	for(wchar_t character: sentence_str){
		auto itr = _token_ids.find(character);
		if(itr == _token_ids.end()){
			_token_ids[character] = _token_ids.size();
		}
	}
	int* character_ids = new int[sentence_str.size()];
	for(int i = 0;i < sentence_str.size();i++){
		character_ids[i] = _token_ids[sentence_str[i]];
	}
	Sentence* sentence = new Sentence(sentence_str, character_ids);
	std::vector<int> segments {2, 3, 4, 3, 2};
	sentence->split(segments);

	double lambda_0 = 1;
	int max_word_length = 4;
	int max_sentence_length = sentence->size();
	double g0 = 1.0 / (double)_token_ids.size();
	double initial_lambda_a = 4;
	double initial_lambda_b = 1;
	double vpylm_beta_stop = 4;
	double vpylm_beta_pass = 1;
	python::model::NPYLM* py_npylm = new python::model::NPYLM(max_word_length, max_sentence_length, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);

	int num_character_ids = _token_ids.size();
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

	Model* model = new Model(py_npylm, py_crf, lambda_0, max_word_length, sentence->size());
	Lattice* lattice = model->_lattice;
	npylm::NPYLM* npylm = model->_npylm;

	double prob_n = compute_forward_probability(lattice, sentence, true);
	double prob_u = compute_forward_probability(lattice, sentence, false);
	assert(std::abs(prob_n - prob_u) < 1e-16);

	delete[] character_ids;
	delete sentence;
	delete model;
	delete py_npylm;
	delete py_crf;
}

void test_compute_backward_probability(){
	std::unordered_map<wchar_t, int> _token_ids;
	std::wstring sentence_str = L"ああいいいううううえええおおああいいいううううえええおおああいいいううううえええおおああいいいううううえええおおああいいいううううえええおお";
	for(wchar_t character: sentence_str){
		auto itr = _token_ids.find(character);
		if(itr == _token_ids.end()){
			_token_ids[character] = _token_ids.size();
		}
	}
	int* character_ids = new int[sentence_str.size()];
	for(int i = 0;i < sentence_str.size();i++){
		character_ids[i] = _token_ids[sentence_str[i]];
	}
	Sentence* sentence = new Sentence(sentence_str, character_ids);
	std::vector<int> segments {2, 3, 4, 3, 2, 2, 3, 4, 3, 2, 2, 3, 4, 3, 2, 2, 3, 4, 3, 2, 2, 3, 4, 3, 2};
	sentence->split(segments);

	double lambda_0 = 1;
	int max_word_length = 4;
	int max_sentence_length = sentence->size();
	double g0 = 1.0 / (double)_token_ids.size();
	double initial_lambda_a = 4;
	double initial_lambda_b = 1;
	double vpylm_beta_stop = 4;
	double vpylm_beta_pass = 1;
	python::model::NPYLM* py_npylm = new python::model::NPYLM(max_word_length, max_sentence_length, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);

	int num_character_ids = _token_ids.size();
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

	Model* model = new Model(py_npylm, py_crf, lambda_0, max_word_length, sentence->size());
	Lattice* lattice = model->_lattice;
	npylm::NPYLM* npylm = model->_npylm;

	double prob_n = compute_backward_probability(lattice, sentence, true);
	double prob_u = compute_backward_probability(lattice, sentence, false);
	assert(std::abs(prob_n - prob_u) < 1e-16);

	delete[] character_ids;
	delete sentence;
	delete model;
	delete py_npylm;
	delete py_crf;
}

void test_enumerate_proportional_log_p_substring_given_sentence(){
	std::unordered_map<wchar_t, int> _token_ids;
	std::wstring sentence_str = L"ああいいいううううえええおおああいいいううううえええおおああいいいううううえええおおああいいいううううえええおおああいいいううううえええおお";
	for(wchar_t character: sentence_str){
		auto itr = _token_ids.find(character);
		if(itr == _token_ids.end()){
			_token_ids[character] = _token_ids.size();
		}
	}
	int* character_ids = new int[sentence_str.size()];
	for(int i = 0;i < sentence_str.size();i++){
		character_ids[i] = _token_ids[sentence_str[i]];
	}
	Sentence* sentence = new Sentence(sentence_str, character_ids);
	std::vector<int> segments {2, 3, 4, 3, 2, 2, 3, 4, 3, 2, 2, 3, 4, 3, 2, 2, 3, 4, 3, 2, 2, 3, 4, 3, 2};
	sentence->split(segments);

	double lambda_0 = 1;
	int max_word_length = 4;
	int max_sentence_length = sentence->size();
	double g0 = 1.0 / (double)_token_ids.size();
	double initial_lambda_a = 4;
	double initial_lambda_b = 1;
	double vpylm_beta_stop = 4;
	double vpylm_beta_pass = 1;
	python::model::NPYLM* py_npylm = new python::model::NPYLM(max_word_length, max_sentence_length, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);

	int num_character_ids = _token_ids.size();
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

	Model* model = new Model(py_npylm, py_crf, lambda_0, max_word_length, sentence->size());
	Lattice* lattice = model->_lattice;
	npylm::NPYLM* npylm = model->_npylm;

	double** pc_s = new double*[sentence->size() + 1];
	double** log_pc_s = new double*[sentence->size() + 1];
	for(int t = 0;t < sentence->size() + 1;t++){
		pc_s[t] = new double[max_word_length + 1];
		log_pc_s[t] = new double[max_word_length + 1];
		for(int k = 0;k < max_word_length + 1;k++){
			pc_s[t][k] = -1;
			log_pc_s[t][k] = 0;
		}
	}
	lattice->_enumerate_forward_probabilities(sentence, lattice->_alpha, lattice->_pw_h, lattice->_log_z_alpha, false);
	lattice->_enumerate_backward_probabilities(sentence, lattice->_beta, lattice->_pw_h, lattice->_log_z_beta, false);
	lattice->_enumerate_proportional_p_substring_given_sentence(sentence, lattice->_alpha, lattice->_beta, pc_s);

	lattice->_enumerate_forward_probabilities(sentence, lattice->_alpha, lattice->_pw_h, lattice->_log_z_alpha, true);
	lattice->_enumerate_backward_probabilities(sentence, lattice->_beta, lattice->_pw_h, lattice->_log_z_beta, true);
	lattice->_enumerate_proportional_log_p_substring_given_sentence(sentence, lattice->_alpha, lattice->_beta, lattice->_log_z_alpha, lattice->_log_z_beta, log_pc_s);

	for(int t = 1;t <= sentence->size();t++){
		for(int k = 1;k <= std::min(t, max_word_length);k++){
			cout << pc_s[t][k] << ", " << log(pc_s[t][k]) << "==" << log_pc_s[t][k] << endl;
			assert(abs(log(pc_s[t][k]) - log_pc_s[t][k]) < 1e-12);
		}
	}
	delete[] character_ids;
	delete sentence;
	delete model;
	delete py_npylm;
	delete py_crf;
}

int main(int argc, char *argv[]){
	setlocale(LC_CTYPE, "ja_JP.UTF-8");
	std::ios_base::sync_with_stdio(false);
	std::locale default_loc("ja_JP.UTF-8");
	std::locale::global(default_loc);
	std::locale ctype_default(std::locale::classic(), default_loc, std::locale::ctype); //※
	std::wcout.imbue(ctype_default);
	std::wcin.imbue(ctype_default);
	test_compute_forward_probability();
	cout << "OK" << endl;
	test_compute_backward_probability();
	cout << "OK" << endl;
	test_enumerate_proportional_log_p_substring_given_sentence();
	cout << "OK" << endl;
}
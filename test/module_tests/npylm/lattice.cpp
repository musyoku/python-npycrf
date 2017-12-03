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

std::unordered_map<wchar_t, int> token_ids;

Sentence* generate_sentence(std::wstring &sentence_str, std::vector<int> &segments){
	for(wchar_t character: sentence_str){
		auto itr = token_ids.find(character);
		if(itr == token_ids.end()){
			token_ids[character] = token_ids.size();
		}
	}
	int* character_ids = new int[sentence_str.size()];
	for(int i = 0;i < sentence_str.size();i++){
		character_ids[i] = token_ids[sentence_str[i]];
	}
	Sentence* sentence = new Sentence(sentence_str, character_ids);
	sentence->split(segments);
	return sentence;
}

Sentence* generate_sentence_1(){
	std::vector<int> segments {4, 3, 2, 2, 3, 4, 3, 2, 2, 3, 4, 3, 2, 2, 3, 4, 3, 2, 2, 3, 4, 3, 2, 2, 3};
	std::wstring sentence_str = L"ううううえええおおああいいいううううえええおおああいいいううううえええおおああいいいううううえええおおああいいいううううえええおおああいいい";
	return generate_sentence(sentence_str, segments);
}

Sentence* generate_sentence_2(){
	std::vector<int> segments {3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3};
	std::wstring sentence_str = L"あああいいうううええおおおあああいいうううええおおおあああいいうううええおおおあああいいうううええおおおあああいいうううええおおお";
	return generate_sentence(sentence_str, segments);
}

Sentence* generate_sentence_3(){
	std::vector<int> segments {4, 1, 2, 1, 1, 4, 1, 2, 1, 1, 4, 1, 2, 1, 1, 4, 1, 2, 1, 1, 4, 1, 2, 1, 1};
	std::wstring sentence_str = L"ああああいううえおああああいううえおああああいううえおああああいううえおああああいううえお";
	return generate_sentence(sentence_str, segments);
}

Sentence* generate_sentence_4(){
	std::vector<int> segments {1, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1, 4, 1, 4, 4};
	std::wstring sentence_str = L"あいいいいうええええおおおおあいいいいうええええおおおおあいいいいうええええおおおおあいいいいうええええおおおおあいいいいうええええおおおお";
	return generate_sentence(sentence_str, segments);
}

class Variables {
public:
	Model* model;
	python::model::NPYLM* py_npylm;
	python::model::CRF* py_crf;
	Variables(){
		double lambda_0 = 1;
		int max_word_length = 4;
		int max_sentence_length = 100;
		double g0 = 0.001;
		double initial_lambda_a = 4;
		double initial_lambda_b = 1;
		double vpylm_beta_stop = 4;
		double vpylm_beta_pass = 1;
		py_npylm = new python::model::NPYLM(max_word_length, max_sentence_length, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);

		int num_character_ids = 5;
		int num_character_types = 281;
		int feature_x_unigram_start = -2;
		int feature_x_unigram_end = 2;
		int feature_x_bigram_start = -2;
		int feature_x_bigram_end = 1;
		int feature_x_identical_1_start = -2;
		int feature_x_identical_1_end = 1;
		int feature_x_identical_2_start = -3;
		int feature_x_identical_2_end = 1;
		py_crf = new python::model::CRF(num_character_ids, num_character_types, feature_x_unigram_start, feature_x_unigram_end, feature_x_bigram_start, feature_x_bigram_end, feature_x_identical_1_start, feature_x_identical_1_end, feature_x_identical_2_start, feature_x_identical_2_end);

		model = new Model(py_npylm, py_crf, lambda_0, max_word_length, 100);
		Lattice* lattice = model->_lattice;
		npylm::NPYLM* npylm = model->_npylm;
		lattice->reserve(max_word_length, 100);

		Sentence* sentence = generate_sentence_1();
		npylm_add_customers(npylm, sentence);
		delete sentence;

		sentence = generate_sentence_2();
		npylm_add_customers(npylm, sentence);
		delete sentence;

		sentence = generate_sentence_3();
		npylm_add_customers(npylm, sentence);
		delete sentence;

	}
	~Variables(){
		delete model;
		delete py_npylm;
		delete py_crf;
	}
};

void test_compute_marginal_p_x(){
	Variables* var = new Variables();
	Lattice* lattice = var->model->_lattice;
	Sentence* sentence = generate_sentence_1();

	double px_u = lattice->compute_marginal_p_x(sentence, true);
	double px_n = lattice->compute_marginal_p_x(sentence, false);
	double px_b = lattice->_compute_marginal_p_x_backward(sentence, lattice->_beta, lattice->_pw_h);
	assert(std::abs(px_u - px_n) < 1e-12);
	assert(std::abs(px_u - px_b) < 1e-12);

	delete sentence;
	delete var;
}

void test_viterbi_decode(){
	Variables* var = new Variables();
	Lattice* lattice = var->model->_lattice;
	Sentence* sentence = generate_sentence_1();
	std::vector<int> segments;
	lattice->viterbi_decode(sentence, segments);
	for(int i = 0;i < segments.size();i++){
		assert(segments[i] == sentence->_segments[i + 2]);
	}
	delete sentence;
	delete var;
}

void test_scaling(){
	Variables* var = new Variables();
	Lattice* lattice = var->model->_lattice;
	Sentence* sentence = generate_sentence_1();
	double*** alpha;
	double*** beta;
	int seq_capacity = lattice->_max_sentence_length + 1;
	int word_capacity = lattice->_max_word_length + 1;
	lattice::_init_array(alpha, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_init_array(beta, seq_capacity + 1, word_capacity, word_capacity);
	lattice->_clear_pw_h_tkji(lattice->_pw_h);
	lattice->_clear_word_id_cache(lattice->_substring_word_id_cache, sentence->size() + 1);
	lattice->_enumerate_forward_variables(sentence, alpha, lattice->_pw_h, NULL, false);
	lattice->_enumerate_backward_variables(sentence, beta, lattice->_pw_h, NULL, false);

	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);

	for(int t = 1;t <= sentence->size();t++){
		double scaling = 1;
		for(int m = 1;m <= t;m++){
			scaling /= lattice->_scaling[m];
		}
		for(int k = 1;k <= std::min(t, lattice->_max_word_length);k++){
			for(int j = (t - k == 0) ? 0 : 1;j <= std::min(t - k, lattice->_max_word_length);j++){
				assert(alpha[t][k][j] > 0);
				assert(beta[t][k][j] > 0);
				assert(lattice->_alpha[t][k][j] > 0);
				assert(lattice->_beta[t][k][j] > 0);
				assert(std::abs(alpha[t][k][j] - scaling * lattice->_alpha[t][k][j]) < 1e-12);
			}
		}
	}
	lattice::_delete_array(alpha, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_delete_array(beta, seq_capacity + 1, word_capacity, word_capacity);

	delete sentence;
	delete var;
}

void test_enumerate_proportional_p_substring_given_sentence(){
	Variables* var = new Variables();
	Lattice* lattice = var->model->_lattice;
	Sentence* sentence = generate_sentence_1();
	double*** alpha;
	double*** beta;
	int seq_capacity = lattice->_max_sentence_length + 1;
	int word_capacity = lattice->_max_word_length + 1;
	lattice->_clear_pw_h_tkji(lattice->_pw_h);
	lattice->_clear_word_id_cache(lattice->_substring_word_id_cache, sentence->size() + 1);
	lattice::_init_array(alpha, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_init_array(beta, seq_capacity + 1, word_capacity, word_capacity);
	lattice->_enumerate_forward_variables(sentence, alpha, lattice->_pw_h, NULL, false);
	lattice->_enumerate_backward_variables(sentence, beta, lattice->_pw_h, NULL, false);
	double Zs = lattice->compute_marginal_p_x(sentence, true);

	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	// double Zs = 1;
	// for(int m = 1;m <= sentence->size() + 1;m++){
	// 	Zs /= lattice->_scaling[m];
	// }

	for(int t = 1;t <= sentence->size();t++){
		for(int k = 1;k <= std::min(t, lattice->_max_word_length);k++){
			double sum_probability = 0;
			double _sum_probability = 0;
			for(int j = (t - k == 0) ? 0 : 1;j <= std::min(t - k, lattice->_max_word_length);j++){
				assert(alpha[t][k][j] > 0);
				assert(beta[t][k][j] > 0);
				assert(lattice->_alpha[t][k][j] > 0);
				assert(lattice->_beta[t][k][j] > 0);
				sum_probability += alpha[t][k][j] * beta[t][k][j];
				_sum_probability += lattice->_alpha[t][k][j] * lattice->_beta[t][k][j];
			}
			assert(sum_probability <= Zs);
			_sum_probability *= lattice->_scaling[sentence->size() + 1];
			sum_probability /= Zs;
			assert(std::abs(sum_probability - _sum_probability) < 1e-12);
		}
	}

	lattice::_delete_array(alpha, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_delete_array(beta, seq_capacity + 1, word_capacity, word_capacity);

	delete sentence;
	delete var;
}

void test_enumerate_marginal_p_path_given_sentence(){
	Variables* var = new Variables();
	Lattice* lattice = var->model->_lattice;
	Sentence* sentence = generate_sentence_4();
	double*** alpha;
	double*** beta;
	double** pc_s;
	double*** pz_s;
	int seq_capacity = lattice->_max_sentence_length + 1;
	int word_capacity = lattice->_max_word_length + 1;
	lattice->_clear_pw_h_tkji(lattice->_pw_h);
	lattice->_clear_word_id_cache(lattice->_substring_word_id_cache, sentence->size() + 1);
	lattice::_init_array(alpha, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_init_array(beta, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_init_array(pc_s, seq_capacity, word_capacity);
	lattice::_init_array(pz_s, seq_capacity + 1, 2, 2);

	lattice->_enumerate_forward_variables(sentence, alpha, lattice->_pw_h, NULL, false);
	lattice->_enumerate_backward_variables(sentence, beta, lattice->_pw_h, NULL, false);
	double Zs = lattice->compute_marginal_p_x(sentence, true);
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];

	lattice->_enumerate_proportional_p_substring_given_sentence(sentence->size(), alpha, beta, pc_s, Zs);
	lattice->_enumerate_proportional_p_substring_given_sentence(sentence->size(), lattice->_alpha, lattice->_beta, lattice->_pc_s, _Zs);

	for(int t = 1;t <= sentence->size();t++){
		for(int k = 1;k <= std::min(t, lattice->_max_word_length);k++){
			assert(pc_s[t][k] > 0);
			assert(lattice->_pc_s[t][k] > 0);
			assert(std::abs(lattice->_pc_s[t][k] - pc_s[t][k]) < 1e-12);
		}
	}

	lattice->_enumerate_marginal_p_path_given_sentence(sentence->size(), pz_s, pc_s);
	lattice->_enumerate_marginal_p_path_given_sentence(sentence->size(), lattice->_pz_s, lattice->_pc_s);

	for(int t = 1;t <= sentence->size();t++){
		cout << pz_s[t][0][0] << ", " << lattice->_pz_s[t][0][0] << endl;
	}

	lattice::_delete_array(alpha, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_delete_array(beta, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_delete_array(pc_s, seq_capacity, word_capacity);
	lattice::_delete_array(pz_s, seq_capacity + 1, 2, 2);

	delete sentence;
	delete var;
}

int main(int argc, char *argv[]){
	setlocale(LC_CTYPE, "ja_JP.UTF-8");
	std::ios_base::sync_with_stdio(false);
	std::locale default_loc("ja_JP.UTF-8");
	std::locale::global(default_loc);
	std::locale ctype_default(std::locale::classic(), default_loc, std::locale::ctype); //※
	std::wcout.imbue(ctype_default);
	std::wcin.imbue(ctype_default);
	test_compute_marginal_p_x();
	cout << "OK" << endl;
	test_viterbi_decode();
	cout << "OK" << endl;
	test_scaling();
	cout << "OK" << endl;
	test_enumerate_proportional_p_substring_given_sentence();
	cout << "OK" << endl;
	test_enumerate_marginal_p_path_given_sentence();
	cout << "OK" << endl;
}
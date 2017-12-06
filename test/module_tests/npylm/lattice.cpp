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
	std::vector<int> segments {3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 1};
	std::wstring sentence_str = L"あああいいうううええおおおあああいいうううええおおおあああいいうううええおおおあああいいうううええおおおあああいいうううええおおおう";
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

Sentence* generate_sentence_5(){
	std::vector<int> segments {1};
	std::wstring sentence_str = L"あ";
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

		int num_character_ids = 8;
		int num_character_types = 281;
		int feature_x_unigram_start = -2;
		int feature_x_unigram_end = 2;
		int feature_x_bigram_start = -2;
		int feature_x_bigram_end = 1;
		int feature_x_identical_1_start = -2;
		int feature_x_identical_1_end = 1;
		int feature_x_identical_2_start = -3;
		int feature_x_identical_2_end = 1;
		double sigma = 1.0;
		py_crf = new python::model::CRF(num_character_ids, num_character_types, feature_x_unigram_start, feature_x_unigram_end, feature_x_bigram_start, feature_x_bigram_end, feature_x_identical_1_start, feature_x_identical_1_end, feature_x_identical_2_start, feature_x_identical_2_end, sigma);

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

void assert_test_compute_normalizing_constant(Sentence* sentence, Lattice* lattice, Model* model){
	double zs_u = lattice->compute_normalizing_constant(sentence, true);
	double zs_n = lattice->compute_normalizing_constant(sentence, false);
	double zs_b = lattice->_compute_normalizing_constant_backward(sentence, lattice->_beta, lattice->_pw_h);
	double propotional_log_py_x = model->compute_log_proportional_p_y_given_sentence(sentence);
	double log_py_x_u = propotional_log_py_x - log(zs_u);
	double log_py_x_n = propotional_log_py_x - log(zs_n);
	double log_py_x_b = propotional_log_py_x - log(zs_b);
	assert(std::abs(log_py_x_u - log_py_x_n) < 1e-12);
	assert(std::abs(log_py_x_u - log_py_x_b) < 1e-12);
}

void test_compute_normalizing_constant(){
	Variables* var = new Variables();
	Model* model = var->model;
	Lattice* lattice = model->_lattice;

	Sentence* sentence = generate_sentence_1();
	assert_test_compute_normalizing_constant(sentence, lattice, model);
	delete sentence;

	sentence = generate_sentence_2();
	assert_test_compute_normalizing_constant(sentence, lattice, model);
	delete sentence;

	sentence = generate_sentence_5();
	double log_Zs = log(model->compute_normalizing_constant(sentence));
	double log_py = model->compute_log_proportional_p_y_given_sentence(sentence);
	assert(std::abs(log_Zs - log_py) < 1e-12);

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
				if(std::abs(1 - alpha[t][k][j] / (scaling * lattice->_alpha[t][k][j])) >= 1e-12){
					cout << alpha[t][k][j] << ", " << scaling * lattice->_alpha[t][k][j] << endl;
				}
				assert(std::abs(1 - alpha[t][k][j] / (scaling * lattice->_alpha[t][k][j])) < 1e-12);
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
	double Zs = lattice->compute_normalizing_constant(sentence, true);

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
	double Zs = lattice->compute_normalizing_constant(sentence, true);
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];

	lattice->_enumerate_proportional_p_substring_given_sentence(pc_s, sentence->size(), alpha, beta, Zs);
	lattice->_enumerate_proportional_p_substring_given_sentence(lattice->_pc_s, sentence->size(), lattice->_alpha, lattice->_beta, _Zs);

	for(int t = 1;t <= sentence->size();t++){
		for(int k = 1;k <= std::min(t, lattice->_max_word_length);k++){
			assert(pc_s[t][k] > 0);
			assert(lattice->_pc_s[t][k] > 0);
			assert(std::abs(lattice->_pc_s[t][k] - pc_s[t][k]) < 1e-12);
		}
	}

	lattice->_enumerate_marginal_p_path_given_sentence(pz_s, sentence->size(), pc_s);
	lattice->_enumerate_marginal_p_path_given_sentence(lattice->_pz_s, sentence->size(), lattice->_pc_s);

	for(int t = 1;t <= sentence->size();t++){
		assert(std::abs(pz_s[t][0][0] - lattice->_pz_s[t][0][0]) < 1e-12);
		assert(std::abs(pz_s[t][0][1] - lattice->_pz_s[t][0][1]) < 1e-12);
		assert(std::abs(pz_s[t][1][0] - lattice->_pz_s[t][1][0]) < 1e-12);
		assert(std::abs(pz_s[t][1][1] - lattice->_pz_s[t][1][1]) < 1e-12);
	}

	lattice::_delete_array(alpha, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_delete_array(beta, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_delete_array(pc_s, seq_capacity, word_capacity);
	lattice::_delete_array(pz_s, seq_capacity + 1, 2, 2);

	delete sentence;
	delete var;
}

void test_grad_unigram(){
	Variables* var = new Variables();
	Model* model = var->model;
	Lattice* lattice = model->_lattice;
	Sentence* sentence = generate_sentence_4();
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];
	lattice->_enumerate_proportional_p_substring_given_sentence(lattice->_pc_s, sentence->size(), lattice->_alpha, lattice->_beta, _Zs);
	lattice->_enumerate_marginal_p_path_given_sentence(lattice->_pz_s, sentence->size(), lattice->_pc_s);

	crf::CRF* crf = var->py_crf->_crf;
	int const* character_ids = sentence->_character_ids;
	wchar_t const* characters = sentence->_characters;
	int character_ids_length = sentence->size();

	for(int pos = 1;pos <= crf->_x_range_unigram;pos++){
		for(int x_i = 0;x_i < token_ids.size();x_i++){
			// cout << "index = " << crf->_index_w_unigram_u(0, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
			// cout << "index = " << crf->_index_w_unigram_u(1, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
			assert(pos == (crf->_index_w_unigram_u(0, pos, x_i) % (crf->_x_range_unigram * 2)) / 2 + 1);
			assert(pos == (crf->_index_w_unigram_u(1, pos, x_i) % (crf->_x_range_unigram * 2)) / 2 + 1);
		}
	}

	// sentence->dump_words();

	for(int k = 0;k < crf->_w_size_unigram_u;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int pos = (k % (crf->_x_range_unigram * 2)) / 2 + 1;
		int t_start = std::max(1, -(crf->_x_unigram_start + pos - 1) + 1);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_unigram_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = 1;t < t_start;t++){
			int s = (i < sentence->_num_segments - 1) ? sentence->_start[i] + 1 : sentence->size() + 1;
			yt_1 = yt;
			yt = 0;
			if(t == s){
				i = (i < sentence->_num_segments - 1) ? i + 1 : sentence->_num_segments - 1;
				yt = 1;
			}
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
		}
		for(int t = t_start;t <= t_end;t++){
			int s = (i < sentence->_num_segments - 1) ? sentence->_start[i] + 1 : sentence->size() + 1;
			yt_1 = yt;
			yt = 0;
			if(t == s || t == sentence->size() + 2){
				i = (i < sentence->_num_segments - 1) ? i + 1 : sentence->_num_segments - 1;
				yt = 1;
			}
			int r = crf->_x_unigram_start + (pos - 1);
			int index = t + r - 1;
			assert(0 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : CHARACTER_ID_EOS;
			double pi_k = (k == crf->_index_w_unigram_u(yt, pos, x_i)) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_unigram_u(1, pos, x_i) == k) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_unigram_u(0, pos, x_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_unigram_u(1, pos, x_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_unigram_u(0, pos, x_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_unigram_u(1, pos, x_i) == k) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", r = " << r << ", index = " << index << ", x_i = " << x_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		if(k > 0){
			crf->_w_unigram[k] -= 1e-8;
		}
		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_w_unigram[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		if(std::abs(true_grad - grad) >= 1e-4){
			cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		}
		assert(std::abs(true_grad - grad) < 1e-4);
	}
	crf->_w_unigram[crf->_w_size_unigram_u - 1] -= 1e-8;

	for(int pos = 1;pos <= crf->_x_range_unigram;pos++){
		for(int x_i = 0;x_i < token_ids.size();x_i++){
			// cout << "index = " << crf->_index_w_unigram_b(0, 0, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
			// cout << "index = " << crf->_index_w_unigram_b(1, 0, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
			// cout << "index = " << crf->_index_w_unigram_b(0, 1, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
			// cout << "index = " << crf->_index_w_unigram_b(1, 1, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;

			assert(pos == ((crf->_index_w_unigram_b(0, 0, pos, x_i) - crf->_w_size_unigram_u) % (crf->_x_range_unigram * 2 * 2)) / 4 + 1);
			assert(pos == ((crf->_index_w_unigram_b(1, 0, pos, x_i) - crf->_w_size_unigram_u) % (crf->_x_range_unigram * 2 * 2)) / 4 + 1);
			assert(pos == ((crf->_index_w_unigram_b(0, 1, pos, x_i) - crf->_w_size_unigram_u) % (crf->_x_range_unigram * 2 * 2)) / 4 + 1);
			assert(pos == ((crf->_index_w_unigram_b(1, 1, pos, x_i) - crf->_w_size_unigram_u) % (crf->_x_range_unigram * 2 * 2)) / 4 + 1);
		}
	}

	for(int k = crf->_w_size_unigram_u;k < crf->_w_size_unigram_u + crf->_w_size_unigram_b;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int pos = ((k - crf->_w_size_unigram_u) % (crf->_x_range_unigram * 2 * 2)) / 4 + 1;
		int t_start = std::max(1, -(crf->_x_unigram_start + pos - 1) + 1);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_unigram_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = 1;t < t_start;t++){
			int s = (i < sentence->_num_segments - 1) ? sentence->_start[i] + 1 : sentence->size() + 1;
			yt_1 = yt;
			yt = 0;
			if(t == s){
				i = (i < sentence->_num_segments - 1) ? i + 1 : sentence->_num_segments - 1;
				yt = 1;
			}
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
		}
		for(int t = t_start;t <= t_end;t++){
			int s = (i < sentence->_num_segments - 1) ? sentence->_start[i] + 1 : sentence->size() + 1;
			yt_1 = yt;
			yt = 0;
			if(t == s || t == sentence->size() + 2){
				i = (i < sentence->_num_segments - 1) ? i + 1 : sentence->_num_segments - 1;
				yt = 1;
			}
			int r = crf->_x_unigram_start + (pos - 1);
			int index = t + r - 1;
			assert(0 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : CHARACTER_ID_EOS;
			double pi_k = (k == crf->_index_w_unigram_b(yt_1, yt, pos, x_i)) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_unigram_b(1, 1, pos, x_i) == k) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_unigram_b(0, 0, pos, x_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_unigram_b(0, 1, pos, x_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_unigram_b(1, 0, pos, x_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_unigram_b(1, 1, pos, x_i) == k) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", r = " << r << ", index = " << index << ", x_i = " << x_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		if(k > 0){
			crf->_w_unigram[k] -= 1e-8;
		}
		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_w_unigram[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		if(std::abs(true_grad - grad) >= 1e-4){
			cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		}
		assert(std::abs(true_grad - grad) < 1e-4);
	}

	delete sentence;
	delete var;
}

void test_grad_bigram(){
	Variables* var = new Variables();
	Model* model = var->model;
	Lattice* lattice = model->_lattice;
	Sentence* sentence = generate_sentence_4();
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];
	lattice->_enumerate_proportional_p_substring_given_sentence(lattice->_pc_s, sentence->size(), lattice->_alpha, lattice->_beta, _Zs);
	lattice->_enumerate_marginal_p_path_given_sentence(lattice->_pz_s, sentence->size(), lattice->_pc_s);

	crf::CRF* crf = var->py_crf->_crf;
	int const* character_ids = sentence->_character_ids;
	wchar_t const* characters = sentence->_characters;
	int character_ids_length = sentence->size();

	for(int pos = 1;pos <= crf->_x_range_bigram;pos++){
		for(int x_i = 0;x_i < token_ids.size();x_i++){
			for(int x_i_1 = 0;x_i_1 < token_ids.size();x_i_1++){
				// cout << "index = " << crf->_index_w_bigram_u(0, pos, x_i_1, x_i) << ", pos = " << pos << ", x_i_1 = " << x_i_1 << ", x_i = " << x_i << endl;
				// cout << "index = " << crf->_index_w_bigram_u(1, pos, x_i_1, x_i) << ", pos = " << pos << ", x_i_1 = " << x_i_1 << ", x_i = " << x_i << endl;
				assert(pos == (crf->_index_w_bigram_u(0, pos, x_i_1, x_i) % (crf->_x_range_bigram * 2)) / 2 + 1);
				assert(pos == (crf->_index_w_bigram_u(1, pos, x_i_1, x_i) % (crf->_x_range_bigram * 2)) / 2 + 1);
			}
		}
	}

	// sentence->dump_words();

	for(int k = 0;k < crf->_w_size_bigram_u;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int pos = (k % (crf->_x_range_bigram * 2)) / 2 + 1;
		int t_start = std::max(1, -(crf->_x_bigram_start + pos - 1) + 2);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_bigram_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = 1;t < t_start;t++){
			int s = (i < sentence->_num_segments - 1) ? sentence->_start[i] + 1 : sentence->size() + 1;
			yt_1 = yt;
			yt = 0;
			if(t == s){
				i = (i < sentence->_num_segments - 1) ? i + 1 : sentence->_num_segments - 1;
				yt = 1;
			}
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
		}
		for(int t = t_start;t <= t_end;t++){
			int s = (i < sentence->_num_segments - 1) ? sentence->_start[i] + 1 : sentence->size() + 1;
			yt_1 = yt;
			yt = 0;
			if(t == s || t == sentence->size() + 2){
				i = (i < sentence->_num_segments - 1) ? i + 1 : sentence->_num_segments - 1;
				yt = 1;
			}
			int r = crf->_x_bigram_start + (pos - 1);
			int index = t + r - 1;
			assert(1 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : CHARACTER_ID_EOS;
			int x_i_1 = (index - 1 < character_ids_length) ? character_ids[index - 1] : CHARACTER_ID_EOS;
			double pi_k = (k == crf->_index_w_bigram_u(yt, pos, x_i_1, x_i)) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_bigram_u(1, pos, x_i_1, x_i) == k) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_bigram_u(0, pos, x_i_1, x_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_bigram_u(1, pos, x_i_1, x_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_bigram_u(0, pos, x_i_1, x_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_bigram_u(1, pos, x_i_1, x_i) == k) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", r = " << r << ", index = " << index << ", x_i = " << x_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		if(k > 0){
			crf->_w_bigram[k] -= 1e-8;
		}
		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_w_bigram[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		assert(std::abs(true_grad - grad) < 1e-4);

	}

	crf->_w_bigram[crf->_w_size_unigram_u - 1] -= 1e-8;

	for(int pos = 1;pos <= crf->_x_range_bigram;pos++){
		for(int x_i = 0;x_i < token_ids.size();x_i++){
			for(int x_i_1 = 0;x_i_1 < token_ids.size();x_i_1++){
				assert(pos == ((crf->_index_w_bigram_b(0, 0, pos, x_i_1, x_i) - crf->_w_size_bigram_u) % (crf->_x_range_bigram * 2 * 2)) / 4 + 1);
				assert(pos == ((crf->_index_w_bigram_b(0, 1, pos, x_i_1, x_i) - crf->_w_size_bigram_u) % (crf->_x_range_bigram * 2 * 2)) / 4 + 1);
				assert(pos == ((crf->_index_w_bigram_b(1, 0, pos, x_i_1, x_i) - crf->_w_size_bigram_u) % (crf->_x_range_bigram * 2 * 2)) / 4 + 1);
				assert(pos == ((crf->_index_w_bigram_b(1, 1, pos, x_i_1, x_i) - crf->_w_size_bigram_u) % (crf->_x_range_bigram * 2 * 2)) / 4 + 1);
			}
		}
	}
	
	for(int k = crf->_w_size_bigram_u;k < crf->_w_size_bigram_u + crf->_w_size_bigram_b;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int pos = ((k - crf->_w_size_bigram_u) % (crf->_x_range_bigram * 2 * 2)) / 4 + 1;
		int t_start = std::max(1, -(crf->_x_bigram_start + pos - 1) + 2);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_bigram_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = 1;t < t_start;t++){
			int s = (i < sentence->_num_segments - 1) ? sentence->_start[i] + 1 : sentence->size() + 1;
			yt_1 = yt;
			yt = 0;
			if(t == s){
				i = (i < sentence->_num_segments - 1) ? i + 1 : sentence->_num_segments - 1;
				yt = 1;
			}
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
		}
		for(int t = t_start;t <= t_end;t++){
			int s = (i < sentence->_num_segments - 1) ? sentence->_start[i] + 1 : sentence->size() + 1;
			yt_1 = yt;
			yt = 0;
			if(t == s || t == sentence->size() + 2){
				i = (i < sentence->_num_segments - 1) ? i + 1 : sentence->_num_segments - 1;
				yt = 1;
			}
			int r = crf->_x_bigram_start + (pos - 1);
			int index = t + r - 1;
			assert(1 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : CHARACTER_ID_EOS;
			int x_i_1 = (index - 1 < character_ids_length) ? character_ids[index - 1] : CHARACTER_ID_EOS;
			double pi_k = (k == crf->_index_w_bigram_b(yt_1, yt, pos, x_i_1, x_i)) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_bigram_b(1, 1, pos, x_i_1, x_i) == k) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_bigram_b(0, 0, pos, x_i_1, x_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_bigram_b(0, 1, pos, x_i_1, x_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_bigram_b(1, 0, pos, x_i_1, x_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_bigram_b(1, 1, pos, x_i_1, x_i) == k) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", r = " << r << ", index = " << index << ", x_i = " << x_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		if(k > 0){
			crf->_w_bigram[k] -= 1e-8;
		}
		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_w_bigram[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		assert(std::abs(true_grad - grad) < 1e-4);

	}

	delete sentence;
	delete var;
}

void test_grad_identical_1(){
	Variables* var = new Variables();
	Model* model = var->model;
	Lattice* lattice = model->_lattice;
	Sentence* sentence = generate_sentence_4();
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];
	lattice->_enumerate_proportional_p_substring_given_sentence(lattice->_pc_s, sentence->size(), lattice->_alpha, lattice->_beta, _Zs);
	lattice->_enumerate_marginal_p_path_given_sentence(lattice->_pz_s, sentence->size(), lattice->_pc_s);

	crf::CRF* crf = var->py_crf->_crf;
	int const* character_ids = sentence->_character_ids;
	wchar_t const* characters = sentence->_characters;
	int character_ids_length = sentence->size();

	for(int pos = 1;pos <= crf->_x_range_identical_1;pos++){
		// cout << "index = " << crf->_index_w_identical_1_u(0, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
		// cout << "index = " << crf->_index_w_identical_1_u(1, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
		assert(pos == crf->_index_w_identical_1_u(0, pos) / 2 + 1);
		assert(pos == crf->_index_w_identical_1_u(1, pos) / 2 + 1);
	}

	// sentence->dump_words();

	for(int k = 0;k < crf->_w_size_identical_1_u;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int pos = k / 2 + 1;
		int t_start = std::max(1, -(crf->_x_identical_1_start + pos - 1) + 2);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_identical_1_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = 1;t < t_start;t++){
			int s = (i < sentence->_num_segments - 1) ? sentence->_start[i] + 1 : sentence->size() + 1;
			yt_1 = yt;
			yt = 0;
			if(t == s){
				i = (i < sentence->_num_segments - 1) ? i + 1 : sentence->_num_segments - 1;
				yt = 1;
			}
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
		}
		for(int t = t_start;t <= t_end;t++){
			int s = (i < sentence->_num_segments - 1) ? sentence->_start[i] + 1 : sentence->size() + 1;
			yt_1 = yt;
			yt = 0;
			if(t == s || t == sentence->size() + 2){
				i = (i < sentence->_num_segments - 1) ? i + 1 : sentence->_num_segments - 1;
				yt = 1;
			}
			int r = crf->_x_identical_1_start + (pos - 1);
			int index = t + r - 1;
			assert(1 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : CHARACTER_ID_EOS;
			int x_i_1 = (index - 1 < character_ids_length) ? character_ids[index - 1] : CHARACTER_ID_EOS;
			double pi_k = (k == crf->_index_w_identical_1_u(yt, pos) && x_i_1 == x_i) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_identical_1_u(1, pos) == k && x_i_1 == x_i) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_identical_1_u(0, pos) == k && x_i_1 == x_i) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_identical_1_u(1, pos) == k && x_i_1 == x_i) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_identical_1_u(0, pos) == k && x_i_1 == x_i) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_identical_1_u(1, pos) == k && x_i_1 == x_i) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", r = " << r << ", index = " << index << ", x_i = " << x_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		if(k > 0){
			crf->_w_identical_1[k] -= 1e-8;
		}
		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_w_identical_1[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		assert(std::abs(true_grad - grad) < 1e-4);

	}

	crf->_w_identical_1[crf->_w_size_identical_1_u - 1] -= 1e-8;

	for(int pos = 1;pos <= crf->_x_range_identical_1;pos++){
		assert(pos == (crf->_index_w_identical_1_b(0, 0, pos) - crf->_w_size_identical_1_u) / 4 + 1);
		assert(pos == (crf->_index_w_identical_1_b(0, 1, pos) - crf->_w_size_identical_1_u) / 4 + 1);
		assert(pos == (crf->_index_w_identical_1_b(1, 0, pos) - crf->_w_size_identical_1_u) / 4 + 1);
		assert(pos == (crf->_index_w_identical_1_b(1, 1, pos) - crf->_w_size_identical_1_u) / 4 + 1);
	}
	
	for(int k = crf->_w_size_identical_1_u;k < crf->_w_size_identical_1_u + crf->_w_size_identical_1_b;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int pos = (k - crf->_w_size_identical_1_u) / 4 + 1;
		int t_start = std::max(1, -(crf->_x_identical_1_start + pos - 1) + 2);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_identical_1_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = 1;t < t_start;t++){
			int s = (i < sentence->_num_segments - 1) ? sentence->_start[i] + 1 : sentence->size() + 1;
			yt_1 = yt;
			yt = 0;
			if(t == s){
				i = (i < sentence->_num_segments - 1) ? i + 1 : sentence->_num_segments - 1;
				yt = 1;
			}
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
		}
		for(int t = t_start;t <= t_end;t++){
			int s = (i < sentence->_num_segments - 1) ? sentence->_start[i] + 1 : sentence->size() + 1;
			yt_1 = yt;
			yt = 0;
			if(t == s || t == sentence->size() + 2){
				i = (i < sentence->_num_segments - 1) ? i + 1 : sentence->_num_segments - 1;
				yt = 1;
			}
			int r = crf->_x_identical_1_start + (pos - 1);
			int index = t + r - 1;
			assert(1 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : CHARACTER_ID_EOS;
			int x_i_1 = (index - 1 < character_ids_length) ? character_ids[index - 1] : CHARACTER_ID_EOS;
			double pi_k = (k == crf->_index_w_identical_1_b(yt_1, yt, pos) && x_i_1 == x_i) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_identical_1_b(1, 1, pos) == k && x_i_1 == x_i) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_identical_1_b(0, 0, pos) == k && x_i_1 == x_i) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_identical_1_b(0, 1, pos) == k && x_i_1 == x_i) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_identical_1_b(1, 0, pos) == k && x_i_1 == x_i) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_identical_1_b(1, 1, pos) == k && x_i_1 == x_i) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", r = " << r << ", index = " << index << ", x_i = " << x_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		if(k > 0){
			crf->_w_identical_1[k] -= 1e-8;
		}
		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_w_identical_1[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		cout << std::abs(true_grad - grad) << endl;
		assert(std::abs(true_grad - grad) < 1e-4);

	}


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
	token_ids[CHARACTER_ID_UNK] = token_ids.size();
	token_ids[CHARACTER_ID_BOS] = token_ids.size();
	token_ids[CHARACTER_ID_EOS] = token_ids.size();

	test_compute_normalizing_constant();
	cout << "OK" << endl;
	test_viterbi_decode();
	cout << "OK" << endl;
	test_scaling();
	cout << "OK" << endl;
	test_enumerate_proportional_p_substring_given_sentence();
	cout << "OK" << endl;
	test_enumerate_marginal_p_path_given_sentence();
	cout << "OK" << endl;
	test_grad_unigram();
	cout << "OK" << endl;
	test_grad_bigram();
	cout << "OK" << endl;
	test_grad_identical_1();
	cout << "OK" << endl;
}
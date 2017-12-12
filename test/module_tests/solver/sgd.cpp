#include  <iostream>
#include <chrono>
#include "../../../src/npycrf/sampler.h"
#include "../../../src/npycrf/array.h"
#include "../../../src/npycrf/ctype.h"
#include "../../../src/npycrf/solver/sgd.h"
#include "../../../src/python/npycrf.h"
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
	array<int> character_ids(sentence_str.size());
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
	NPYCRF* model;
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
		py_npylm = new python::model::NPYLM(max_word_length, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);

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
		py_crf = new python::model::CRF(num_character_ids, feature_x_unigram_start, feature_x_unigram_end, feature_x_bigram_start, feature_x_bigram_end, feature_x_identical_1_start, feature_x_identical_1_end, feature_x_identical_2_start, feature_x_identical_2_end, 1.0, sigma);

		model = new NPYCRF(py_npylm, py_crf);
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

void test_backward_unigram(bool pure_crf_mode){
	Variables* var = new Variables();
	NPYCRF* model = var->model;
	Lattice* lattice = model->_lattice;
	lattice->set_pure_crf_mode(pure_crf_mode);
	Sentence* sentence = generate_sentence_4();
	lattice->_clear_word_id_cache();
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];
	lattice->_enumerate_proportional_p_substring_given_sentence(lattice->_pc_s, sentence->size(), lattice->_alpha, lattice->_beta, _Zs);
	lattice->_enumerate_marginal_p_path_given_sentence(lattice->_pz_s, sentence->size(), lattice->_pc_s);

	crf::CRF* crf = var->py_crf->_crf;
	sentence->_features = crf->extract_features(sentence);
	double regularization_constant = 1.0;
	solver::SGD* sgd = new solver::SGD(crf, regularization_constant);
	sgd->_backward_unigram(sentence, lattice->_pz_s);

	array<int> &character_ids = sentence->_character_ids;
	wchar_t const* characters = sentence->_characters;
	int character_ids_length = sentence->size();

	for(int pos = 1;pos <= crf->_x_range_unigram;pos++){
		for(int x_i = 0;x_i < token_ids.size();x_i++){
			// cout << "index = " << crf->_index_w_unigram_u(0, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
			// cout << "index = " << crf->_index_w_unigram_u(1, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
			assert(pos == ((crf->_index_w_unigram_u(0, pos, x_i) - crf->_offset_w_unigram_u) % (crf->_x_range_unigram * 2)) / 2 + 1);
			assert(pos == ((crf->_index_w_unigram_u(1, pos, x_i) - crf->_offset_w_unigram_u) % (crf->_x_range_unigram * 2)) / 2 + 1);
		}
	}

	double* grad_w_unigram = new double[crf->_w_size_unigram_u];
	for(int k = 0;k < crf->_w_size_unigram_u;k++){
		grad_w_unigram[k] = 0;
	}

	// sentence->dump_words();

	for(int k = crf->_offset_w_unigram_u;k < crf->_offset_w_unigram_u + crf->_w_size_unigram_u;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int pos = ((k - crf->_offset_w_unigram_u) % (crf->_x_range_unigram * 2)) / 2 + 1;
		int t_start = std::max(1, -(crf->_x_unigram_start + pos - 1) + 1);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_unigram_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			int r = crf->_x_unigram_start + (pos - 1);
			int index = t + r - 1;
			assert(0 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : SPECIAL_CHARACTER_END;
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

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
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
		grad_w_unigram[k - crf->_offset_w_unigram_u] = grad;
	}

	for(int k = 0;k < crf->_w_size_unigram_u;k++){
		// cout << "k = " << k << ", " << grad_w_unigram[k - crf->_offset_w_unigram_u] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_unigram[k] - sgd->_grad_weight[k + crf->_offset_w_unigram_u]) < 1e-12);
	}

	delete[] grad_w_unigram;

	grad_w_unigram = new double[crf->_w_size_unigram_b];
	for(int k = 0;k < crf->_w_size_unigram_b;k++){
		grad_w_unigram[k] = 0;
	}

	for(int pos = 1;pos <= crf->_x_range_unigram;pos++){
		for(int x_i = 0;x_i < token_ids.size();x_i++){
			// cout << "index = " << crf->_index_w_unigram_b(0, 0, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
			// cout << "index = " << crf->_index_w_unigram_b(1, 0, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
			// cout << "index = " << crf->_index_w_unigram_b(0, 1, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
			// cout << "index = " << crf->_index_w_unigram_b(1, 1, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;

			assert(pos == ((crf->_index_w_unigram_b(0, 0, pos, x_i) - crf->_offset_w_unigram_b) % (crf->_x_range_unigram * 2 * 2)) / 4 + 1);
			assert(pos == ((crf->_index_w_unigram_b(1, 0, pos, x_i) - crf->_offset_w_unigram_b) % (crf->_x_range_unigram * 2 * 2)) / 4 + 1);
			assert(pos == ((crf->_index_w_unigram_b(0, 1, pos, x_i) - crf->_offset_w_unigram_b) % (crf->_x_range_unigram * 2 * 2)) / 4 + 1);
			assert(pos == ((crf->_index_w_unigram_b(1, 1, pos, x_i) - crf->_offset_w_unigram_b) % (crf->_x_range_unigram * 2 * 2)) / 4 + 1);
		}
	}

	for(int k = crf->_offset_w_unigram_b;k < crf->_offset_w_unigram_b + crf->_w_size_unigram_b;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int pos = ((k - crf->_offset_w_unigram_b) % (crf->_x_range_unigram * 2 * 2)) / 4 + 1;
		int t_start = std::max(1, -(crf->_x_unigram_start + pos - 1) + 1);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_unigram_start + pos - 1));
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			int r = crf->_x_unigram_start + (pos - 1);
			int index = t + r - 1;
			assert(0 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : SPECIAL_CHARACTER_END;
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

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
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
		grad_w_unigram[k - crf->_offset_w_unigram_b] = grad;
	}

	for(int k = crf->_offset_w_unigram_b;k < crf->_offset_w_unigram_b + crf->_w_size_unigram_b;k++){
		// cout << "k = " << k << ", " << grad_w_unigram[k] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_unigram[k - crf->_offset_w_unigram_b] - sgd->_grad_weight[k]) < 1e-12);
	}

	delete[] grad_w_unigram;
	delete sgd;
	delete sentence;
	delete var;
}

void test_backward_bigram(bool pure_crf_mode){
	Variables* var = new Variables();
	NPYCRF* model = var->model;
	Lattice* lattice = model->_lattice;
	lattice->_clear_word_id_cache();
	lattice->set_pure_crf_mode(pure_crf_mode);
	Sentence* sentence = generate_sentence_4();
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];
	lattice->_enumerate_proportional_p_substring_given_sentence(lattice->_pc_s, sentence->size(), lattice->_alpha, lattice->_beta, _Zs);
	lattice->_enumerate_marginal_p_path_given_sentence(lattice->_pz_s, sentence->size(), lattice->_pc_s);

	crf::CRF* crf = var->py_crf->_crf;
	sentence->_features = crf->extract_features(sentence);
	double regularization_constant = 1.0;
	solver::SGD* sgd = new solver::SGD(crf, regularization_constant);
	sgd->_backward_bigram(sentence, lattice->_pz_s);

	double* grad_w_bigram = new double[crf->_w_size_bigram_u];
	for(int k = 0;k < crf->_w_size_bigram_u;k++){
		grad_w_bigram[k] = 0;
	}

	array<int> &character_ids = sentence->_character_ids;
	wchar_t const* characters = sentence->_characters;
	int character_ids_length = sentence->size();

	for(int pos = 1;pos <= crf->_x_range_bigram;pos++){
		for(int x_i = 0;x_i < token_ids.size();x_i++){
			for(int x_i_1 = 0;x_i_1 < token_ids.size();x_i_1++){
				// cout << "index = " << crf->_index_w_bigram_u(0, pos, x_i_1, x_i) << ", pos = " << pos << ", x_i_1 = " << x_i_1 << ", x_i = " << x_i << endl;
				// cout << "index = " << crf->_index_w_bigram_u(1, pos, x_i_1, x_i) << ", pos = " << pos << ", x_i_1 = " << x_i_1 << ", x_i = " << x_i << endl;
				assert(pos == ((crf->_index_w_bigram_u(0, pos, x_i_1, x_i) - crf->_offset_w_bigram_u) % (crf->_x_range_bigram * 2)) / 2 + 1);
				assert(pos == ((crf->_index_w_bigram_u(1, pos, x_i_1, x_i) - crf->_offset_w_bigram_u) % (crf->_x_range_bigram * 2)) / 2 + 1);
			}
		}
	}
	// sentence->dump_words();

	for(int k = crf->_offset_w_bigram_u;k < crf->_offset_w_bigram_u + crf->_w_size_bigram_u;k++){
		double grad = 0;
		int pos = ((k - crf->_offset_w_bigram_u) % (crf->_x_range_bigram * 2)) / 2 + 1;
		int t_start = std::max(1, -(crf->_x_bigram_start + pos - 1) + 2);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_bigram_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			int r = crf->_x_bigram_start + (pos - 1);
			int index = t + r - 1;
			assert(1 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : SPECIAL_CHARACTER_END;
			int x_i_1 = (index - 1 < character_ids_length) ? character_ids[index - 1] : SPECIAL_CHARACTER_END;
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

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		assert(std::abs(true_grad - grad) < 1e-4);
		grad_w_bigram[k - crf->_offset_w_bigram_u] = grad;
	}

	for(int k = 0;k < crf->_w_size_bigram_u;k++){
		// cout << "k = " << k << ", " << grad_w_bigram[k] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_bigram[k] - sgd->_grad_weight[k + crf->_offset_w_bigram_u]) < 1e-12);
	}

	delete[] grad_w_bigram;

	grad_w_bigram = new double[crf->_w_size_bigram_b];
	for(int k = 0;k < crf->_w_size_bigram_b;k++){
		grad_w_bigram[k] = 0;
	}

	for(int pos = 1;pos <= crf->_x_range_bigram;pos++){
		for(int x_i = 0;x_i < token_ids.size();x_i++){
			for(int x_i_1 = 0;x_i_1 < token_ids.size();x_i_1++){
				assert(pos == ((crf->_index_w_bigram_b(0, 0, pos, x_i_1, x_i) - crf->_offset_w_bigram_b) % (crf->_x_range_bigram * 2 * 2)) / 4 + 1);
				assert(pos == ((crf->_index_w_bigram_b(0, 1, pos, x_i_1, x_i) - crf->_offset_w_bigram_b) % (crf->_x_range_bigram * 2 * 2)) / 4 + 1);
				assert(pos == ((crf->_index_w_bigram_b(1, 0, pos, x_i_1, x_i) - crf->_offset_w_bigram_b) % (crf->_x_range_bigram * 2 * 2)) / 4 + 1);
				assert(pos == ((crf->_index_w_bigram_b(1, 1, pos, x_i_1, x_i) - crf->_offset_w_bigram_b) % (crf->_x_range_bigram * 2 * 2)) / 4 + 1);
			}
		}
	}
	
	for(int k = crf->_offset_w_bigram_b;k < crf->_offset_w_bigram_b + crf->_w_size_bigram_b;k++){
		double grad = 0;
		int pos = ((k - crf->_offset_w_bigram_b) % (crf->_x_range_bigram * 2 * 2)) / 4 + 1;
		int t_start = std::max(1, -(crf->_x_bigram_start + pos - 1) + 2);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_bigram_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			int r = crf->_x_bigram_start + (pos - 1);
			int index = t + r - 1;
			assert(1 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : SPECIAL_CHARACTER_END;
			int x_i_1 = (index - 1 < character_ids_length) ? character_ids[index - 1] : SPECIAL_CHARACTER_END;
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

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		assert(std::abs(true_grad - grad) < 1e-4);
		grad_w_bigram[k - crf->_offset_w_bigram_b] = grad;

	}

	for(int k = 0;k < crf->_w_size_bigram_b;k++){
		// cout << "k = " << k << ", " << grad_w_bigram[k] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_bigram[k] - sgd->_grad_weight[k + crf->_offset_w_bigram_b]) < 1e-12);
	}

	delete[] grad_w_bigram;
	delete sgd;
	delete sentence;
	delete var;
}

void test_backward_identical_1(bool pure_crf_mode){
	Variables* var = new Variables();
	NPYCRF* model = var->model;
	Lattice* lattice = model->_lattice;
	lattice->_clear_word_id_cache();
	lattice->set_pure_crf_mode(pure_crf_mode);
	Sentence* sentence = generate_sentence_4();
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];
	lattice->_enumerate_proportional_p_substring_given_sentence(lattice->_pc_s, sentence->size(), lattice->_alpha, lattice->_beta, _Zs);
	lattice->_enumerate_marginal_p_path_given_sentence(lattice->_pz_s, sentence->size(), lattice->_pc_s);

	crf::CRF* crf = var->py_crf->_crf;
	sentence->_features = crf->extract_features(sentence);
	double regularization_constant = 1.0;
	solver::SGD* sgd = new solver::SGD(crf, regularization_constant);
	sgd->_backward_identical_1(sentence, lattice->_pz_s);

	double* grad_w_identical_1 = new double[crf->_w_size_identical_1_u];
	for(int k = 0;k < crf->_w_size_identical_1_u;k++){
		grad_w_identical_1[k] = 0;
	}

	array<int> &character_ids = sentence->_character_ids;
	wchar_t const* characters = sentence->_characters;
	int character_ids_length = sentence->size();

	for(int pos = 1;pos <= crf->_x_range_identical_1;pos++){
		// cout << "index = " << crf->_index_w_identical_1_u(0, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
		// cout << "index = " << crf->_index_w_identical_1_u(1, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
		assert(pos == (crf->_index_w_identical_1_u(0, pos) - crf->_offset_w_identical_1_u) / 2 + 1);
		assert(pos == (crf->_index_w_identical_1_u(1, pos) - crf->_offset_w_identical_1_u) / 2 + 1);
	}

	// sentence->dump_words();

	for(int k = crf->_offset_w_identical_1_u;k < crf->_offset_w_identical_1_u + crf->_w_size_identical_1_u;k++){
		double grad = 0;
		int pos = (k - crf->_offset_w_identical_1_u) / 2 + 1;
		int t_start = std::max(1, -(crf->_x_identical_1_start + pos - 1) + 2);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_identical_1_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			int r = crf->_x_identical_1_start + (pos - 1);
			int index = t + r - 1;
			assert(1 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : SPECIAL_CHARACTER_END;
			int x_i_1 = (index - 1 < character_ids_length) ? character_ids[index - 1] : SPECIAL_CHARACTER_END;
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

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		assert(std::abs(true_grad - grad) < 1e-4);
		grad_w_identical_1[k - crf->_offset_w_identical_1_u] = grad;
	}

	for(int k = 0;k < crf->_w_size_identical_1_u;k++){
		// cout << "k = " << k << ", " << grad_w_identical_1[k] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_identical_1[k] - sgd->_grad_weight[k + crf->_offset_w_identical_1_u]) < 1e-12);
	}
	delete[] grad_w_identical_1;

	grad_w_identical_1 = new double[crf->_w_size_identical_1_b];
	for(int k = 0;k < crf->_w_size_identical_1_b;k++){
		grad_w_identical_1[k] = 0;
	}

	for(int pos = 1;pos <= crf->_x_range_identical_1;pos++){
		assert(pos == (crf->_index_w_identical_1_b(0, 0, pos) - crf->_offset_w_identical_1_b) / 4 + 1);
		assert(pos == (crf->_index_w_identical_1_b(0, 1, pos) - crf->_offset_w_identical_1_b) / 4 + 1);
		assert(pos == (crf->_index_w_identical_1_b(1, 0, pos) - crf->_offset_w_identical_1_b) / 4 + 1);
		assert(pos == (crf->_index_w_identical_1_b(1, 1, pos) - crf->_offset_w_identical_1_b) / 4 + 1);
	}
	
	for(int k = crf->_offset_w_identical_1_b;k < crf->_offset_w_identical_1_b + crf->_w_size_identical_1_b;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int pos = (k - crf->_offset_w_identical_1_b) / 4 + 1;
		int t_start = std::max(1, -(crf->_x_identical_1_start + pos - 1) + 2);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_identical_1_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			int r = crf->_x_identical_1_start + (pos - 1);
			int index = t + r - 1;
			assert(1 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : SPECIAL_CHARACTER_END;
			int x_i_1 = (index - 1 < character_ids_length) ? character_ids[index - 1] : SPECIAL_CHARACTER_END;
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

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		assert(std::abs(true_grad - grad) < 1e-4);
		grad_w_identical_1[k - crf->_offset_w_identical_1_b] = grad;
	}

	for(int k = 0;k < crf->_w_size_identical_1_b;k++){
		// cout << "k = " << k << ", " << grad_w_identical_1[k] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_identical_1[k] - sgd->_grad_weight[k + crf->_offset_w_identical_1_b]) < 1e-12);
	}

	delete[] grad_w_identical_1;
	delete sgd;
	delete sentence;
	delete var;
}

void test_backward_identical_2(bool pure_crf_mode){
	Variables* var = new Variables();
	NPYCRF* model = var->model;
	Lattice* lattice = model->_lattice;
	lattice->_clear_word_id_cache();
	lattice->set_pure_crf_mode(pure_crf_mode);
	Sentence* sentence = generate_sentence_4();
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];
	lattice->_enumerate_proportional_p_substring_given_sentence(lattice->_pc_s, sentence->size(), lattice->_alpha, lattice->_beta, _Zs);
	lattice->_enumerate_marginal_p_path_given_sentence(lattice->_pz_s, sentence->size(), lattice->_pc_s);

	crf::CRF* crf = var->py_crf->_crf;
	sentence->_features = crf->extract_features(sentence);
	double regularization_constant = 1.0;
	solver::SGD* sgd = new solver::SGD(crf, regularization_constant);
	sgd->_backward_identical_2(sentence, lattice->_pz_s);

	double* grad_w_identical_2 = new double[crf->_w_size_identical_2_u];
	for(int k = 0;k < crf->_w_size_identical_2_u;k++){
		grad_w_identical_2[k] = 0;
	}

	array<int> &character_ids = sentence->_character_ids;
	wchar_t const* characters = sentence->_characters;
	int character_ids_length = sentence->size();

	for(int pos = 1;pos <= crf->_x_range_identical_2;pos++){
		// cout << "index = " << crf->_index_w_identical_2_u(0, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
		// cout << "index = " << crf->_index_w_identical_2_u(1, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
		assert(pos == (crf->_index_w_identical_2_u(0, pos) - crf->_offset_w_identical_2_u) / 2 + 1);
		assert(pos == (crf->_index_w_identical_2_u(1, pos) - crf->_offset_w_identical_2_u) / 2 + 1);
	}

	// sentence->dump_words();

	for(int k = crf->_offset_w_identical_2_u;k < crf->_offset_w_identical_2_u + crf->_w_size_identical_2_u;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int pos = (k - crf->_offset_w_identical_2_u) / 2 + 1;
		int t_start = std::max(1, -(crf->_x_identical_2_start + pos - 1) + 3);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_identical_2_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			int r = crf->_x_identical_2_start + (pos - 1);
			int index = t + r - 1;
			assert(2 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : SPECIAL_CHARACTER_END;
			int x_i_2 = (index - 2 < character_ids_length) ? character_ids[index - 2] : SPECIAL_CHARACTER_END;
			double pi_k = (k == crf->_index_w_identical_2_u(yt, pos) && x_i_2 == x_i) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_identical_2_u(1, pos) == k && x_i_2 == x_i) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_identical_2_u(0, pos) == k && x_i_2 == x_i) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_identical_2_u(1, pos) == k && x_i_2 == x_i) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_identical_2_u(0, pos) == k && x_i_2 == x_i) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_identical_2_u(1, pos) == k && x_i_2 == x_i) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", r = " << r << ", index = " << index << ", x_i = " << x_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		assert(std::abs(true_grad - grad) < 1e-4);
		grad_w_identical_2[k - crf->_offset_w_identical_2_u] = grad;		
	}

	for(int k = 0;k < crf->_w_size_identical_2_u;k++){
		// cout << "k = " << k << ", " << grad_w_identical_2[k] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_identical_2[k] - sgd->_grad_weight[k + crf->_offset_w_identical_2_u]) < 1e-12);
	}
	delete[] grad_w_identical_2;

	grad_w_identical_2 = new double[crf->_w_size_identical_2_b];
	for(int k = 0;k < crf->_w_size_identical_2_b;k++){
		grad_w_identical_2[k] = 0;
	}

	for(int pos = 1;pos <= crf->_x_range_identical_2;pos++){
		assert(pos == (crf->_index_w_identical_2_b(0, 0, pos) - crf->_offset_w_identical_2_b) / 4 + 1);
		assert(pos == (crf->_index_w_identical_2_b(0, 1, pos) - crf->_offset_w_identical_2_b) / 4 + 1);
		assert(pos == (crf->_index_w_identical_2_b(1, 0, pos) - crf->_offset_w_identical_2_b) / 4 + 1);
		assert(pos == (crf->_index_w_identical_2_b(1, 1, pos) - crf->_offset_w_identical_2_b) / 4 + 1);
	}
	
	for(int k = crf->_offset_w_identical_2_b;k < crf->_offset_w_identical_2_b + crf->_w_size_identical_2_b;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int pos = (k - crf->_offset_w_identical_2_b) / 4 + 1;
		int t_start = std::max(1, -(crf->_x_identical_2_start + pos - 1) + 3);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_identical_2_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			int r = crf->_x_identical_2_start + (pos - 1);
			int index = t + r - 1;
			assert(2 <= index);
			int x_i = (index < character_ids_length) ? character_ids[index] : SPECIAL_CHARACTER_END;
			int x_i_2 = (index - 2 < character_ids_length) ? character_ids[index - 2] : SPECIAL_CHARACTER_END;
			double pi_k = (k == crf->_index_w_identical_2_b(yt_1, yt, pos) && x_i_2 == x_i) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_identical_2_b(1, 1, pos) == k && x_i_2 == x_i) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_identical_2_b(0, 0, pos) == k && x_i_2 == x_i) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_identical_2_b(0, 1, pos) == k && x_i_2 == x_i) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_identical_2_b(1, 0, pos) == k && x_i_2 == x_i) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_identical_2_b(1, 1, pos) == k && x_i_2 == x_i) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", r = " << r << ", index = " << index << ", x_i = " << x_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		assert(std::abs(true_grad - grad) < 1e-4);
		grad_w_identical_2[k - crf->_offset_w_identical_2_b] = grad;		

	}

	for(int k = 0;k < crf->_w_size_identical_2_b;k++){
		// cout << "k = " << k << ", " << grad_w_identical_2[k] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_identical_2[k] - sgd->_grad_weight[k + crf->_offset_w_identical_2_b]) < 1e-12);
	}

	delete[] grad_w_identical_2;
	delete sgd;
	delete sentence;
	delete var;
}

void test_backward_character_type_unigram(bool pure_crf_mode){
	Variables* var = new Variables();
	NPYCRF* model = var->model;
	Lattice* lattice = model->_lattice;
	lattice->_clear_word_id_cache();
	lattice->set_pure_crf_mode(pure_crf_mode);
	Sentence* sentence = generate_sentence_4();
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];
	lattice->_enumerate_proportional_p_substring_given_sentence(lattice->_pc_s, sentence->size(), lattice->_alpha, lattice->_beta, _Zs);
	lattice->_enumerate_marginal_p_path_given_sentence(lattice->_pz_s, sentence->size(), lattice->_pc_s);

	crf::CRF* crf = var->py_crf->_crf;
	sentence->_features = crf->extract_features(sentence);
	double regularization_constant = 1.0;
	solver::SGD* sgd = new solver::SGD(crf, regularization_constant);
	sgd->_backward_unigram_type(sentence, lattice->_pz_s);

	double* grad_w_unigram_type = new double[crf->_w_size_unigram_type_u];
	for(int k = 0;k < crf->_w_size_unigram_type_u;k++){
		grad_w_unigram_type[k] = 0;
	}
	array<int> &character_ids = sentence->_character_ids;
	wchar_t const* characters = sentence->_characters;
	int character_ids_length = sentence->size();

	// sentence->dump_words();

	for(int k = crf->_offset_w_unigram_type_u;k < crf->_offset_w_unigram_type_u + crf->_w_size_unigram_type_u;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int t_start = 1;
		int t_end = sentence->size() + 2;
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			wchar_t c_i = (t <= character_ids_length) ? characters[t - 1] : 0;
			unsigned int type_i = (t <= character_ids_length) ? ctype::get_type(c_i) : CTYPE_UNKNOWN;
			double pi_k = (k == crf->_index_w_unigram_type_u(yt, type_i)) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_unigram_type_u(1, type_i) == k) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_unigram_type_u(0, type_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_unigram_type_u(1, type_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_unigram_type_u(0, type_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_unigram_type_u(1, type_i) == k) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", type_i = " << type_i << ", x_i = " << x_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		assert(std::abs(true_grad - grad) < 1e-4);
		grad_w_unigram_type[k - crf->_offset_w_unigram_type_u] = grad;
	}

	for(int k = 0;k < crf->_w_size_unigram_type_u;k++){
		// cout << "k = " << k << ", " << grad_w_unigram_type[k] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_unigram_type[k] - sgd->_grad_weight[k + crf->_offset_w_unigram_type_u]) < 1e-12);
	}
	delete[] grad_w_unigram_type;

	grad_w_unigram_type = new double[crf->_w_size_unigram_type_b];
	for(int k = 0;k < crf->_w_size_unigram_type_b;k++){
		grad_w_unigram_type[k] = 0;
	}

	for(int k = crf->_offset_w_unigram_type_b;k < crf->_offset_w_unigram_type_b + crf->_w_size_unigram_type_b;k++){
		double grad = 0;
		int t_start = 1;
		int t_end = sentence->size() + 2;
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			unsigned int type_i = (t <= character_ids_length) ? ctype::get_type(characters[t - 1]) : CTYPE_UNKNOWN;
			double pi_k = (k == crf->_index_w_unigram_type_b(yt_1, yt, type_i)) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_unigram_type_b(1, 1, type_i) == k) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_unigram_type_b(0, 0, type_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_unigram_type_b(0, 1, type_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_unigram_type_b(1, 0, type_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_unigram_type_b(1, 1, type_i) == k) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", r = " << r << ", index = " << index << ", x_i = " << x_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		assert(std::abs(true_grad - grad) < 1e-4);
		grad_w_unigram_type[k - crf->_offset_w_unigram_type_b] = grad;
	}

	for(int k = 0;k < crf->_w_size_unigram_type_b;k++){
		// cout << "k = " << k << ", " << grad_w_unigram_type[k] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_unigram_type[k] - sgd->_grad_weight[k + crf->_offset_w_unigram_type_b]) < 1e-12);
	}

	delete[] grad_w_unigram_type;
	delete sgd;
	delete sentence;
	delete var;
}

void test_backward_character_type_bigram(bool pure_crf_mode){
	Variables* var = new Variables();
	NPYCRF* model = var->model;
	Lattice* lattice = model->_lattice;
	lattice->_clear_word_id_cache();
	lattice->set_pure_crf_mode(pure_crf_mode);
	Sentence* sentence = generate_sentence_4();
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];
	lattice->_enumerate_proportional_p_substring_given_sentence(lattice->_pc_s, sentence->size(), lattice->_alpha, lattice->_beta, _Zs);
	lattice->_enumerate_marginal_p_path_given_sentence(lattice->_pz_s, sentence->size(), lattice->_pc_s);

	crf::CRF* crf = var->py_crf->_crf;
	sentence->_features = crf->extract_features(sentence);
	double regularization_constant = 1.0;
	solver::SGD* sgd = new solver::SGD(crf, regularization_constant);
	sgd->_backward_bigram_type(sentence, lattice->_pz_s);

	double* grad_w_bigram_type = new double[crf->_w_size_bigram_type_u];
	for(int k = 0;k < crf->_w_size_bigram_type_u;k++){
		grad_w_bigram_type[k] = 0;
	}
	array<int> &character_ids = sentence->_character_ids;
	wchar_t const* characters = sentence->_characters;
	int character_ids_length = sentence->size();

	// sentence->dump_words();

	// cout << crf->_w_size_bigram_type_u << endl;
	// cout  << endl;

	for(int type_i = 0;type_i < 281;type_i++){
		for(int type_i_1 = 0;type_i_1 < 281;type_i_1++){
			assert(type_i == (crf->_index_w_bigram_type_u(0, type_i_1, type_i) - crf->_offset_w_bigram_type_u) / (281 * 2));
			assert(type_i == (crf->_index_w_bigram_type_u(1, type_i_1, type_i) - crf->_offset_w_bigram_type_u) / (281 * 2));
			assert(type_i_1 == (crf->_index_w_bigram_type_u(0, type_i_1, type_i) - crf->_offset_w_bigram_type_u) % (281 * 2) / 2);
			assert(type_i_1 == (crf->_index_w_bigram_type_u(1, type_i_1, type_i) - crf->_offset_w_bigram_type_u) % (281 * 2) / 2);
		}
	}

	for(int k = crf->_offset_w_bigram_type_u;k < crf->_offset_w_bigram_type_u + crf->_w_size_bigram_type_u;k++){
		unsigned int type_i = (k - crf->_offset_w_bigram_type_u) / (281 * 2);
		unsigned int type_i_1 = (k - crf->_offset_w_bigram_type_u) % (281 * 2) / 2;
		if(!((type_i == CTYPE_HIRAGANA || type_i == CTYPE_UNKNOWN) && (type_i_1 == CTYPE_HIRAGANA || type_i_1 == CTYPE_UNKNOWN))){
			continue;
		}
		// cout << "k = " << k / (281 * 2) << endl;
		double grad = 0;
		int t_start = 2;
		int t_end = sentence->size() + 2;
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			wchar_t c_i = (t <= character_ids_length) ? characters[t - 1] : 0;
			unsigned int type_i = (t <= character_ids_length) ? ctype::get_type(c_i) : CTYPE_UNKNOWN;
			wchar_t c_i_1 = (t - 1 <= character_ids_length) ? characters[t - 2] : 0;
			unsigned int type_i_1 = (t - 1 <= character_ids_length) ? ctype::get_type(c_i_1) : CTYPE_UNKNOWN;
			double pi_k = (k == crf->_index_w_bigram_type_u(yt, type_i_1, type_i)) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_bigram_type_u(1, type_i_1, type_i) == k) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_bigram_type_u(0, type_i_1, type_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_bigram_type_u(1, type_i_1, type_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_bigram_type_u(0, type_i_1, type_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_bigram_type_u(1, type_i_1, type_i) == k) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", type_i_1 = " << type_i_1 << ", type_i = " << type_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << std::abs(true_grad - grad) << endl;
		if(std::abs(true_grad - grad) >= 1e-4){
			cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
			cout << std::abs(true_grad - grad) << endl;
		}
		assert(std::abs(true_grad - grad) < 1e-4);
		grad_w_bigram_type[k - crf->_offset_w_bigram_type_u] = grad;
	}
	
	for(int k = 0;k < crf->_w_size_bigram_type_u;k++){
		// cout << "k = " << k << ", " << grad_w_bigram_type[k] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_bigram_type[k] - sgd->_grad_weight[k + crf->_offset_w_bigram_type_u]) < 1e-12);
	}
	delete[] grad_w_bigram_type;

	grad_w_bigram_type = new double[crf->_w_size_bigram_type_b];
	for(int k = 0;k < crf->_w_size_bigram_type_b;k++){
		grad_w_bigram_type[k] = 0;
	}

	for(unsigned int type_i = 0;type_i < 281;type_i++){
		for(unsigned int type_i_1 = 0;type_i_1 < 281;type_i_1++){
			assert(type_i == (crf->_index_w_bigram_type_b(0, 0, type_i_1, type_i) - crf->_offset_w_bigram_type_b) / (281 * 4));
			assert(type_i == (crf->_index_w_bigram_type_b(0, 1, type_i_1, type_i) - crf->_offset_w_bigram_type_b) / (281 * 4));
			assert(type_i == (crf->_index_w_bigram_type_b(1, 0, type_i_1, type_i) - crf->_offset_w_bigram_type_b) / (281 * 4));
			assert(type_i == (crf->_index_w_bigram_type_b(1, 1, type_i_1, type_i) - crf->_offset_w_bigram_type_b) / (281 * 4));
			assert(type_i_1 == (crf->_index_w_bigram_type_b(0, 0, type_i_1, type_i) - crf->_offset_w_bigram_type_b) % (281 * 4) / 4);
			assert(type_i_1 == (crf->_index_w_bigram_type_b(0, 1, type_i_1, type_i) - crf->_offset_w_bigram_type_b) % (281 * 4) / 4);
			assert(type_i_1 == (crf->_index_w_bigram_type_b(1, 0, type_i_1, type_i) - crf->_offset_w_bigram_type_b) % (281 * 4) / 4);
			assert(type_i_1 == (crf->_index_w_bigram_type_b(1, 1, type_i_1, type_i) - crf->_offset_w_bigram_type_b) % (281 * 4) / 4);
		}
	}


	for(int k = crf->_offset_w_bigram_type_b;k < crf->_offset_w_bigram_type_b + crf->_w_size_bigram_type_b;k++){
		unsigned int type_i = (k - crf->_offset_w_bigram_type_b) / (281 * 4);
		unsigned int type_i_1 = (k - crf->_offset_w_bigram_type_b) % (281 * 4) / 4;
		if(!((type_i == CTYPE_HIRAGANA || type_i == CTYPE_UNKNOWN) && (type_i_1 == CTYPE_HIRAGANA || type_i_1 == CTYPE_UNKNOWN))){
			continue;
		}
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int t_start = 2;
		int t_end = sentence->size() + 2;
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			wchar_t c_i = (t <= character_ids_length) ? characters[t - 1] : 0;
			unsigned int type_i = (t <= character_ids_length) ? ctype::get_type(c_i) : CTYPE_UNKNOWN;
			wchar_t c_i_1 = (t - 1 <= character_ids_length) ? characters[t - 2] : 0;
			unsigned int type_i_1 = (t - 1 <= character_ids_length) ? ctype::get_type(c_i_1) : CTYPE_UNKNOWN;
			double pi_k = (k == crf->_index_w_bigram_type_b(yt_1, yt, type_i_1, type_i)) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_bigram_type_b(1, 1, type_i_1, type_i) == k) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_bigram_type_b(0, 0, type_i_1, type_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_bigram_type_b(0, 1, type_i_1, type_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_bigram_type_b(1, 0, type_i_1, type_i) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_bigram_type_b(1, 1, type_i_1, type_i) == k) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", r = " << r << ", index = " << index << ", x_i = " << x_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
		// cout << _log_Zs << " == " << _log_py << endl;
		double true_grad = (_log_py - log_py) / 1e-8;
		if(true_grad == 0 && grad == 0){
			continue;
		}
		// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
		// cout << std::abs(true_grad - grad) << endl;
		assert(std::abs(true_grad - grad) < 1e-4);
		grad_w_bigram_type[k - crf->_offset_w_bigram_type_b] = grad;
	}

	for(int k = 0;k < crf->_w_size_bigram_type_b;k++){
		// cout << "k = " << k << ", " << grad_w_bigram_type[k] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_bigram_type[k] - sgd->_grad_weight[k + crf->_offset_w_bigram_type_b]) < 1e-12);
	}

	delete[] grad_w_bigram_type;
	delete sgd;
	delete sentence;
	delete var;
}

void test_backward_label(bool pure_crf_mode){
	Variables* var = new Variables();
	NPYCRF* model = var->model;
	Lattice* lattice = model->_lattice;
	lattice->_clear_word_id_cache();
	lattice->set_pure_crf_mode(pure_crf_mode);
	Sentence* sentence = generate_sentence_4();
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];
	lattice->_enumerate_proportional_p_substring_given_sentence(lattice->_pc_s, sentence->size(), lattice->_alpha, lattice->_beta, _Zs);
	lattice->_enumerate_marginal_p_path_given_sentence(lattice->_pz_s, sentence->size(), lattice->_pc_s);

	crf::CRF* crf = var->py_crf->_crf;
	sentence->_features = crf->extract_features(sentence);
	double regularization_constant = 1.0;
	solver::SGD* sgd = new solver::SGD(crf, regularization_constant);
	sgd->_backward_label(sentence, lattice->_pz_s);

	double* grad_w_label = new double[crf->_w_size_label_u];
	for(int k = 0;k < crf->_w_size_label_u;k++){
		grad_w_label[k] = 0;
	}
	array<int> &character_ids = sentence->_character_ids;
	wchar_t const* characters = sentence->_characters;
	int character_ids_length = sentence->size();

	// sentence->dump_words();

	for(int k = crf->_offset_w_label_u;k < crf->_offset_w_label_u + crf->_w_size_label_u;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int t_start = 1;
		int t_end = sentence->size() + 2;
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			double pi_k = (k == crf->_index_w_label_u(yt)) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_label_u(1) == k) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_label_u(0) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_label_u(1) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_label_u(0) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_label_u(1) == k) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", r = " << r << ", index = " << index << ", x_i = " << x_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
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
		grad_w_label[k - crf->_offset_w_label_u] = grad;
	}


	for(int k = 0;k < crf->_w_size_label_u;k++){
		// cout << "k = " << k << ", " << grad_w_label[k] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_label[k] - sgd->_grad_weight[k + crf->_offset_w_label_u]) < 1e-12);
	}

	delete[] grad_w_label;

	grad_w_label = new double[crf->_w_size_label_b];
	for(int k = 0;k < crf->_w_size_label_b;k++){
		grad_w_label[k] = 0;
	}

	for(int k = crf->_offset_w_label_b;k < crf->_offset_w_label_b + crf->_w_size_label_b;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int t_start = 2;
		int t_end = sentence->size() + 2;
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
			double pi_k = (k == crf->_index_w_label_b(yt_1, yt)) ? 1 : 0;
			// cout << "t = " << t << ", s = " << s << ", yt_1 = " << yt_1 << ", yt = " << yt << ", seg = " << sentence->_segments[i] << ", i = " << i << endl;
			double sum_expectation = 0;
			if(t == sentence->size() + 2){
				sum_expectation += (crf->_index_w_label_b(1, 1) == k) ? 1 : 0;
			}else{
				sum_expectation += lattice->_pz_s[t - 1][0][0] * ((crf->_index_w_label_b(0, 0) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][0][1] * ((crf->_index_w_label_b(0, 1) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][0] * ((crf->_index_w_label_b(1, 0) == k) ? 1 : 0);
				sum_expectation += lattice->_pz_s[t - 1][1][1] * ((crf->_index_w_label_b(1, 1) == k) ? 1 : 0);
				// cout << "0-0: " << lattice->_pz_s[t - 1][0][0] << endl;
				// cout << "0-1: " << lattice->_pz_s[t - 1][0][1] << endl;
				// cout << "1-0: " << lattice->_pz_s[t - 1][1][0] << endl;
				// cout << "1-1: " << lattice->_pz_s[t - 1][1][1] << endl;
			}
			grad += pi_k - sum_expectation;
			// cout << "t = " << t << ", r = " << r << ", index = " << index << ", x_i = " << x_i << ", yt_1 = " << yt_1 << ", yt = " << yt << ", pi_k = " << pi_k << ", sum_expectation = " << sum_expectation << endl;
		}

		double log_Zs = log(model->compute_normalizing_constant(sentence));
		double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
		// cout << log_Zs << " == " << log_py << endl;
		crf->_parameter->_all_weights[k] += 1e-8;
		double _log_Zs = log(model->compute_normalizing_constant(sentence));
		double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
		crf->_parameter->_all_weights[k] -= 1e-8;
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
		grad_w_label[k - crf->_offset_w_label_b] = grad;
	}


	for(int k = 0;k < crf->_w_size_label_b;k++){
		// cout << "k = " << k << ", " << grad_w_label[k] << " == " << sgd->_grad_weight[k] << endl;
		assert(std::abs(grad_w_label[k] - sgd->_grad_weight[k + crf->_offset_w_label_b]) < 1e-12);
	}

	
	delete sentence;
	delete var;
}

void test_backward_lambda_0(bool pure_crf_mode){
	Variables* var = new Variables();
	NPYCRF* model = var->model;
	Lattice* lattice = model->_lattice;
	lattice->_clear_word_id_cache();
	lattice->set_pure_crf_mode(pure_crf_mode);
	Sentence* sentence = generate_sentence_4();
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_pw_h, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_pw_h, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];
	lattice->_enumerate_proportional_p_substring_given_sentence(lattice->_pc_s, sentence->size(), lattice->_alpha, lattice->_beta, _Zs);
	lattice->_enumerate_marginal_p_path_given_sentence(lattice->_pz_s, sentence->size(), lattice->_pc_s);

	crf::CRF* crf = var->py_crf->_crf;
	sentence->_features = crf->extract_features(sentence);
	double regularization_constant = 1.0;
	solver::SGD* sgd = new solver::SGD(crf, regularization_constant);
	sgd->_backward_label(sentence, lattice->_pz_s);

	array<int> &character_ids = sentence->_character_ids;
	wchar_t const* characters = sentence->_characters;
	int character_ids_length = sentence->size();

	// sentence->dump_words();
	double grad = 0;

	

	double log_Zs = log(model->compute_normalizing_constant(sentence));
	double log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - log_Zs;
	// cout << log_Zs << " == " << log_py << endl;
	crf->_parameter->_lambda_0 += 1e-8;
	double _log_Zs = log(model->compute_normalizing_constant(sentence));
	double _log_py = model->compute_log_proportional_p_y_given_sentence(sentence) - _log_Zs;
	crf->_parameter->_lambda_0 -= 1e-8;
	// cout << _log_Zs << " == " << _log_py << endl;
	double true_grad = (_log_py - log_py) / 1e-8;
	// cout << "k = " << k << ", " << grad << ", " << true_grad << endl;
	// cout << std::abs(true_grad - grad) << endl;
	if(std::abs(true_grad - grad) >= 1e-4){
		cout << grad << ", " << true_grad << endl;
	}
	assert(std::abs(true_grad - grad) < 1e-4);

	for(int k = 0;k < crf->_w_size_label_u;k++){
		assert(std::abs(grad - sgd->_grad_lambda_0) < 1e-12);
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
	token_ids[SPECIAL_CHARACTER_UNK] = token_ids.size();
	token_ids[SPECIAL_CHARACTER_BEGIN] = token_ids.size();
	token_ids[SPECIAL_CHARACTER_END] = token_ids.size();
	sampler::set_seed(0);

	test_backward_lambda_0(false);
	test_backward_lambda_0(true);
	cout << "OK" << endl;
	test_backward_label(false);
	test_backward_label(true);
	cout << "OK" << endl;
	test_backward_character_type_bigram(false);
	test_backward_character_type_bigram(true);
	cout << "OK" << endl;
	test_backward_character_type_unigram(false);
	test_backward_character_type_unigram(true);
	cout << "OK" << endl;
	test_backward_identical_2(false);
	test_backward_identical_2(true);
	cout << "OK" << endl;
	test_backward_identical_1(false);
	test_backward_identical_1(true);
	cout << "OK" << endl;
	test_backward_bigram(false);
	test_backward_bigram(true);
	cout << "OK" << endl;
	test_backward_unigram(false);
	test_backward_unigram(true);
	cout << "OK" << endl;
}
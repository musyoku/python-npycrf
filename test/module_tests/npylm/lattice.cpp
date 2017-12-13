#include  <iostream>
#include <chrono>
#include "../../../src/npycrf/sampler.h"
#include "../../../src/npycrf/ctype.h"
#include "../../../src/npycrf/array.h"
#include "../../../src/python/npycrf.h"
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

void test_indices(){
	Variables* var = new Variables();
	NPYCRF* model = var->model;
	crf::CRF* crf = var->py_crf->_crf;
	for(int i = 0;i < crf->_weight_size;i++){
		crf->_parameter->_all_weights[i] = 0;
	}

	// label
	for(int y_i = 0;y_i <= 1;y_i++){
		int k = crf->_index_w_label_u(y_i);
		assert(crf->_parameter->_all_weights[k] == 0);
		crf->_parameter->_all_weights[k] += 1;
		for(int y_i_1 = 0;y_i_1 <= 1;y_i_1++){
			int k = crf->_index_w_label_b(y_i_1, y_i);
			assert(crf->_parameter->_all_weights[k] == 0);
			crf->_parameter->_all_weights[k] += 1;
		}
	}

	// unigram
	for(int y_i = 0;y_i <= 1;y_i++){
		for(int x_i = 0;x_i < crf->_num_character_ids;x_i++){
			for(int i = 1;i <= crf->_x_range_unigram;i++){
				int k = crf->_index_w_unigram_u(y_i, i, x_i);
				assert(crf->_parameter->_all_weights[k] == 0);
				crf->_parameter->_all_weights[k] += 1;
				for(int y_i_1 = 0;y_i_1 <= 1;y_i_1++){
					int k = crf->_index_w_unigram_b(y_i_1, y_i, i, x_i);
					assert(crf->_parameter->_all_weights[k] == 0);
					crf->_parameter->_all_weights[k] += 1;
				}
			}
		}
	}

	// bigram
	for(int y_i = 0;y_i <= 1;y_i++){
		for(int x_i = 0;x_i < crf->_num_character_ids;x_i++){
			for(int x_i_1 = 0;x_i_1 < crf->_num_character_ids;x_i_1++){
				for(int i = 1;i <= crf->_x_range_bigram;i++){
					int k = crf->_index_w_bigram_u(y_i, i, x_i_1, x_i);
					assert(crf->_parameter->_all_weights[k] == 0);
					crf->_parameter->_all_weights[k] += 1;
					for(int y_i_1 = 0;y_i_1 <= 1;y_i_1++){
						int k = crf->_index_w_bigram_b(y_i_1, y_i, i, x_i_1, x_i);
						assert(crf->_parameter->_all_weights[k] == 0);
						crf->_parameter->_all_weights[k] += 1;
					}
				}
			}
		}
	}

	// identical
	for(int y_i = 0;y_i <= 1;y_i++){
		for(int i = 1;i <= crf->_x_range_identical_1;i++){
			int k = crf->_index_w_identical_1_u(y_i, i);
			assert(crf->_parameter->_all_weights[k] == 0);
			crf->_parameter->_all_weights[k] += 1;
			for(int y_i_1 = 0;y_i_1 <= 1;y_i_1++){
				int k = crf->_index_w_identical_1_b(y_i_1, y_i, i);
				assert(crf->_parameter->_all_weights[k] == 0);
				crf->_parameter->_all_weights[k] += 1;
			}
		}
	}

	// identical
	for(int y_i = 0;y_i <= 1;y_i++){
		for(int i = 1;i <= crf->_x_range_identical_2;i++){
			int k = crf->_index_w_identical_2_u(y_i, i);
			assert(crf->_parameter->_all_weights[k] == 0);
			crf->_parameter->_all_weights[k] += 1;
			for(int y_i_1 = 0;y_i_1 <= 1;y_i_1++){
				int k = crf->_index_w_identical_2_b(y_i_1, y_i, i);
				assert(crf->_parameter->_all_weights[k] == 0);
				crf->_parameter->_all_weights[k] += 1;
			}
		}
	}

	// unigram type
	for(int y_i = 0;y_i <= 1;y_i++){
		for(int type_i = 0;type_i < crf->_num_character_types;type_i++){
			int k = crf->_index_w_unigram_type_u(y_i, type_i);
			assert(crf->_parameter->_all_weights[k] == 0);
			crf->_parameter->_all_weights[k] += 1;
			for(int y_i_1 = 0;y_i_1 <= 1;y_i_1++){
				int k = crf->_index_w_unigram_type_b(y_i_1, y_i, type_i);
				assert(crf->_parameter->_all_weights[k] == 0);
				crf->_parameter->_all_weights[k] += 1;
			}
		}
	}

	// bigram type
	for(int y_i = 0;y_i <= 1;y_i++){
		for(int type_i = 0;type_i < crf->_num_character_types;type_i++){
			for(int type_i_1 = 0;type_i_1 < crf->_num_character_types;type_i_1++){
				int k = crf->_index_w_bigram_type_u(y_i, type_i_1, type_i);
				assert(crf->_parameter->_all_weights[k] == 0);
				crf->_parameter->_all_weights[k] += 1;
				for(int y_i_1 = 0;y_i_1 <= 1;y_i_1++){
					int k = crf->_index_w_bigram_type_b(y_i_1, y_i, type_i_1, type_i);
					assert(crf->_parameter->_all_weights[k] == 0);
					crf->_parameter->_all_weights[k] += 1;
				}
			}
		}
	}

	for(int i = 0;i < crf->_weight_size;i++){
		assert(crf->_parameter->_all_weights[i] == 1);
	}
	delete var;
}

void assert_test_compute_normalizing_constant(Sentence* sentence, Lattice* lattice, NPYCRF* model){
	double zs_u = lattice->compute_normalizing_constant(sentence, true);
	double log_zs_u = lattice->compute_log_normalizing_constant(sentence, true);
	double zs_n = lattice->compute_normalizing_constant(sentence, false);
	double log_zs_n = lattice->compute_log_normalizing_constant(sentence, false);
	double zs_b = lattice->_compute_normalizing_constant_backward(sentence, lattice->_beta, lattice->_p_transition_tkji);
	double propotional_log_py_x = model->compute_log_proportional_p_y_given_sentence(sentence);
	double log_py_x_u = propotional_log_py_x - log(zs_u);
	double log_py_x_u_ = propotional_log_py_x - log_zs_u;
	double log_py_x_n = propotional_log_py_x - log(zs_n);
	double log_py_x_n_ = propotional_log_py_x - log_zs_n;
	double log_py_x_b = propotional_log_py_x - log(zs_b);
	assert(std::abs(log_py_x_u - log_py_x_n) < 1e-12);
	assert(std::abs(log_py_x_u - log_py_x_b) < 1e-12);
	assert(std::abs(log_py_x_u - log_py_x_u_) < 1e-12);
	assert(std::abs(log_py_x_u - log_py_x_n_) < 1e-12);
}

void test_compute_normalizing_constant(bool pure_crf_mode){
	Variables* var = new Variables();
	NPYCRF* model = var->model;
	Lattice* lattice = model->_lattice;
	lattice->set_pure_crf_mode(pure_crf_mode);

	Sentence* sentence = generate_sentence_1();
	assert_test_compute_normalizing_constant(sentence, lattice, model);
	delete sentence;

	sentence = generate_sentence_2();
	assert_test_compute_normalizing_constant(sentence, lattice, model);
	delete sentence;

	sentence = generate_sentence_5();
	double log_Zs = log(model->compute_normalizing_constant(sentence));

	lattice->_npylm->reserve(sentence->size());	// キャッシュの再確保
	double log_crf = lattice->_crf->compute_log_p_y_given_sentence(sentence);
	double log_npylm = lattice->_npylm->compute_log_p_y_given_sentence(sentence);
	double log_py = pure_crf_mode ? log_crf : log_crf + model->get_lambda_0() * log_npylm;
	
	cout << log_Zs << ", " << log_py << endl;
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
	Sentence* _sentence = sentence->copy();
	_sentence->split(segments);
	_sentence->dump_words();
	delete _sentence;
	for(int i = 0;i < segments.size();i++){
		assert(segments[i] == sentence->_segments[i + 2]);
	}
	delete sentence;
	delete var;
}

void test_scaling(bool pure_crf_mode){
	Variables* var = new Variables();
	Lattice* lattice = var->model->_lattice;
	lattice->set_pure_crf_mode(pure_crf_mode);
	Sentence* sentence = generate_sentence_1();
	double*** alpha;
	double*** beta;
	int seq_capacity = lattice->_max_sentence_length + 1;
	int word_capacity = lattice->_max_word_length + 1;
	lattice::_init_array(alpha, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_init_array(beta, seq_capacity + 1, word_capacity, word_capacity);
	lattice->_clear_p_transition_tkji();
	lattice->_clear_word_id_cache();
	lattice->_enumerate_forward_variables(sentence, alpha, lattice->_p_transition_tkji, NULL, false);
	lattice->_enumerate_backward_variables(sentence, beta, lattice->_p_transition_tkji, NULL, false);

	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_p_transition_tkji, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_p_transition_tkji, lattice->_scaling, true);

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

void test_enumerate_proportional_p_substring_given_sentence(bool pure_crf_mode){
	Variables* var = new Variables();
	Lattice* lattice = var->model->_lattice;
	lattice->set_pure_crf_mode(pure_crf_mode);
	Sentence* sentence = generate_sentence_1();
	double*** alpha;
	double*** beta;
	int seq_capacity = lattice->_max_sentence_length + 1;
	int word_capacity = lattice->_max_word_length + 1;
	lattice->_clear_p_transition_tkji();
	lattice->_clear_word_id_cache();
	lattice::_init_array(alpha, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_init_array(beta, seq_capacity + 1, word_capacity, word_capacity);
	lattice->_enumerate_forward_variables(sentence, alpha, lattice->_p_transition_tkji, NULL, false);
	lattice->_enumerate_backward_variables(sentence, beta, lattice->_p_transition_tkji, NULL, false);
	double Zs = lattice->compute_normalizing_constant(sentence, true);

	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_p_transition_tkji, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_p_transition_tkji, lattice->_scaling, true);
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

void test_enumerate_marginal_p_path_given_sentence(bool pure_crf_mode){
	Variables* var = new Variables();
	Lattice* lattice = var->model->_lattice;
	lattice->set_pure_crf_mode(pure_crf_mode);
	Sentence* sentence = generate_sentence_4();
	double*** alpha;
	double*** beta;
	double** pc_s;
	double*** pz_s;
	int seq_capacity = lattice->_max_sentence_length + 1;
	int word_capacity = lattice->_max_word_length + 1;
	lattice->_clear_p_transition_tkji();
	lattice->_clear_word_id_cache();
	lattice::_init_array(alpha, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_init_array(beta, seq_capacity + 1, word_capacity, word_capacity);
	lattice::_init_array(pc_s, seq_capacity, word_capacity);
	lattice::_init_array(pz_s, seq_capacity + 1, 2, 2);

	lattice->_enumerate_forward_variables(sentence, alpha, lattice->_p_transition_tkji, NULL, false);
	lattice->_enumerate_backward_variables(sentence, beta, lattice->_p_transition_tkji, NULL, false);
	double Zs = lattice->compute_normalizing_constant(sentence, true);
	lattice->_enumerate_forward_variables(sentence, lattice->_alpha, lattice->_p_transition_tkji, lattice->_scaling, true);
	lattice->_enumerate_backward_variables(sentence, lattice->_beta, lattice->_p_transition_tkji, lattice->_scaling, true);
	double _Zs = 1.0 / lattice->_scaling[sentence->size() + 1];

	lattice->_enumerate_marginal_p_substring_given_sentence(pc_s, sentence->size(), alpha, beta, Zs);
	lattice->_enumerate_marginal_p_substring_given_sentence(lattice->_pc_s, sentence->size(), lattice->_alpha, lattice->_beta, _Zs);

	for(int t = 1;t <= sentence->size();t++){
		for(int k = 1;k <= std::min(t, lattice->_max_word_length);k++){
			assert(pc_s[t][k] > 0);
			assert(lattice->_pc_s[t][k] > 0);
			assert(std::abs(lattice->_pc_s[t][k] - pc_s[t][k]) < 1e-12);
		}
	}

	lattice->_enumerate_marginal_p_path_given_sentence_using_p_substring(pz_s, sentence->size(), pc_s);
	lattice->_enumerate_marginal_p_path_given_sentence_using_p_substring(lattice->_pz_s, sentence->size(), lattice->_pc_s);

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
	test_indices();
	cout << "OK" << endl;
	test_compute_normalizing_constant(true);
	test_compute_normalizing_constant(false);
	cout << "OK" << endl;
	// test_viterbi_decode();
	// cout << "OK" << endl;
	test_scaling(true);
	test_scaling(false);
	cout << "OK" << endl;
	test_enumerate_proportional_p_substring_given_sentence(true);
	test_enumerate_proportional_p_substring_given_sentence(false);
	cout << "OK" << endl;
	test_enumerate_marginal_p_path_given_sentence(true);
	test_enumerate_marginal_p_path_given_sentence(false);
	cout << "OK" << endl;
}
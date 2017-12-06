#include  <iostream>
#include <chrono>
#include "../../../src/npycrf/sampler.h"
#include "../../../src/npycrf/ctype.h"
#include "../../../src/npycrf/solver/sgd.h"
#include "../../../src/python/model.h"
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

void test_backward_unigram(){
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
	solver::SGD* sgd = new solver::SGD(lattice, crf);
	sgd->_backward_unigram(sentence);

	int const* character_ids = sentence->_character_ids;
	wchar_t const* characters = sentence->_characters;
	int character_ids_length = sentence->size();

	for(int pos = 1;pos <= crf->_x_range_unigram;pos++){
		for(int x_i = 0;x_i < token_ids.size();x_i++){
			cout << "index = " << crf->_index_w_unigram_u(0, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
			cout << "index = " << crf->_index_w_unigram_u(1, pos, x_i) << ", pos = " << pos << ", x_i = " << x_i << endl;
			assert(pos == (crf->_index_w_unigram_u(0, pos, x_i) % (crf->_x_range_unigram * 2)) / 2 + 1);
			assert(pos == (crf->_index_w_unigram_u(1, pos, x_i) % (crf->_x_range_unigram * 2)) / 2 + 1);
		}
	}

	double* grad_w_label = new double[crf->_w_size_unigram_u + crf->_w_size_unigram_b];
	for(int k = 0;k < crf->_w_size_unigram_u + crf->_w_size_unigram_b;k++){
		grad_w_label[k] = 0;
	}

	sentence->dump_words();

	for(int k = 0;k < crf->_w_size_unigram_u;k++){
		double grad = 0;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		int pos = (k % (crf->_x_range_unigram * 2)) / 2 + 1;
		int t_start = std::max(1, -(crf->_x_unigram_start + pos - 1) + 1);
		int t_end = std::min(sentence->size() + 2, sentence->size() + 2 - (crf->_x_unigram_start + pos - 1));
		// cout << "t_start = " << t_start << ", t_end = " << t_end << endl;
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
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
		grad_w_label[k] = grad;
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
		for(int t = t_start;t <= t_end;t++){
			int yt_1 = sentence->get_crf_label_at(t - 1);
			int yt = sentence->get_crf_label_at(t);
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
		grad_w_label[k] = grad;
	}

	for(int k = 0;k < crf->_w_size_unigram_u + crf->_w_size_unigram_b;k++){
		cout << "k = " << k << ", " << grad_w_label[k] << " == " << sgd->_grad_w_unigram[k] << endl;
		assert(std::abs(grad_w_label[k] - sgd->_grad_w_unigram[k]) < 1e-12);
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
	test_backward_unigram();
	cout << "OK" << endl;
}
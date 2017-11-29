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

class Variables {
public:
	Sentence* sentence;
	Model* model;
	int* character_ids;
	python::model::NPYLM* py_npylm;
	python::model::CRF* py_crf;
	Variables(){
		std::unordered_map<wchar_t, int> _token_ids;
		std::wstring sentence_str = L"ああいいいううううえええおおああいいいううううえええおおああいいいううううえええおおああいいいううううえええおおああいいいううううえええおお";
		for(wchar_t character: sentence_str){
			auto itr = _token_ids.find(character);
			if(itr == _token_ids.end()){
				_token_ids[character] = _token_ids.size();
			}
		}
		character_ids = new int[sentence_str.size()];
		for(int i = 0;i < sentence_str.size();i++){
			character_ids[i] = _token_ids[sentence_str[i]];
		}
		sentence = new Sentence(sentence_str, character_ids);
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
		py_npylm = new python::model::NPYLM(max_word_length, max_sentence_length, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);

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
		py_crf = new python::model::CRF(num_character_ids, num_character_types, feature_x_unigram_start, feature_x_unigram_end, feature_x_bigram_start, feature_x_bigram_end, feature_x_identical_1_start, feature_x_identical_1_end, feature_x_identical_2_start, feature_x_identical_2_end);

		model = new Model(py_npylm, py_crf, lambda_0, max_word_length, sentence->size());
		Lattice* lattice = model->_lattice;
		npylm::NPYLM* npylm = model->_npylm;
		npylm_add_customers(npylm, sentence);
		lattice->reserve(max_word_length, sentence->size());
	}
	~Variables(){
		delete[] character_ids;
		delete sentence;
		delete model;
		delete py_npylm;
		delete py_crf;
	}
};

void test_compute_marginal_p_x(){
	Variables* var = new Variables();
	Lattice* lattice = var->model->_lattice;
	Sentence* sentence = var->sentence;

	double px_u = lattice->compute_marginal_p_x(sentence, true);
	double px_n = lattice->compute_marginal_p_x(sentence, false);
	assert(std::abs(px_u - px_n) < 1e-16);

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
}
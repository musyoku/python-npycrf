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
}
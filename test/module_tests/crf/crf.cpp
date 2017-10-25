#include <unordered_map>
#include <iostream>
#include <cassert>
#include "../../../src/npycrf/sentence.h"
#include "../../../src/npycrf/crf/crf.h"

using namespace npycrf::crf;
using namespace npycrf;
using std::cout;
using std::flush;
using std::endl;

void test_init(){
	int num_character_ids = 100;
	int num_character_types = 281;
	int feature_x_unigram_start = -2;
	int feature_x_unigram_end = 2;
	int feature_x_bigram_start = -2;
	int feature_x_bigram_end = 1;
	int feature_x_identical_1_start = -2;
	int feature_x_identical_1_end = 1;
	int feature_x_identical_2_start = -3;
	int feature_x_identical_2_end = 1;
	CRF* crf = new CRF(num_character_ids,
					   num_character_types,
					   feature_x_unigram_start,
					   feature_x_unigram_end,
					   feature_x_bigram_start,
					   feature_x_bigram_end,
					   feature_x_identical_1_start,
					   feature_x_identical_1_end,
					   feature_x_identical_2_start,
					   feature_x_identical_2_end
	);

	double value = 1.5;
	for(int y_i = 0;y_i < 2;y_i++){
		for(int x_i = 0;x_i < num_character_ids;x_i++){
			for(int i = 0;i < feature_x_unigram_end - feature_x_unigram_start + 1;i++){
				crf->set_w_unigram_u(y_i, i, x_i, value);
				for(int y_i_1 = 0;y_i_1 < 2;y_i_1++){
					crf->set_w_unigram_b(y_i_1, y_i, i, x_i, value);
				}
			}
			for(int i = 0;i < feature_x_bigram_end - feature_x_bigram_start + 1;i++){
				for(int x_i_1 = 0;x_i_1 < num_character_ids;x_i_1++){
					crf->set_w_bigram_u(y_i, i, x_i_1, x_i, value);
					for(int y_i_1 = 0;y_i_1 < 2;y_i_1++){
						crf->set_w_bigram_b(y_i_1, y_i, i, x_i_1, x_i, value);
					}
				}
			}
			
		}
		for(int i = 0;i < feature_x_identical_1_end - feature_x_identical_1_start + 1;i++){
			crf->set_w_identical_1_u(y_i, i, value);
			for(int y_i_1 = 0;y_i_1 < 2;y_i_1++){
				crf->set_w_identical_1_b(y_i_1, y_i, i, value);
			}
		}
		for(int i = 0;i < feature_x_identical_2_end - feature_x_identical_2_start + 1;i++){
			crf->set_w_identical_2_u(y_i, i, value);
			for(int y_i_1 = 0;y_i_1 < 2;y_i_1++){
				crf->set_w_identical_2_b(y_i_1, y_i, i, value);
			}
		}
		for(int type_i = 0;type_i < num_character_types;type_i++){
			crf->set_w_unigram_type_u(y_i, type_i, value);
			for(int y_i_1 = 0;y_i_1 < 2;y_i_1++){
				crf->set_w_unigram_type_b(y_i_1, y_i, type_i, value);
			}
			for(int type_i_1 = 0;type_i_1 < num_character_types;type_i_1++){
				crf->set_w_bigram_type_u(y_i, type_i_1, type_i, value);
				for(int y_i_1 = 0;y_i_1 < 2;y_i_1++){
					crf->set_w_bigram_type_b(y_i_1, y_i, type_i_1, type_i, value);
				}
			}
		}
	}

	for(int i = 0;i < crf->_w_size_unigram_u;i++){
		assert(crf->_w_unigram_u[i] == value);
	}
	for(int i = 0;i < crf->_w_size_unigram_b;i++){
		assert(crf->_w_unigram_b[i] == value);
	}
	for(int i = 0;i < crf->_w_size_bigram_u;i++){
		assert(crf->_w_bigram_u[i] == value);
	}
	for(int i = 0;i < crf->_w_size_bigram_b;i++){
		assert(crf->_w_bigram_b[i] == value);
	}
	for(int i = 0;i < crf->_w_size_identical_1_u;i++){
		assert(crf->_w_identical_1_u[i] == value);
	}
	for(int i = 0;i < crf->_w_size_identical_1_b;i++){
		assert(crf->_w_identical_1_b[i] == value);
	}
	for(int i = 0;i < crf->_w_size_identical_2_u;i++){
		assert(crf->_w_identical_2_u[i] == value);
	}
	for(int i = 0;i < crf->_w_size_identical_2_b;i++){
		assert(crf->_w_identical_2_b[i] == value);
	}
	for(int i = 0;i < crf->_w_size_unigram_type_u;i++){
		assert(crf->_w_unigram_type_u[i] == value);
	}
	for(int i = 0;i < crf->_w_size_unigram_type_b;i++){
		assert(crf->_w_unigram_type_b[i] == value);
	}
	for(int i = 0;i < crf->_w_size_bigram_type_u;i++){
		assert(crf->_w_bigram_type_u[i] == value);
	}
	for(int i = 0;i < crf->_w_size_bigram_type_b;i++){
		assert(crf->_w_bigram_type_b[i] == value);
	}
	delete crf;
}

void test_compute_path_cost(){
	std::unordered_map<wchar_t, int> _token_ids;
	std::wstring sentence_str = L"あいうえおかきくけこさしすせそ";
	for(wchar_t character: sentence_str){
		_token_ids[character] = _token_ids.size();
	}

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
	CRF* crf = new CRF(num_character_ids,
					   num_character_types,
					   feature_x_unigram_start,
					   feature_x_unigram_end,
					   feature_x_bigram_start,
					   feature_x_bigram_end,
					   feature_x_identical_1_start,
					   feature_x_identical_1_end,
					   feature_x_identical_2_start,
					   feature_x_identical_2_end
	);
	int* character_ids = new int[sentence_str.size()];
	for(int i = 0;i < sentence_str.size();i++){
		character_ids[i] = _token_ids[sentence_str[i]];
	}
	for(int i = 0;i < crf->_x_range_unigram;i++){
		for(int x_i = 0;x_i < _token_ids.size();x_i++){
			crf->set_w_unigram_u(0, i, x_i, x_i);
			crf->set_w_unigram_b(0, 0, i, x_i, x_i * 2);
		}
	}
	for(int i = 2;i <= sentence_str.size();i++){
		double cost = crf->_compute_cost_unigram_features(character_ids, sentence_str.size(), i, 0, 0);
		double true_cost = 0;
		for(int t = std::max(0, i - crf->_x_range_unigram);t < i;t++){
			true_cost += character_ids[t] * 3;
		}
		assert(cost == true_cost);
	}
	for(int i = 0;i < crf->_x_range_bigram;i++){
		for(int x_i = 0;x_i < _token_ids.size();x_i++){
			for(int x_i_1 = 0;x_i_1 < _token_ids.size();x_i_1++){
				crf->set_w_bigram_u(0, i, x_i_1, x_i, x_i * x_i_1);
				crf->set_w_bigram_b(0, 0, i, x_i_1, x_i, x_i * x_i_1 * 2);
			}
		}
	}
	for(int i = 2;i <= sentence_str.size();i++){
		double cost = crf->_compute_cost_bigram_features(character_ids, sentence_str.size(), i, 0, 0);
		double true_cost = 0;
		for(int t = std::max(1, i - crf->_x_range_bigram);t < i;t++){
			true_cost += character_ids[t] * character_ids[t - 1] * 3;
		}
		assert(cost == true_cost);
	}
	
	delete crf;
	delete[] character_ids;
}

int main(){
	test_init();
	cout << "OK" << endl;
	test_compute_path_cost();
	cout << "OK" << endl;
	return 0;
}
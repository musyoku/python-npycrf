#include <iostream>
#include "sgd.h"
#include "../ctype.h"

namespace npycrf {
	namespace solver {
		SGD::SGD(crf::CRF* crf, double regularization_constant){
			_crf = crf;
			_regularization_constant = regularization_constant;
			_grad_weight = new double[crf->_parameter->_weight_size];
			clear_grads();
		}
		SGD::~SGD(){
			delete[] _grad_weight;
		}
		void SGD::clear_grads(){
			for(int i = 0;i < _crf->_parameter->_weight_size;i++){
				_grad_weight[i] = 0;
			}
		}
		void SGD::update(double learning_rate){
			crf::Parameter* params = _crf->_parameter;
			assert(params->_all_weights != NULL);
			for(int i = 0;i < params->_weight_size;i++){
				params->_all_weights[i] += learning_rate * _grad_weight[i] - _regularization_constant * learning_rate * params->_all_weights[i];
				if(_grad_weight[i] != 0){
					params->_num_updates[i] += 1;
				}
			}
		}
		// CRFの勾配計算について
		// http://www.ism.ac.jp/editsec/toukei/pdf/64-2-179.pdf
		void SGD::backward(Sentence* sentence, double*** pz_s){
			_backward_unigram(sentence, pz_s);
			_backward_bigram(sentence, pz_s);
			_backward_identical_1(sentence, pz_s);
			_backward_unigram_type(sentence, pz_s);
			_backward_bigram_type(sentence, pz_s);
			_backward_label(sentence, pz_s);
		}
		void SGD::_backward_unigram(Sentence* sentence, double*** pz_s){
			array<int> &character_ids = sentence->_character_ids;
			wchar_t const* characters = sentence->_characters;
			int character_ids_length = sentence->size();
			for(int i = 1;i <= sentence->size() + 2;i++){
				// ラベルを取得
				int y_i_1 = sentence->get_crf_label_at(i - 1);
				int y_i = sentence->get_crf_label_at(i);

				int r_start = std::max(1, i + _crf->_x_unigram_start);
				int r_end = std::min(character_ids_length + 2, i + _crf->_x_unigram_end);	// <eos>2つを考慮
				for(int r = r_start;r <= r_end;r++){
					int pos = r - i - _crf->_x_unigram_start + 1;	// [1, _x_range_unigram]
					int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;

					// 発火
					int k_u = _crf->_index_w_unigram_u(y_i, pos, x_i);
					_grad_weight[k_u] += 1;
					int k_b = _crf->_index_w_unigram_b(y_i_1, y_i, pos, x_i);
					_grad_weight[k_b] += 1;

					// 発火の期待値
					int k_0 = _crf->_index_w_unigram_u(0, pos, x_i);
					int k_1 = _crf->_index_w_unigram_u(1, pos, x_i);
					_grad_weight[k_0] -= pz_s[i - 1][0][0];
					_grad_weight[k_0] -= pz_s[i - 1][1][0];
					_grad_weight[k_1] -= pz_s[i - 1][0][1];
					_grad_weight[k_1] -= pz_s[i - 1][1][1];

					int k_0_0 = _crf->_index_w_unigram_b(0, 0, pos, x_i);
					int k_0_1 = _crf->_index_w_unigram_b(0, 1, pos, x_i);
					int k_1_0 = _crf->_index_w_unigram_b(1, 0, pos, x_i);
					int k_1_1 = _crf->_index_w_unigram_b(1, 1, pos, x_i);
					_grad_weight[k_0_0] -= pz_s[i - 1][0][0];
					_grad_weight[k_0_1] -= pz_s[i - 1][0][1];
					_grad_weight[k_1_0] -= pz_s[i - 1][1][0];
					_grad_weight[k_1_1] -= pz_s[i - 1][1][1];
				}
			}
		}
		void SGD::_backward_bigram(Sentence* sentence, double*** pz_s){
			array<int> &character_ids = sentence->_character_ids;
			wchar_t const* characters = sentence->_characters;
			int character_ids_length = sentence->size();
			for(int i = 1;i <= sentence->size() + 2;i++){
				// ラベルを取得
				int y_i_1 = sentence->get_crf_label_at(i - 1);
				int y_i = sentence->get_crf_label_at(i);

				int r_start = std::max(2, i + _crf->_x_bigram_start);
				int r_end = std::min(character_ids_length + 2, i + _crf->_x_bigram_end);
				for(int r = r_start;r <= r_end;r++){
					int pos = r - i - _crf->_x_unigram_start + 1;	// [1, _x_range_bigram]
					int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
					int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : SPECIAL_CHARACTER_END;

					// 発火
					int k_u = _crf->_index_w_bigram_u(y_i, pos, x_i_1, x_i);
					_grad_weight[k_u] += 1;
					int k_b = _crf->_index_w_bigram_b(y_i_1, y_i, pos, x_i_1, x_i);
					_grad_weight[k_b] += 1;

					// 発火の期待値
					int k_0 = _crf->_index_w_bigram_u(0, pos, x_i_1, x_i);
					int k_1 = _crf->_index_w_bigram_u(1, pos, x_i_1, x_i);
					_grad_weight[k_0] -= pz_s[i - 1][0][0];
					_grad_weight[k_0] -= pz_s[i - 1][1][0];
					_grad_weight[k_1] -= pz_s[i - 1][0][1];
					_grad_weight[k_1] -= pz_s[i - 1][1][1];

					int k_0_0 = _crf->_index_w_bigram_b(0, 0, pos, x_i_1, x_i);
					int k_0_1 = _crf->_index_w_bigram_b(0, 1, pos, x_i_1, x_i);
					int k_1_0 = _crf->_index_w_bigram_b(1, 0, pos, x_i_1, x_i);
					int k_1_1 = _crf->_index_w_bigram_b(1, 1, pos, x_i_1, x_i);
					_grad_weight[k_0_0] -= pz_s[i - 1][0][0];
					_grad_weight[k_0_1] -= pz_s[i - 1][0][1];
					_grad_weight[k_1_0] -= pz_s[i - 1][1][0];
					_grad_weight[k_1_1] -= pz_s[i - 1][1][1];
				}
			}
		}
		void SGD::_backward_identical_1(Sentence* sentence, double*** pz_s){
			array<int> &character_ids = sentence->_character_ids;
			wchar_t const* characters = sentence->_characters;
			int character_ids_length = sentence->size();
			for(int i = 1;i <= sentence->size() + 2;i++){
				// ラベルを取得
				int y_i_1 = sentence->get_crf_label_at(i - 1);
				int y_i = sentence->get_crf_label_at(i);

				int r_start = std::max(2, i + _crf->_x_identical_1_start);
				int r_end = std::min(character_ids_length + 2, i + _crf->_x_identical_1_end);
				for(int r = r_start;r <= r_end;r++){
					int pos = r - i - _crf->_x_identical_1_start + 1;	// [1, _x_range_identical_1]
					int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
					int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : SPECIAL_CHARACTER_END;
					if(x_i == x_i_1){
						// 発火
						int k_u = _crf->_index_w_identical_1_u(y_i, pos);
						_grad_weight[k_u] += 1;
						int k_b = _crf->_index_w_identical_1_b(y_i_1, y_i, pos);
						_grad_weight[k_b] += 1;

						// 発火の期待値
						int k_0 = _crf->_index_w_identical_1_u(0, pos);
						int k_1 = _crf->_index_w_identical_1_u(1, pos);
						_grad_weight[k_0] -= pz_s[i - 1][0][0];
						_grad_weight[k_0] -= pz_s[i - 1][1][0];
						_grad_weight[k_1] -= pz_s[i - 1][0][1];
						_grad_weight[k_1] -= pz_s[i - 1][1][1];

						int k_0_0 = _crf->_index_w_identical_1_b(0, 0, pos);
						int k_0_1 = _crf->_index_w_identical_1_b(0, 1, pos);
						int k_1_0 = _crf->_index_w_identical_1_b(1, 0, pos);
						int k_1_1 = _crf->_index_w_identical_1_b(1, 1, pos);
						_grad_weight[k_0_0] -= pz_s[i - 1][0][0];
						_grad_weight[k_0_1] -= pz_s[i - 1][0][1];
						_grad_weight[k_1_0] -= pz_s[i - 1][1][0];
						_grad_weight[k_1_1] -= pz_s[i - 1][1][1];
					}
				}
			}
		}
		void SGD::_backward_identical_2(Sentence* sentence, double*** pz_s){
			array<int> &character_ids = sentence->_character_ids;
			wchar_t const* characters = sentence->_characters;
			int character_ids_length = sentence->size();
			for(int i = 1;i <= sentence->size() + 2;i++){
				// ラベルを取得
				int y_i_1 = sentence->get_crf_label_at(i - 1);
				int y_i = sentence->get_crf_label_at(i);

				int r_start = std::max(3, i + _crf->_x_identical_2_start);
				int r_end = std::min(character_ids_length + 2, i + _crf->_x_identical_2_end);
				for(int r = r_start;r <= r_end;r++){
					int pos = r - i - _crf->_x_identical_2_start + 1;	// [1, _x_range_identical_2]
					int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
					int x_i_2 = (r - 2 <= character_ids_length) ? character_ids[r - 3] : SPECIAL_CHARACTER_END;
					if(x_i == x_i_2){
						// 発火
						int k_u = _crf->_index_w_identical_2_u(y_i, pos);
						_grad_weight[k_u] += 1;
						int k_b = _crf->_index_w_identical_2_b(y_i_1, y_i, pos);
						_grad_weight[k_b] += 1;

						// 発火の期待値
						int k_0 = _crf->_index_w_identical_2_u(0, pos);
						int k_1 = _crf->_index_w_identical_2_u(1, pos);
						_grad_weight[k_0] -= pz_s[i - 1][0][0];
						_grad_weight[k_0] -= pz_s[i - 1][1][0];
						_grad_weight[k_1] -= pz_s[i - 1][0][1];
						_grad_weight[k_1] -= pz_s[i - 1][1][1];

						int k_0_0 = _crf->_index_w_identical_2_b(0, 0, pos);
						int k_0_1 = _crf->_index_w_identical_2_b(0, 1, pos);
						int k_1_0 = _crf->_index_w_identical_2_b(1, 0, pos);
						int k_1_1 = _crf->_index_w_identical_2_b(1, 1, pos);
						_grad_weight[k_0_0] -= pz_s[i - 1][0][0];
						_grad_weight[k_0_1] -= pz_s[i - 1][0][1];
						_grad_weight[k_1_0] -= pz_s[i - 1][1][0];
						_grad_weight[k_1_1] -= pz_s[i - 1][1][1];
					}
				}
			}
		}
		void SGD::_backward_unigram_type(Sentence* sentence, double*** pz_s){
			array<int> &character_ids = sentence->_character_ids;
			wchar_t const* characters = sentence->_characters;
			int character_ids_length = sentence->size();
			for(int i = 1;i <= sentence->size() + 2;i++){
				// ラベルを取得
				int y_i_1 = sentence->get_crf_label_at(i - 1);
				int y_i = sentence->get_crf_label_at(i);
				int type_i = (i <= character_ids_length) ? ctype::get_type(characters[i - 1]) : CTYPE_UNKNOWN;

				// 発火
				int k_u = _crf->_index_w_unigram_type_u(y_i, type_i);
				_grad_weight[k_u] += 1;
				int k_b = _crf->_index_w_unigram_type_b(y_i_1, y_i, type_i);
				_grad_weight[k_b] += 1;

				// 発火の期待値
				int k_0 = _crf->_index_w_unigram_type_u(0, type_i);
				int k_1 = _crf->_index_w_unigram_type_u(1, type_i);
				_grad_weight[k_0] -= pz_s[i - 1][0][0];
				_grad_weight[k_0] -= pz_s[i - 1][1][0];
				_grad_weight[k_1] -= pz_s[i - 1][0][1];
				_grad_weight[k_1] -= pz_s[i - 1][1][1];

				int k_0_0 = _crf->_index_w_unigram_type_b(0, 0, type_i);
				int k_0_1 = _crf->_index_w_unigram_type_b(0, 1, type_i);
				int k_1_0 = _crf->_index_w_unigram_type_b(1, 0, type_i);
				int k_1_1 = _crf->_index_w_unigram_type_b(1, 1, type_i);
				_grad_weight[k_0_0] -= pz_s[i - 1][0][0];
				_grad_weight[k_0_1] -= pz_s[i - 1][0][1];
				_grad_weight[k_1_0] -= pz_s[i - 1][1][0];
				_grad_weight[k_1_1] -= pz_s[i - 1][1][1];
			}
		}
		void SGD::_backward_bigram_type(Sentence* sentence, double*** pz_s){
			array<int> &character_ids = sentence->_character_ids;
			wchar_t const* characters = sentence->_characters;
			int character_ids_length = sentence->size();
			for(int i = 2;i <= sentence->size() + 2;i++){
				// ラベルを取得
				int y_i_1 = sentence->get_crf_label_at(i - 1);
				int y_i = sentence->get_crf_label_at(i);
				int type_i = (i <= character_ids_length) ? ctype::get_type(characters[i - 1]) : CTYPE_UNKNOWN;
				int type_i_1 = (i - 1 <= character_ids_length) ? ctype::get_type(characters[i - 2]) : CTYPE_UNKNOWN;

				// 発火
				int k_u = _crf->_index_w_bigram_type_u(y_i, type_i_1, type_i);
				_grad_weight[k_u] += 1;
				int k_b = _crf->_index_w_bigram_type_b(y_i_1, y_i, type_i_1, type_i);
				_grad_weight[k_b] += 1;

				// 発火の期待値
				int k_0 = _crf->_index_w_bigram_type_u(0, type_i_1, type_i);
				int k_1 = _crf->_index_w_bigram_type_u(1, type_i_1, type_i);
				_grad_weight[k_0] -= pz_s[i - 1][0][0];
				_grad_weight[k_0] -= pz_s[i - 1][1][0];
				_grad_weight[k_1] -= pz_s[i - 1][0][1];
				_grad_weight[k_1] -= pz_s[i - 1][1][1];

				int k_0_0 = _crf->_index_w_bigram_type_b(0, 0, type_i_1, type_i);
				int k_0_1 = _crf->_index_w_bigram_type_b(0, 1, type_i_1, type_i);
				int k_1_0 = _crf->_index_w_bigram_type_b(1, 0, type_i_1, type_i);
				int k_1_1 = _crf->_index_w_bigram_type_b(1, 1, type_i_1, type_i);
				_grad_weight[k_0_0] -= pz_s[i - 1][0][0];
				_grad_weight[k_0_1] -= pz_s[i - 1][0][1];
				_grad_weight[k_1_0] -= pz_s[i - 1][1][0];
				_grad_weight[k_1_1] -= pz_s[i - 1][1][1];
			}
		}
		void SGD::_backward_label(Sentence* sentence, double*** pz_s){
			array<int> &character_ids = sentence->_character_ids;
			wchar_t const* characters = sentence->_characters;
			int character_ids_length = sentence->size();
			for(int i = 2;i <= sentence->size() + 2;i++){
				// ラベルを取得
				int y_i_1 = sentence->get_crf_label_at(i - 1);
				int y_i = sentence->get_crf_label_at(i);

				// 発火
				int k_u = _crf->_index_w_label_u(y_i);
				_grad_weight[k_u] += 1;
				int k_b = _crf->_index_w_label_b(y_i_1, y_i);
				_grad_weight[k_b] += 1;

				// 発火の期待値
				int k_0 = _crf->_index_w_label_u(0);
				int k_1 = _crf->_index_w_label_u(1);
				_grad_weight[k_0] -= pz_s[i - 1][0][0];
				_grad_weight[k_0] -= pz_s[i - 1][1][0];
				_grad_weight[k_1] -= pz_s[i - 1][0][1];
				_grad_weight[k_1] -= pz_s[i - 1][1][1];

				int k_0_0 = _crf->_index_w_label_b(0, 0);
				int k_0_1 = _crf->_index_w_label_b(0, 1);
				int k_1_0 = _crf->_index_w_label_b(1, 0);
				int k_1_1 = _crf->_index_w_label_b(1, 1);
				_grad_weight[k_0_0] -= pz_s[i - 1][0][0];
				_grad_weight[k_0_1] -= pz_s[i - 1][0][1];
				_grad_weight[k_1_0] -= pz_s[i - 1][1][0];
				_grad_weight[k_1_1] -= pz_s[i - 1][1][1];
			}
		}
	}
}
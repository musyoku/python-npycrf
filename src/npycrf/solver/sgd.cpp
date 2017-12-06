#include <iostream>
#include "sgd.h"
#include "../ctype.h"

namespace npycrf {
	namespace solver {
		SGD::SGD(Lattice* lattice, crf::CRF* crf){
			_lattice = lattice;
			_crf = crf;
			_grad_w_label = new double[crf->_w_size_label_u + crf->_w_size_label_b];
			_grad_w_unigram = new double[crf->_w_size_unigram_u + crf->_w_size_unigram_b];
			_grad_w_bigram = new double[crf->_w_size_bigram_u + crf->_w_size_bigram_b];
			_grad_w_identical_1 = new double[crf->_w_size_identical_1_u + crf->_w_size_identical_1_b];
			_grad_w_identical_2 = new double[crf->_w_size_identical_2_u + crf->_w_size_identical_2_b];
			_grad_w_unigram_type = new double[crf->_w_size_unigram_type_u + crf->_w_size_unigram_type_b];
			_grad_w_bigram_type = new double[crf->_w_size_bigram_type_u + crf->_w_size_bigram_type_b];
			clear_grads();
		}
		SGD::~SGD(){
			delete[] _grad_w_label;
			delete[] _grad_w_unigram;
			delete[] _grad_w_bigram;
			delete[] _grad_w_identical_1;
			delete[] _grad_w_identical_2;
			delete[] _grad_w_unigram_type;
			delete[] _grad_w_bigram_type;
		}
		void SGD::clear_grads(){
			for(int i = 0;i < _crf->_w_size_label_u + _crf->_w_size_label_b;i++){
				_grad_w_label[i] = 0;
			}
			for(int i = 0;i < _crf->_w_size_unigram_u + _crf->_w_size_unigram_b;i++){
				_grad_w_unigram[i] = 0;
			}
			for(int i = 0;i < _crf->_w_size_bigram_u + _crf->_w_size_bigram_b;i++){
				_grad_w_bigram[i] = 0;
			}
			for(int i = 0;i < _crf->_w_size_identical_1_u + _crf->_w_size_identical_1_b;i++){
				_grad_w_identical_1[i] = 0;
			}
			for(int i = 0;i < _crf->_w_size_identical_2_u + _crf->_w_size_identical_2_b;i++){
				_grad_w_identical_2[i] = 0;
			}
			for(int i = 0;i < _crf->_w_size_unigram_type_u + _crf->_w_size_unigram_type_b;i++){
				_grad_w_unigram_type[i] = 0;
			}
			for(int i = 0;i < _crf->_w_size_bigram_type_u + _crf->_w_size_bigram_type_b;i++){
				_grad_w_bigram_type[i] = 0;
			}
		}
		// CRFの勾配計算について
		// http://www.ism.ac.jp/editsec/toukei/pdf/64-2-179.pdf
		void SGD::backward(Sentence* sentence){
			_backward_unigram(sentence);
			_backward_bigram(sentence);
			_backward_identical_1(sentence);
			_backward_unigram_type(sentence);
			_backward_bigram_type(sentence);
		}
		void SGD::_backward_unigram(Sentence* sentence){
			int const* character_ids = sentence->_character_ids;
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
					int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;

					// 発火
					int k_u = _crf->_index_w_unigram_u(y_i, pos, x_i);
					_grad_w_unigram[k_u] += 1;
					int k_b = _crf->_index_w_unigram_b(y_i_1, y_i, pos, x_i);
					_grad_w_unigram[k_b] += 1;

					// 発火の期待値
					int k_0 = _crf->_index_w_unigram_u(0, pos, x_i);
					int k_1 = _crf->_index_w_unigram_u(1, pos, x_i);
					_grad_w_unigram[k_0] -= _lattice->_pz_s[i - 1][0][0];
					_grad_w_unigram[k_0] -= _lattice->_pz_s[i - 1][1][0];
					_grad_w_unigram[k_1] -= _lattice->_pz_s[i - 1][0][1];
					_grad_w_unigram[k_1] -= _lattice->_pz_s[i - 1][1][1];

					int k_0_0 = _crf->_index_w_unigram_b(0, 0, pos, x_i);
					int k_0_1 = _crf->_index_w_unigram_b(0, 1, pos, x_i);
					int k_1_0 = _crf->_index_w_unigram_b(1, 0, pos, x_i);
					int k_1_1 = _crf->_index_w_unigram_b(1, 1, pos, x_i);
					_grad_w_unigram[k_0_0] -= _lattice->_pz_s[i - 1][0][0];
					_grad_w_unigram[k_0_1] -= _lattice->_pz_s[i - 1][0][1];
					_grad_w_unigram[k_1_0] -= _lattice->_pz_s[i - 1][1][0];
					_grad_w_unigram[k_1_1] -= _lattice->_pz_s[i - 1][1][1];
				}
			}
		}
		void SGD::_backward_bigram(Sentence* sentence){
			int const* character_ids = sentence->_character_ids;
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
					int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
					int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : CHARACTER_ID_EOS;

					// 発火
					int k_u = _crf->_index_w_bigram_u(y_i, pos, x_i_1, x_i);
					_grad_w_bigram[k_u] += 1;
					int k_b = _crf->_index_w_bigram_b(y_i_1, y_i, pos, x_i_1, x_i);
					_grad_w_bigram[k_b] += 1;

					// 発火の期待値
					int k_0 = _crf->_index_w_bigram_u(0, pos, x_i_1, x_i);
					int k_1 = _crf->_index_w_bigram_u(1, pos, x_i_1, x_i);
					_grad_w_bigram[k_0] -= _lattice->_pz_s[i - 1][0][0];
					_grad_w_bigram[k_0] -= _lattice->_pz_s[i - 1][1][0];
					_grad_w_bigram[k_1] -= _lattice->_pz_s[i - 1][0][1];
					_grad_w_bigram[k_1] -= _lattice->_pz_s[i - 1][1][1];

					int k_0_0 = _crf->_index_w_bigram_b(0, 0, pos, x_i_1, x_i);
					int k_0_1 = _crf->_index_w_bigram_b(0, 1, pos, x_i_1, x_i);
					int k_1_0 = _crf->_index_w_bigram_b(1, 0, pos, x_i_1, x_i);
					int k_1_1 = _crf->_index_w_bigram_b(1, 1, pos, x_i_1, x_i);
					_grad_w_bigram[k_0_0] -= _lattice->_pz_s[i - 1][0][0];
					_grad_w_bigram[k_0_1] -= _lattice->_pz_s[i - 1][0][1];
					_grad_w_bigram[k_1_0] -= _lattice->_pz_s[i - 1][1][0];
					_grad_w_bigram[k_1_1] -= _lattice->_pz_s[i - 1][1][1];
				}
			}
		}
		void SGD::_backward_identical_1(Sentence* sentence){
			int const* character_ids = sentence->_character_ids;
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
					int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
					int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : CHARACTER_ID_EOS;
					if(x_i == x_i_1){
						// 発火
						int k_u = _crf->_index_w_identical_1_u(y_i, pos);
						_grad_w_identical_1[k_u] += 1;
						int k_b = _crf->_index_w_identical_1_b(y_i_1, y_i, pos);
						_grad_w_identical_1[k_b] += 1;

						// 発火の期待値
						int k_0 = _crf->_index_w_identical_1_u(0, pos);
						int k_1 = _crf->_index_w_identical_1_u(1, pos);
						_grad_w_identical_1[k_0] -= _lattice->_pz_s[i - 1][0][0];
						_grad_w_identical_1[k_0] -= _lattice->_pz_s[i - 1][1][0];
						_grad_w_identical_1[k_1] -= _lattice->_pz_s[i - 1][0][1];
						_grad_w_identical_1[k_1] -= _lattice->_pz_s[i - 1][1][1];

						int k_0_0 = _crf->_index_w_identical_1_b(0, 0, pos);
						int k_0_1 = _crf->_index_w_identical_1_b(0, 1, pos);
						int k_1_0 = _crf->_index_w_identical_1_b(1, 0, pos);
						int k_1_1 = _crf->_index_w_identical_1_b(1, 1, pos);
						_grad_w_identical_1[k_0_0] -= _lattice->_pz_s[i - 1][0][0];
						_grad_w_identical_1[k_0_1] -= _lattice->_pz_s[i - 1][0][1];
						_grad_w_identical_1[k_1_0] -= _lattice->_pz_s[i - 1][1][0];
						_grad_w_identical_1[k_1_1] -= _lattice->_pz_s[i - 1][1][1];
					}
				}
			}
		}
		void SGD::_backward_identical_2(Sentence* sentence){
			int const* character_ids = sentence->_character_ids;
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
					int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
					int x_i_2 = (r - 2 <= character_ids_length) ? character_ids[r - 3] : CHARACTER_ID_EOS;
					if(x_i == x_i_2){
						// 発火
						int k_u = _crf->_index_w_identical_2_u(y_i, pos);
						_grad_w_identical_2[k_u] += 1;
						int k_b = _crf->_index_w_identical_2_b(y_i_1, y_i, pos);
						_grad_w_identical_2[k_b] += 1;

						// 発火の期待値
						int k_0 = _crf->_index_w_identical_2_u(0, pos);
						int k_1 = _crf->_index_w_identical_2_u(1, pos);
						_grad_w_identical_2[k_0] -= _lattice->_pz_s[i - 1][0][0];
						_grad_w_identical_2[k_0] -= _lattice->_pz_s[i - 1][1][0];
						_grad_w_identical_2[k_1] -= _lattice->_pz_s[i - 1][0][1];
						_grad_w_identical_2[k_1] -= _lattice->_pz_s[i - 1][1][1];

						int k_0_0 = _crf->_index_w_identical_2_b(0, 0, pos);
						int k_0_1 = _crf->_index_w_identical_2_b(0, 1, pos);
						int k_1_0 = _crf->_index_w_identical_2_b(1, 0, pos);
						int k_1_1 = _crf->_index_w_identical_2_b(1, 1, pos);
						_grad_w_identical_2[k_0_0] -= _lattice->_pz_s[i - 1][0][0];
						_grad_w_identical_2[k_0_1] -= _lattice->_pz_s[i - 1][0][1];
						_grad_w_identical_2[k_1_0] -= _lattice->_pz_s[i - 1][1][0];
						_grad_w_identical_2[k_1_1] -= _lattice->_pz_s[i - 1][1][1];
					}
				}
			}
		}
		void SGD::_backward_unigram_type(Sentence* sentence){
			int const* character_ids = sentence->_character_ids;
			wchar_t const* characters = sentence->_characters;
			int character_ids_length = sentence->size();
			for(int i = 1;i <= sentence->size() + 2;i++){
				// ラベルを取得
				int y_i_1 = sentence->get_crf_label_at(i - 1);
				int y_i = sentence->get_crf_label_at(i);
				int type_i = (i <= character_ids_length) ? ctype::get_type(characters[i - 1]) : CTYPE_UNKNOWN;

				// 発火
				int k_u = _crf->_index_w_unigram_type_u(y_i, type_i);
				_grad_w_unigram_type[k_u] += 1;
				int k_b = _crf->_index_w_unigram_type_b(y_i_1, y_i, type_i);
				_grad_w_unigram_type[k_b] += 1;

				// 発火の期待値
				int k_0 = _crf->_index_w_unigram_type_u(0, type_i);
				int k_1 = _crf->_index_w_unigram_type_u(1, type_i);
				_grad_w_unigram_type[k_0] -= _lattice->_pz_s[i - 1][0][0];
				_grad_w_unigram_type[k_0] -= _lattice->_pz_s[i - 1][1][0];
				_grad_w_unigram_type[k_1] -= _lattice->_pz_s[i - 1][0][1];
				_grad_w_unigram_type[k_1] -= _lattice->_pz_s[i - 1][1][1];

				int k_0_0 = _crf->_index_w_unigram_type_b(0, 0, type_i);
				int k_0_1 = _crf->_index_w_unigram_type_b(0, 1, type_i);
				int k_1_0 = _crf->_index_w_unigram_type_b(1, 0, type_i);
				int k_1_1 = _crf->_index_w_unigram_type_b(1, 1, type_i);
				_grad_w_unigram_type[k_0_0] -= _lattice->_pz_s[i - 1][0][0];
				_grad_w_unigram_type[k_0_1] -= _lattice->_pz_s[i - 1][0][1];
				_grad_w_unigram_type[k_1_0] -= _lattice->_pz_s[i - 1][1][0];
				_grad_w_unigram_type[k_1_1] -= _lattice->_pz_s[i - 1][1][1];
			}
		}
		void SGD::_backward_bigram_type(Sentence* sentence){
			int const* character_ids = sentence->_character_ids;
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
				_grad_w_bigram_type[k_u] += 1;
				int k_b = _crf->_index_w_bigram_type_b(y_i_1, y_i, type_i_1, type_i);
				_grad_w_bigram_type[k_b] += 1;

				// 発火の期待値
				int k_0 = _crf->_index_w_bigram_type_u(0, type_i_1, type_i);
				int k_1 = _crf->_index_w_bigram_type_u(1, type_i_1, type_i);
				_grad_w_bigram_type[k_0] -= _lattice->_pz_s[i - 1][0][0];
				_grad_w_bigram_type[k_0] -= _lattice->_pz_s[i - 1][1][0];
				_grad_w_bigram_type[k_1] -= _lattice->_pz_s[i - 1][0][1];
				_grad_w_bigram_type[k_1] -= _lattice->_pz_s[i - 1][1][1];

				int k_0_0 = _crf->_index_w_bigram_type_b(0, 0, type_i_1, type_i);
				int k_0_1 = _crf->_index_w_bigram_type_b(0, 1, type_i_1, type_i);
				int k_1_0 = _crf->_index_w_bigram_type_b(1, 0, type_i_1, type_i);
				int k_1_1 = _crf->_index_w_bigram_type_b(1, 1, type_i_1, type_i);
				_grad_w_bigram_type[k_0_0] -= _lattice->_pz_s[i - 1][0][0];
				_grad_w_bigram_type[k_0_1] -= _lattice->_pz_s[i - 1][0][1];
				_grad_w_bigram_type[k_1_0] -= _lattice->_pz_s[i - 1][1][0];
				_grad_w_bigram_type[k_1_1] -= _lattice->_pz_s[i - 1][1][1];
			}
		}
	}
}
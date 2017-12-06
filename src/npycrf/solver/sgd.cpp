#include <iostream>
#include "sgd.h"

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
		}
		void SGD::_backward_unigram(Sentence* sentence){
			int const* character_ids = sentence->_character_ids;
			wchar_t const* characters = sentence->_characters;
			int character_ids_length = sentence->size();
			for(int t = 1;t <= sentence->size() + 2;t++){
				// ラベルを取得
				int yt_1 = sentence->get_crf_label_at(t - 1);
				int yt = sentence->get_crf_label_at(t);
				assert(yt == sentence->get_crf_label_at(t));

				// ラベルunigram・bigram素性の範囲を網羅
				int r_start = std::max(1, t + _crf->_x_unigram_start);
				int r_end = std::min(character_ids_length + 2, t + _crf->_x_unigram_end);	// <eos>2つを考慮
				for(int r = r_start;r <= r_end;r++){
					int pos = r - t - _crf->_x_unigram_start + 1;	// [1, _x_range_unigram]
					int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;

					// 発火
					int k_u = _crf->_index_w_unigram_u(yt, pos, x_i);
					_grad_w_unigram[k_u] += 1;
					int k_b = _crf->_index_w_unigram_b(yt_1, yt, pos, x_i);
					_grad_w_unigram[k_b] += 1;

					// 発火の期待値
					int k_0 = _crf->_index_w_unigram_u(0, pos, x_i);
					int k_1 = _crf->_index_w_unigram_u(1, pos, x_i);
					_grad_w_unigram[k_0] -= _lattice->_pz_s[t - 1][0][0];
					_grad_w_unigram[k_0] -= _lattice->_pz_s[t - 1][1][0];
					_grad_w_unigram[k_1] -= _lattice->_pz_s[t - 1][0][1];
					_grad_w_unigram[k_1] -= _lattice->_pz_s[t - 1][1][1];

					int k_0_0 = _crf->_index_w_unigram_b(0, 0, pos, x_i);
					int k_0_1 = _crf->_index_w_unigram_b(0, 1, pos, x_i);
					int k_1_0 = _crf->_index_w_unigram_b(1, 0, pos, x_i);
					int k_1_1 = _crf->_index_w_unigram_b(1, 1, pos, x_i);
					_grad_w_unigram[k_0_0] -= _lattice->_pz_s[t - 1][0][0];
					_grad_w_unigram[k_0_1] -= _lattice->_pz_s[t - 1][0][1];
					_grad_w_unigram[k_1_0] -= _lattice->_pz_s[t - 1][1][0];
					_grad_w_unigram[k_1_1] -= _lattice->_pz_s[t - 1][1][1];
				}
			}
		}
	}
}
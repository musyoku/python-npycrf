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
			for(int i = 0;i < crf->_w_size_label_u + crf->_w_size_label_b;i++){
				_grad_w_label[i] = 0;
			}
			for(int i = 0;i < crf->_w_size_unigram_u + crf->_w_size_unigram_b;i++){
				_w_grad_unigram[i] = 0;
			}
			for(int i = 0;i < crf->_w_size_bigram_u + crf->_w_size_bigram_b;i++){
				_wgrad__bigram[i] = 0;
			}
			for(int i = 0;i < crf->_w_size_identical_1_u + crf->_w_size_identical_1_b;i++){
				_w_idengrad_tical_1[i] = 0;
			}
			for(int i = 0;i < crf->_w_size_identical_2_u + crf->_w_size_identical_2_b;i++){
				_w_idengrad_tical_2[i] = 0;
			}
			for(int i = 0;i < crf->_w_size_unigram_type_u + crf->_w_size_unigram_type_b;i++){
				_w_unigrgrad_am_type[i] = 0;
			}
			for(int i = 0;i < crf->_w_size_bigram_type_u + crf->_w_size_bigram_type_b;i++){
				_w_bigrgrad_am_type[i] = 0;
			}
		}
		void SGD::backward(Sentence* sentence){

		}
	}
}
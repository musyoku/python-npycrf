#pragma once
#include "../common.h"
#include "../sentence.h"
#include "../crf/crf.h"
#include "../lattice.h"

namespace npycrf {
	namespace solver {
		class SGD {
		public:
			crf::CRF* _crf;
			Lattice* _lattice;
			double _grad_bias;
			double* _grad_w_label;			// (y_i), (y_{i-1}, y_i)
			double* _grad_w_unigram;		// (y_i, i, x_i), (y_{i-1}, y_i, i, x_i)
			double* _grad_w_bigram;			// (y_i, i, x_{i-1}, x_i), (y_{i-1}, y_i, i, x_{i-1}, x_i)
			double* _grad_w_identical_1;	// (y_i, i), (y_{i-1}, y_i, i)
			double* _grad_w_identical_2;	// (y_i, i), (y_{i-1}, y_i, i)
			double* _grad_w_unigram_type;	// (y_i, type), (y_{i-1}, y_i, type)
			double* _grad_w_bigram_type;	// (y_i, type, type), (y_{i-1}, y_i, type, type)
			SGD(Lattice* lattice, crf::CRF* crf);
			~SGD();
			void clear_grads();
			void backward(Sentence* sentence);
			void _backward_unigram(Sentence* sentence);
			void _backward_bigram(Sentence* sentence);
			void _backward_identical_1(Sentence* sentence);
			void _backward_identical_2(Sentence* sentence);
			void _backward_unigram_type(Sentence* sentence);
			void _backward_bigram_type(Sentence* sentence);
		};
	}
}
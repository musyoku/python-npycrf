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
			double _grad_bias;
			double* _grad_w_label;			// (y_i), (y_{i-1}, y_i)
			double* _grad_w_unigram;		// (y_i, i, x_i), (y_{i-1}, y_i, i, x_i)
			double* _grad_w_bigram;			// (y_i, i, x_{i-1}, x_i), (y_{i-1}, y_i, i, x_{i-1}, x_i)
			double* _grad_w_identical_1;	// (y_i, i), (y_{i-1}, y_i, i)
			double* _grad_w_identical_2;	// (y_i, i), (y_{i-1}, y_i, i)
			double* _grad_w_unigram_type;	// (y_i, type), (y_{i-1}, y_i, type)
			double* _grad_w_bigram_type;	// (y_i, type, type), (y_{i-1}, y_i, type, type)
			SGD(crf::CRF* crf);
			~SGD();
			void clear_grads();
			void update(double learning_rate);
			void backward(Sentence* sentence, double*** pz_s);
			void _backward_unigram(Sentence* sentence, double*** pz_s);
			void _backward_bigram(Sentence* sentence, double*** pz_s);
			void _backward_identical_1(Sentence* sentence, double*** pz_s);
			void _backward_identical_2(Sentence* sentence, double*** pz_s);
			void _backward_unigram_type(Sentence* sentence, double*** pz_s);
			void _backward_bigram_type(Sentence* sentence, double*** pz_s);
			void _backward_label(Sentence* sentence, double*** pz_s);
		};
	}
}
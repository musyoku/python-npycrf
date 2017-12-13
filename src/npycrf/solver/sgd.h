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
			double _regularization_constant;
			double _grad_bias;
			double* _grad_weight;
			double _grad_lambda_0;
			SGD(crf::CRF* crf, double regularization_constant);
			~SGD();
			void clear_grads();
			void update(double learning_rate);
			void backward_crf(Sentence* sentence, double*** pz_s);
			void backward_lambda_0(Sentence* sentence, double**** p_conc_tkji, double**** pw_h_tkji, int max_word_length);
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
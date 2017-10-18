#pragma once
#include "../common.h"

// [1] A discriminative latent variable chinese segmenter with hybrid word/character information
//     https://pdfs.semanticscholar.org/0bd3/a662e19467aa21c1e1e0a51db397e9936b70.pdf

namespace npycrf {
	namespace crf {
		class CRF {
		public:
			double* _w_unigram;
			double** _w_bigram;
			double* _w_identical_1;
			double* _w_identical_2;
			// 用いる素性は以下の4通り（デフォルト値の例）[1]
			// i-2, i-1, i, i+1, i+2の位置のunigram文字
			// i-2, i-1, i, i+1の位置のbigram文字
			// i-2, i-1, i, i+1において、x_i == x_{i+1}
			// i-3, i-2, i-1, i, i+1において、x_i == x_{i+2}
			CRF(int feature_unigram_start = -2,
				int feature_unigram_end = 2,
				int feature_bigram_start = -2,
				int feature_bigram_end = 1,
				int feature_identical_1_start = -2,
				int feature_identical_1_end = 1,
				int feature_identical_2_start = -3,
				int feature_identical_2_end = 1,
				int num_character_ids,		// 文字IDの総数
			);
		};
	}
}
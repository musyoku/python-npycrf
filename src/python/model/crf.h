#pragma once
#include <boost/python.hpp>
#include "../../npycrf/crf/crf.h"
#include "../dictionary.h"
#include "../dataset.h"

namespace npycrf {
	namespace python {
		namespace model {
			class CRF{
			public:
				crf::CRF* _crf;
				CRF(Dataset* dataset_labeled,			// CRF素性の展開に用いる
					int num_character_ids,		// 文字IDの総数
					int feature_x_unigram_start,
					int feature_x_unigram_end,
					int feature_x_bigram_start,
					int feature_x_bigram_end,
					int feature_x_identical_1_start,
					int feature_x_identical_1_end,
					int feature_x_identical_2_start,
					int feature_x_identical_2_end,
					double initial_lambda_0,
					double sigma);
				CRF(std::string filename);
				~CRF();
				int get_num_features();
				double get_lambda_0();
				bool load(std::string filename);
				bool save(std::string filename);
				void print_weight_distribution();
			};
		}
	}
}
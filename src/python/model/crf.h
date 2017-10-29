#pragma once
#include <boost/python.hpp>
#include "../../npycrf/crf/crf.h"

namespace npycrf {
	namespace python {
		namespace model {
			class CRF{
			public:
				crf::CRF* _crf;
				CRF(int num_character_ids,		// 文字IDの総数
					int num_character_types,	// 文字種の総数
					int feature_x_unigram_start = -2,
					int feature_x_unigram_end = 2,
					int feature_x_bigram_start = -2,
					int feature_x_bigram_end = 1,
					int feature_x_identical_1_start = -2,
					int feature_x_identical_1_end = 1,
					int feature_x_identical_2_start = -3,
					int feature_x_identical_2_end = 1);
				CRF(std::string filename);
				~CRF();
				bool load(std::string filename);
				bool save(std::string filename);
			};
		}
	}
}
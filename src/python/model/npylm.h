#pragma once
#include <boost/python.hpp>
#include "../../npycrf/npylm/npylm.h"

namespace npycrf {
	namespace python {
		namespace model {
			class NPYLM{
			public:
				npylm::NPYLM* _npylm;
				NPYLM(int max_word_length, 
					  int max_sentence_length, 
					  double g0, 
					  double initial_lambda_a, 
					  double initial_lambda_b, 
					  double vpylm_beta_stop, 
					  double vpylm_beta_pass);
				NPYLM(std::string filename);
				~NPYLM();
				bool load(std::string filename);
				bool save(std::string filename);
			};
		}
	}
}
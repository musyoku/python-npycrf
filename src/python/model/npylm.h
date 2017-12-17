#pragma once
#include <boost/python.hpp>
#include "../../npycrf/npylm/npylm.h"
#include "../../npycrf/lattice.h"
#include "../dictionary.h"

namespace npycrf {
	namespace python {
		namespace model {
			class NPYLM{
			public:
				npylm::NPYLM* _npylm;
				Lattice* _lattice;
				NPYLM(int max_word_length, 
					  double g0, 
					  double initial_lambda_a, 
					  double initial_lambda_b, 
					  double vpylm_beta_stop, 
					  double vpylm_beta_pass);
				NPYLM(std::string filename);
				~NPYLM();
				void parse(Sentence* sentence);
				boost::python::list python_parse(std::wstring sentence_str, Dictionary* dictionary);
				bool load(std::string filename);
				bool save(std::string filename);
			};
		}
	}
}
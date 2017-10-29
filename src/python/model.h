#pragma once
#include <boost/python.hpp>
#include "../npycrf/npylm/npylm.h"
#include "../npycrf/lattice.h"
#include "dataset.h"
#include "dictionary.h"
#include "model/npylm.h"
#include "model/crf.h"

namespace npycrf {
	namespace python {
		class Model{
		private:
			void _set_locale();
		public:
			npylm::NPYLM* _npylm;
			crf::CRF* _crf;
			Lattice* _lattice;			// forward filtering-backward sampling
			Model(model::NPYLM* py_npylm, model::CRF* py_crf, int max_word_length, int max_sentence_length);
			~Model();
			int get_max_word_length();
			void set_initial_lambda_a(double lambda);
			void set_initial_lambda_b(double lambda);
			void set_vpylm_beta_stop(double stop);
			void set_vpylm_beta_pass(double pass);
			boost::python::list parse(std::wstring sentence_str, Dictionary* dictionary);
		};
	}
}
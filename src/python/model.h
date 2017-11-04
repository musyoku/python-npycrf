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
			double _lambda_0;
			npylm::NPYLM* _npylm;
			crf::CRF* _crf;
			Lattice* _lattice;			// forward filtering-backward sampling
			Model(model::NPYLM* py_npylm, model::CRF* py_crf, double lambda_0, int max_word_length, int max_sentence_length);
			~Model();
			int get_max_word_length();
			void set_initial_lambda_a(double lambda);
			void set_initial_lambda_b(double lambda);
			void set_vpylm_beta_stop(double stop);
			void set_vpylm_beta_pass(double pass);
			double compute_log_p_proportional_y_given_x(Sentence* sentence);
			double compute_z_x(Sentence* sentence, bool normalize = true);
			double compute_forward_probability(std::wstring sentence_str, Dictionary* dictionary, bool normalize = true);
			double compute_backward_probability(std::wstring sentence_str, Dictionary* dictionary, bool normalize = true);
			void parse(Sentence* sentence);
			boost::python::list python_parse(std::wstring sentence_str, Dictionary* dictionary);
		};
	}
}
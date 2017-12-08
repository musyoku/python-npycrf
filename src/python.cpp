#include "python/npycrf.h"
#include "python/dataset.h"
#include "python/dictionary.h"
#include "python/trainer.h"

using namespace npycrf;
using namespace npycrf::python;
using boost::python::arg;
using boost::python::args;

BOOST_PYTHON_MODULE(npycrf){
	boost::python::class_<Dictionary>("dictionary")
	.def("get_num_characters", &Dictionary::get_num_characters)
	.def("save", &Dictionary::save)
	.def("load", &Dictionary::load);

	boost::python::class_<Corpus>("corpus")
	.def("add_words", &Corpus::python_add_words);

	boost::python::class_<Dataset>("dataset", boost::python::init<Corpus*, Dictionary*, double, int>((args("corpus", "dictionary", "train_dev_split", "seed"))))
	.def("get_max_sentence_length", &Dataset::get_max_sentence_length)
	.def("get_size_train", &Dataset::get_size_train)
	.def("get_size_dev", &Dataset::get_size_dev);

	boost::python::class_<Trainer>("trainer", boost::python::init<Dataset*, Dataset*, Dictionary*, NPYCRF*, double>((args("dataset_labeled", "dataset_unlabeled", "dictionary", "npycrf", "crf_regularization_constant"))))
	.def("detect_hash_collision", &Trainer::detect_hash_collision)
	.def("print_segmentation_train", &Trainer::print_segmentation_train)
	.def("print_segmentation_dev", &Trainer::print_segmentation_dev)
	.def("sample_hpylm_vpylm_hyperparameters", &Trainer::sample_hpylm_vpylm_hyperparameters)
	.def("sample_npylm_lambda", &Trainer::sample_npylm_lambda)
	.def("update_p_k_given_vpylm", &Trainer::update_p_k_given_vpylm)
	.def("compute_perplexity_train", &Trainer::compute_perplexity_train)
	.def("compute_perplexity_dev", &Trainer::compute_perplexity_dev)
	.def("compute_log_likelihood_labelded_train", &Trainer::compute_log_likelihood_labelded_train)
	.def("compute_log_likelihood_unlabelded_train", &Trainer::compute_log_likelihood_unlabelded_train)
	.def("compute_log_likelihood_labeled_dev", &Trainer::compute_log_likelihood_labeled_dev)
	.def("compute_log_likelihood_unlabeled_dev", &Trainer::compute_log_likelihood_unlabeled_dev)
	.def("add_labelded_data_to_npylm", &Trainer::add_labelded_data_to_npylm)
	.def("sgd", &Trainer::sgd, (arg("learning_rate"), arg("batchsize")=32, arg("pure_crf")=false))
	.def("gibbs", &Trainer::gibbs, (arg("include_labeled_data")=false));

	boost::python::class_<NPYCRF>("npycrf", boost::python::init<model::NPYLM*, model::CRF*>((args("npylm", "crf"))))
	.def("parse", &NPYCRF::python_parse);

	boost::python::class_<model::CRF>("crf", boost::python::init<int, int, int, int, int, int, int, int, int, double>((args("num_character_ids", "feature_x_unigram_start", "feature_x_unigram_end", "feature_x_bigram_start", "feature_x_bigram_end", "feature_x_identical_1_start", "feature_x_identical_1_end", "feature_x_identical_2_start", "feature_x_identical_2_end", "sigma"))))
	.def("save", &model::CRF::save)
	.def("load", &model::CRF::load);

	boost::python::class_<model::NPYLM>("npylm", boost::python::init<int, double, double, double, double, double>((args("max_word_length", "g0", "initial_lambda_a", "initial_lambda_b", "vpylm_beta_stop", "vpylm_beta_pass"))))
	.def("save", &model::NPYLM::save)
	.def("load", &model::NPYLM::load);
}
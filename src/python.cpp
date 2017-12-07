#include "python/model/crf.h"
#include "python/model/npylm.h"
#include "python/model.h"
#include "python/dataset.h"
#include "python/dictionary.h"
#include "python/trainer.h"

using namespace npycrf;
using namespace npycrf::python;
using boost::python::arg;
using boost::python::args;

BOOST_PYTHON_MODULE(npycrf){
	boost::python::class_<Dictionary>("dictionary")
	.def("save", &Dictionary::save)
	.def("load", &Dictionary::load);

	boost::python::class_<Corpus>("corpus")
	.def("add_words", &Corpus::python_add_words);

	boost::python::class_<Dataset>("dataset", boost::python::init<Corpus*, Dictionary*, double, int>((args("corpus", "dictionary", "train_dev_split", "seed"))))
	.def("get_max_sentence_length", &Dataset::get_max_sentence_length)
	.def("get_size_train", &Dataset::get_size_train)
	.def("get_size_dev", &Dataset::get_size_dev);

	boost::python::class_<Trainer>("trainer", boost::python::init<Dataset*, Dataset*, Dictionary*, Model*, double>((args("dataset_labeled", "dataset_unlabeled", "dictionary", "model", "crf_regularization_constant"))))
	.def("detect_hash_collision", &Trainer::detect_hash_collision)
	.def("print_segmentation_train", &Trainer::print_segmentation_train)
	.def("print_segmentation_dev", &Trainer::print_segmentation_dev)
	.def("sample_hpylm_vpylm_hyperparameters", &Trainer::sample_hpylm_vpylm_hyperparameters)
	.def("sample_npylm_lambda", &Trainer::sample_npylm_lambda)
	.def("update_p_k_given_vpylm", &Trainer::update_p_k_given_vpylm)
	.def("compute_perplexity_train", &Trainer::compute_perplexity_train)
	.def("compute_perplexity_dev", &Trainer::compute_perplexity_dev)
	.def("compute_log_likelihood_train", &Trainer::compute_log_likelihood_train)
	.def("compute_log_likelihood_dev", &Trainer::compute_log_likelihood_dev)
	.def("sgd", &Trainer::sgd, (arg("learning_rate"), arg("batchsize")=32, arg("pure_crf")=false))
	.def("gibbs", &Trainer::gibbs, (arg("include_labeled_data")=false));

	boost::python::class_<Model>("model", boost::python::init<model::NPYLM*, model::CRF*, double>((args("npylm", "crf", "lambda_0"))))
	.def(boost::python::init<std::string>((arg("filename"))))
	.def("parse", &Model::python_parse);
}
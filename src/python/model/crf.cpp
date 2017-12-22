#include <fstream>
#include <iostream>
#include "crf.h"
#include "../../npycrf/common.h"
#include "../../npycrf/ctype.h"

using namespace npycrf::crf::feature;

namespace npycrf {
	namespace python {
		namespace model {
			CRF::CRF(Dataset* dataset_labeled,
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
					 double sigma)
			{
				FeatureExtractor* extractor = new FeatureExtractor(num_character_ids, 
																	CTYPE_NUM_TYPES,
																	feature_x_unigram_start,
																	feature_x_unigram_end,
																	feature_x_bigram_start,
																	feature_x_bigram_end,
																	feature_x_identical_1_start,
																	feature_x_identical_1_end,
																	feature_x_identical_2_start,
																	feature_x_identical_2_end);

				for(Sentence* sentence: dataset_labeled->_sentences_train){
					assert(sentence->_features == NULL);
					sentence->_features = extractor->extract(sentence, true);
				}
				for(Sentence* sentence: dataset_labeled->_sentences_dev){
					assert(sentence->_features == NULL);
					sentence->_features = extractor->extract(sentence, true);
				}
				int weight_size = extractor->_function_id_to_feature_id.size();
				std::cout << "weight_size = " << weight_size << std::endl;
				crf::Parameter* parameter = new crf::Parameter(weight_size, initial_lambda_0, sigma);
				_crf = new crf::CRF(extractor, parameter);
			}
			CRF::CRF(std::string filename){
				_crf = new crf::CRF();
				if(load(filename) == false){
					std::cout << filename << " not found." << std::endl;
					exit(0);
				}
			}
			CRF::~CRF(){
				delete _crf;
			}
			int CRF::get_num_features(){
				return _crf->_parameter->get_num_features();
			}
			double CRF::get_lambda_0(){
				return _crf->_parameter->_lambda_0;
			}
			bool CRF::load(std::string filename){
				bool success = false;
				std::ifstream ifs(filename);
				if(ifs.good()){
					boost::archive::binary_iarchive iarchive(ifs);
					iarchive >> *_crf;
					success = true;
				}
				ifs.close();
				return success;
			}
			bool CRF::save(std::string filename){
				bool success = false;
				std::ofstream ofs(filename);
				if(ofs.good()){
					boost::archive::binary_oarchive oarchive(ofs);
					oarchive << *_crf;
					success = true;
				}
				ofs.close();
				return success;
			}
		}
	}
}
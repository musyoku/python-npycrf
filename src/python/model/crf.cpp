#include <fstream>
#include <iostream>
#include "../../npycrf/common.h"
#include "crf.h"

namespace npycrf {
	namespace python {
		namespace model {
			CRF::CRF(int num_character_ids,		// 文字IDの総数
					 int num_character_types,	// 文字種の総数
					 int feature_x_unigram_start,
					 int feature_x_unigram_end,
					 int feature_x_bigram_start,
					 int feature_x_bigram_end,
					 int feature_x_identical_1_start,
					 int feature_x_identical_1_end,
					 int feature_x_identical_2_start,
					 int feature_x_identical_2_end,
					 double sigma)
			{
				_crf = new crf::CRF(num_character_ids, 
									num_character_types,
									feature_x_unigram_start,
									feature_x_unigram_end,
									feature_x_bigram_start,
									feature_x_bigram_end,
									feature_x_identical_1_start,
									feature_x_identical_1_end,
									feature_x_identical_2_start,
									feature_x_identical_2_end,
									sigma);
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
#include <fstream>
#include <iostream>
#include "../../npycrf/common.h"
#include "npylm.h"

namespace npycrf {
	namespace python {
		namespace model {
			NPYLM::NPYLM(int max_word_length, 		// 可能な単語長の最大値. 英語16, 日本語8程度
						 int max_sentence_length, 
						 double g0, 				// VPYLMのg0
						 double initial_lambda_a,   // 単語長のポアソン分布のλの事前分布のハイパーパラメータ
						 double initial_lambda_b,   // 単語長のポアソン分布のλの事前分布のハイパーパラメータ
						 double vpylm_beta_stop, 	// VPYLMのハイパーパラメータ
						 double vpylm_beta_pass)	// VPYLMのハイパーパラメータ
			{
				_npylm = new npylm::NPYLM(max_word_length, max_sentence_length, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);
			}
			NPYLM::NPYLM(std::string filename){
				_npylm = new npylm::NPYLM();
				if(load(filename) == false){
					std::cout << filename << " not found." << std::endl;
					exit(0);
				}
			}
			NPYLM::~NPYLM(){
				delete _npylm;
			}
			bool NPYLM::load(std::string filename){
				bool success = false;
				std::ifstream ifs(filename);
				if(ifs.good()){
					boost::archive::binary_iarchive iarchive(ifs);
					iarchive >> *_npylm;
					success = true;
				}
				ifs.close();
				return success;
			}
			bool NPYLM::save(std::string filename){
				bool success = false;
				std::ofstream ofs(filename);
				if(ofs.good()){
					boost::archive::binary_oarchive oarchive(ofs);
					oarchive << *_npylm;
					success = true;
				}
				ofs.close();
				return success;
			}
		}
	}
}
#include <fstream>
#include <iostream>
#include "../../npycrf/common.h"
#include "npylm.h"

namespace npycrf {
	namespace python {
		namespace model {
			NPYLM::NPYLM(int max_word_length, 		// 可能な単語長の最大値. 英語16, 日本語8程度
						 double g0, 				// VPYLMのg0
						 double initial_lambda_a,   // 単語長のポアソン分布のλの事前分布のハイパーパラメータ
						 double initial_lambda_b,   // 単語長のポアソン分布のλの事前分布のハイパーパラメータ
						 double vpylm_beta_stop, 	// VPYLMのハイパーパラメータ
						 double vpylm_beta_pass)	// VPYLMのハイパーパラメータ
			{
				_npylm = new npylm::NPYLM(max_word_length, 100, g0, initial_lambda_a, initial_lambda_b, vpylm_beta_stop, vpylm_beta_pass);
				_lattice = new Lattice(_npylm, NULL);
				_lattice->set_pure_npylm_mode(true);
			}
			NPYLM::NPYLM(std::string filename){
				_npylm = new npylm::NPYLM();
				_lattice = new Lattice(_npylm, NULL);
				_lattice->set_pure_npylm_mode(true);
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
			void NPYLM::parse(Sentence* sentence){
				// キャッシュの再確保
				_lattice->reserve(_npylm->_max_word_length, sentence->size());
				_npylm->reserve(sentence->size());
				std::vector<int> segments;		// 分割の一時保存用
				_lattice->viterbi_decode(sentence, segments);
				sentence->split(segments);
			}
			boost::python::list NPYLM::python_parse(std::wstring sentence_str, Dictionary* dictionary){
				// キャッシュの再確保
				_lattice->reserve(_npylm->_max_word_length, sentence_str.size());
				_npylm->reserve(sentence_str.size());
				std::vector<int> segments;		// 分割の一時保存用
				// 構成文字を文字IDに変換
				array<int> character_ids = array<int>(sentence_str.size());
				for(int i = 0;i < sentence_str.size();i++){
					wchar_t character = sentence_str[i];
					int character_id = dictionary->get_character_id(character);
					character_ids[i] = character_id;
				}
				Sentence* sentence = new Sentence(sentence_str, character_ids);
				_lattice->viterbi_decode(sentence, segments);
				sentence->split(segments);
				boost::python::list words;
				for(int n = 0;n < sentence->get_num_segments_without_special_tokens();n++){
					std::wstring word = sentence->get_word_str_at(n + 2);
					words.append(word);
				}
				delete sentence;
				return words;
			}
		}
	}
}
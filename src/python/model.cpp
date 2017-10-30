#include <iostream>
#include "../npycrf/common.h"
#include "model.h"

using namespace npycrf::npylm;
using namespace npycrf::crf;

namespace npycrf {
	namespace python {
		Model::Model(model::NPYLM* py_npylm, model::CRF* py_crf, double lambda_0, int max_word_length, int max_sentence_length){
			_set_locale();
			_npylm = py_npylm->_npylm;
			_crf = py_crf->_crf;
			_lattice = new Lattice(_npylm, _crf, lambda_0, max_word_length, max_sentence_length);
		}
		Model::~Model(){

		}
		// 日本語周り
		void Model::_set_locale(){
			setlocale(LC_CTYPE, "ja_JP.UTF-8");
			std::ios_base::sync_with_stdio(false);
			std::locale default_loc("ja_JP.UTF-8");
			std::locale::global(default_loc);
			std::locale ctype_default(std::locale::classic(), default_loc, std::locale::ctype); //※
			std::wcout.imbue(ctype_default);
			std::wcin.imbue(ctype_default);
		}
		int Model::get_max_word_length(){
			return _npylm->_max_word_length;
		}
		void Model::set_initial_lambda_a(double lambda){
			_npylm->_lambda_a = lambda;
			_npylm->sample_lambda_with_initial_params();
		}
		void Model::set_initial_lambda_b(double lambda){
			_npylm->_lambda_b = lambda;
			_npylm->sample_lambda_with_initial_params();
		}
		void Model::set_vpylm_beta_stop(double stop){
			_npylm->_vpylm->_beta_stop = stop;
		}
		void Model::set_vpylm_beta_pass(double pass){
			_npylm->_vpylm->_beta_pass = pass;
		}
		// normalize=trueならアンダーフローを防ぐ
		double Model::compute_forward_probability(std::wstring sentence_str, Dictionary* dictionary, bool normalize){
			// キャッシュの再確保
			if(sentence_str.size() > _lattice->_max_sentence_length){
				_lattice->delete_arrays();
				_lattice->allocate_arrays(_npylm->_max_word_length, sentence_str.size());
			}
			if(sentence_str.size() > _npylm->_max_sentence_length){
				_npylm->delete_arrays();
				_npylm->allocate_arrays(sentence_str.size());
			}
			std::vector<int> segments;		// 分割の一時保存用
			// 構成文字を辞書に追加し、文字IDに変換
			int* character_ids = new int[sentence_str.size()];
			for(int i = 0;i < sentence_str.size();i++){
				wchar_t character = sentence_str[i];
				int character_id = dictionary->get_character_id(character);
				character_ids[i] = character_id;
			}
			Sentence* sentence = new Sentence(sentence_str, character_ids);
			double probability = _lattice->compute_forward_probability(sentence, normalize);
			delete[] character_ids;
			delete sentence;
			return probability;
		}
		boost::python::list Model::python_parse(std::wstring sentence_str, Dictionary* dictionary){
			// キャッシュの再確保
			if(sentence_str.size() > _lattice->_max_sentence_length){
				_lattice->delete_arrays();
				_lattice->allocate_arrays(_npylm->_max_word_length, sentence_str.size());
			}
			if(sentence_str.size() > _npylm->_max_sentence_length){
				_npylm->delete_arrays();
				_npylm->allocate_arrays(sentence_str.size());
			}
			std::vector<int> segments;		// 分割の一時保存用
			// 構成文字を辞書に追加し、文字IDに変換
			int* character_ids = new int[sentence_str.size()];
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
			delete[] character_ids;
			delete sentence;
			return words;
		}
	}
}
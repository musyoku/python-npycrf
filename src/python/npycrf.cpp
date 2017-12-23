#include <iostream>
#include "../npycrf/common.h"
#include "../npycrf/array.h"
#include "npycrf.h"

using namespace npycrf::npylm;
using namespace npycrf::crf;
using namespace npycrf::crf::feature;

namespace npycrf {
	namespace python {
		NPYCRF::NPYCRF(model::NPYLM* py_npylm, model::CRF* py_crf){
			_set_locale();
			_npylm = py_npylm->_npylm;
			_crf = py_crf->_crf;
			_lattice = new Lattice(_npylm, _crf);
		}
		NPYCRF::~NPYCRF(){
			delete _lattice;
		}
		// 日本語周り
		void NPYCRF::_set_locale(){
			setlocale(LC_CTYPE, "ja_JP.UTF-8");
			std::ios_base::sync_with_stdio(false);
			std::locale default_loc("ja_JP.UTF-8");
			std::locale::global(default_loc);
			std::locale ctype_default(std::locale::classic(), default_loc, std::locale::ctype); //※
			std::wcout.imbue(ctype_default);
			std::wcin.imbue(ctype_default);
		}
		int NPYCRF::get_max_word_length(){
			return _npylm->_max_word_length;
		}
		void NPYCRF::set_initial_lambda_a(double lambda){
			_npylm->_lambda_a = lambda;
			_npylm->sample_lambda_with_initial_params();
		}
		void NPYCRF::set_initial_lambda_b(double lambda){
			_npylm->_lambda_b = lambda;
			_npylm->sample_lambda_with_initial_params();
		}
		void NPYCRF::set_vpylm_beta_stop(double stop){
			_npylm->_vpylm->_beta_stop = stop;
		}
		void NPYCRF::set_vpylm_beta_pass(double pass){
			_npylm->_vpylm->_beta_pass = pass;
		}
		double NPYCRF::get_lambda_0(){
			return _crf->_parameter->_lambda_0;
		}
		void NPYCRF::set_lambda_0(double lambda_0){
			_crf->_parameter->_lambda_0 = lambda_0;
		}
		// 分配関数の計算
		double NPYCRF::compute_normalizing_constant(Sentence* sentence){
			// キャッシュの再確保
			_lattice->reserve(_npylm->_max_word_length, sentence->size());
			_npylm->reserve(sentence->size());
			_npylm->clear_g0_cache();
			double Zs = _lattice->compute_normalizing_constant(sentence, true);
			#ifdef __DEBUG__
				double __Zs = _lattice->_compute_normalizing_constant_backward(sentence, _lattice->_beta, _lattice->_p_transition_tkji);
				double ___Zs = _lattice->compute_normalizing_constant(sentence, false);
				if(std::abs(1 - Zs / __Zs) >= 1e-14){
					std::cout << std::abs(1 - Zs / __Zs) << std::endl;
				}
				if(std::abs(1 - Zs / ___Zs) >= 1e-14){
					std::cout << std::abs(1 - Zs / ___Zs) << std::endl;
				}
				assert(std::abs(1 - Zs / __Zs) < 1e-14);
				assert(std::abs(1 - Zs / ___Zs) < 1e-14);
			#endif 
			return Zs;
		}
		// 分配関数の計算
		double NPYCRF::compute_log_normalizing_constant(Sentence* sentence){
			// キャッシュの再確保
			_lattice->reserve(_npylm->_max_word_length, sentence->size());
			_npylm->reserve(sentence->size());
			_npylm->clear_g0_cache();
			double log_Zs = _lattice->compute_log_normalizing_constant(sentence, true);
			#ifdef __DEBUG__
				double _Zs = _lattice->compute_normalizing_constant(sentence, false);
				assert(std::abs(1 - log_Zs / log(_Zs)) < 1e-14);
			#endif 
			return log_Zs;
		}
		double NPYCRF::python_compute_normalizing_constant(std::wstring sentence_str, Dictionary* dictionary){
			// キャッシュの再確保
			_lattice->reserve(_npylm->_max_word_length, sentence_str.size());
			_npylm->reserve(sentence_str.size());
			_npylm->clear_g0_cache();
			std::vector<int> segments;		// 分割の一時保存用
			// 構成文字を文字IDに変換
			array<int> character_ids = array<int>(sentence_str.size());
			for(int i = 0;i < sentence_str.size();i++){
				wchar_t character = sentence_str[i];
				int character_id = dictionary->get_character_id(character);
				character_ids[i] = character_id;
			}
			Sentence* sentence = new Sentence(sentence_str, character_ids);
			double ret = compute_normalizing_constant(sentence);
			delete sentence;
			return ret;
		}
		// sentenceは分割済みの必要がある
		// 比例のままの確率を返す
		double NPYCRF::compute_log_proportional_p_y_given_sentence(Sentence* sentence){
			_npylm->reserve(sentence->size());	// キャッシュの再確保
			_npylm->clear_g0_cache();
			double log_crf = _crf->compute_log_p_y_given_sentence(sentence);
			double log_npylm = _npylm->compute_log_p_y_given_sentence(sentence);
			double log_py_x = log_crf + get_lambda_0() * log_npylm;
			return log_py_x;
		}
		void NPYCRF::parse(Sentence* sentence){
			// キャッシュの再確保
			_lattice->reserve(_npylm->_max_word_length, sentence->size());
			_npylm->reserve(sentence->size());
			_npylm->clear_g0_cache();
			sentence->_features = _crf->extract_features(sentence, false);		// CRFの素性IDを展開
			std::vector<int> segments;		// 分割の一時保存用
			_lattice->viterbi_decode(sentence, segments);
			sentence->split(segments);
		}
		boost::python::list NPYCRF::python_parse(std::wstring sentence_str, Dictionary* dictionary){
			// キャッシュの再確保
			_lattice->reserve(_npylm->_max_word_length, sentence_str.size());
			_npylm->reserve(sentence_str.size());
			_npylm->clear_g0_cache();
			std::vector<int> segments;		// 分割の一時保存用
			// 構成文字を文字IDに変換
			array<int> character_ids = array<int>(sentence_str.size());
			for(int i = 0;i < sentence_str.size();i++){
				wchar_t character = sentence_str[i];
				int character_id = dictionary->get_character_id(character);
				character_ids[i] = character_id;
			}
			Sentence* sentence = new Sentence(sentence_str, character_ids);
			sentence->_features = _crf->extract_features(sentence, false);		// CRFの素性IDを展開
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
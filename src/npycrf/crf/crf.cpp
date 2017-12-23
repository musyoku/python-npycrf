#include <boost/serialization/split_member.hpp>
#include <iostream>
#include <cassert>
#include "../ctype.h"
#include "../sampler.h"
#include "crf.h"

namespace npycrf {
	namespace crf {
		CRF::CRF(){
			_parameter = new Parameter();
			_extractor = new FeatureExtractor();
		}
		CRF::CRF(FeatureExtractor* extractor, Parameter* parameter)
		{
			_extractor = extractor;
			_parameter = parameter;
		}
		CRF::~CRF(){
			delete _extractor;
			delete _parameter;
		}
		int CRF::get_num_features(){
			return _extractor->_function_id_to_feature_id.size();
		}
		double CRF::w_label_u(int y_i) const {
			int index = _extractor->feature_id_label_u(y_i);
			return _parameter->_weights[index];
		}
		double CRF::w_label_b(int y_i_1, int y_i) const {
			int index = _extractor->feature_id_label_b(y_i_1, y_i);
			return _parameter->_weights[index];
		}
		double CRF::w_unigram_u(int y_i, int i, int x_i) const {
			int index = _extractor->feature_id_unigram_u(y_i, i, x_i);
			return _parameter->_weights[index];
		}
		double CRF::w_unigram_b(int y_i_1, int y_i, int i, int x_i) const {
			int index = _extractor->feature_id_unigram_b(y_i_1, y_i, i, x_i);
			return _parameter->_weights[index];
		}
		double CRF::w_bigram_u(int y_i, int i, int x_i_1, int x_i) const {
			int index = _extractor->feature_id_bigram_u(y_i, i, x_i_1, x_i);
			return _parameter->_weights[index];
		}
		double CRF::w_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i) const {
			int index = _extractor->feature_id_bigram_b(y_i_1, y_i, i, x_i_1, x_i);
			return _parameter->_weights[index];
		}
		double CRF::w_identical_1_u(int y_i, int i) const {
			int index = _extractor->feature_id_identical_1_u(y_i, i);
			return _parameter->_weights[index];
		}
		double CRF::w_identical_1_b(int y_i_1, int y_i, int i) const {
			int index = _extractor->feature_id_identical_1_b(y_i_1, y_i, i);
			return _parameter->_weights[index];
		}
		double CRF::w_identical_2_u(int y_i, int i) const {
			int index = _extractor->feature_id_identical_2_u(y_i, i);
			return _parameter->_weights[index];
		}
		double CRF::w_identical_2_b(int y_i_1, int y_i, int i) const {
			int index = _extractor->feature_id_identical_2_b(y_i_1, y_i, i);
			return _parameter->_weights[index];
		}
		double CRF::w_unigram_type_u(int y_i, int type_i) const {
			int index = _extractor->feature_id_unigram_type_u(y_i, type_i);
			return _parameter->_weights[index];
		}
		double CRF::w_unigram_type_b(int y_i_1, int y_i, int type_i) const {
			int index = _extractor->feature_id_unigram_type_b(y_i_1, y_i, type_i);
			return _parameter->_weights[index];
		}
		double CRF::w_bigram_type_u(int y_i, int type_i_1, int type_i) const {
			int index = _extractor->feature_id_bigram_type_u(y_i, type_i_1, type_i);
			return _parameter->_weights[index];
		}
		double CRF::w_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i) const {
			int index = _extractor->feature_id_bigram_type_b(y_i_1, y_i, type_i_1, type_i);
			return _parameter->_weights[index];
		}
		// void CRF::set_w_label_u(int y_i, double value){
		// 	int index = _extractor->feature_id_label_u(y_i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// void CRF::set_w_label_b(int y_i_1, int y_i, double value){
		// 	int index = _extractor->feature_id_label_b(y_i_1, y_i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// void CRF::set_w_unigram_u(int y_i, int i, int x_i, double value){
		// 	int index = _extractor->feature_id_unigram_u(y_i, i, x_i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// void CRF::set_w_unigram_b(int y_i_1, int y_i, int i, int x_i, double value){
		// 	int index = _extractor->feature_id_unigram_b(y_i_1, y_i, i, x_i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// void CRF::set_w_bigram_u(int y_i, int i, int x_i_1, int x_i, double value){
		// 	int index = _extractor->feature_id_bigram_u(y_i, i, x_i_1, x_i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// void CRF::set_w_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i, double value){
		// 	int index = _extractor->feature_id_bigram_b(y_i_1, y_i, i, x_i_1, x_i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// void CRF::set_w_identical_1_u(int y_i, int i, double value){
		// 	int index = _extractor->feature_id_identical_1_u(y_i, i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// void CRF::set_w_identical_1_b(int y_i_1, int y_i, int i, double value){
		// 	int index = _extractor->feature_id_identical_1_b(y_i_1, y_i, i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// void CRF::set_w_identical_2_u(int y_i, int i, double value){
		// 	int index = _extractor->feature_id_identical_2_u(y_i, i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// void CRF::set_w_identical_2_b(int y_i_1, int y_i, int i, double value){
		// 	int index = _extractor->feature_id_identical_2_b(y_i_1, y_i, i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// void CRF::set_w_unigram_type_u(int y_i, int type_i, double value){
		// 	int index = _extractor->feature_id_unigram_type_u(y_i, type_i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// void CRF::set_w_unigram_type_b(int y_i_1, int y_i, int type_i, double value){
		// 	int index = _extractor->feature_id_unigram_type_b(y_i_1, y_i, type_i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// void CRF::set_w_bigram_type_u(int y_i, int type_i_1, int type_i, double value){
		// 	int index = _extractor->feature_id_bigram_type_u(y_i, type_i_1, type_i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// void CRF::set_w_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i, double value){
		// 	int index = _extractor->feature_id_bigram_type_b(y_i_1, y_i, type_i_1, type_i);
		// 	_parameter->set_weight_at_index(index, value);
		// }
		// γ(s, t) ∝ log{P(c_s^{t - 1}|・)}
		// s、tはともに番号なので1から始まる
		// あるノードから別のノードを辿るV字型のパスのコストの合計
		double CRF::compute_gamma(Sentence* sentence, int s, int t){
			if(t <= 1){
				return 0;
			}
			if(s == sentence->size() + 1){	// P(<eos>|・)に相当するポテンシャル
				assert(t == s + 1);
				return compute_path_cost(sentence, s, t, 1, 1);
			}
			assert(s <= sentence->size());
			assert(t <= sentence->size() + 1);
			assert(s < t);
			int repeat = t - s;
			if(repeat == 1){
				return compute_path_cost(sentence, s, t, 1, 1);
			}
			double sum_cost = compute_path_cost(sentence, s, s + 1, 1, 0) + compute_path_cost(sentence, t - 1, t, 0, 1);
			if(repeat == 2){
				return sum_cost;
			}
			for(int i = 0;i < repeat - 2;i++){
				assert(s + i + 2 <= sentence->size());
				sum_cost += compute_path_cost(sentence, s + i + 1, s + i + 2, 0, 0);
			}
			return sum_cost;
		}
		// 隣接するノード間[i-1,i]のパスのコストを計算
		// yはクラス（0か1）
		// iはノードの位置（1スタートなので注意。インデックスではない.ただし実際は隣接ノードが取れるi>=2のみ可）
		// i_1は本来引数にする必要はないがわかりやすさのため
		double CRF::compute_path_cost(Sentence* sentence, int i_1, int i, int y_i_1, int y_i){
			assert(sentence->_features != NULL);
			array<int> &character_ids = sentence->_character_ids;
			wchar_t const* characters = sentence->_characters;
			int character_ids_length = sentence->size();
			assert(i_1 + 1 == i);
			assert(2 <= i);
			assert(i <= character_ids_length + 2);
			assert(y_i == 0 || y_i == 1);
			assert(y_i_1 == 0 || y_i_1 == 1);
			double cost = 0;
			if(i == character_ids_length + 1){	// </s>
				assert(y_i == 1);
			}
			FeatureIndices* features = sentence->_features;
			for(int m = 0;m < features->_num_features_u(i, y_i);m++){
				int k = features->_feature_indices_u[i][y_i][m];
				cost += _parameter->_weights[k];
			}
			for(int m = 0;m < features->_num_features_b(i, y_i_1, y_i);m++){
				int k = features->_feature_indices_b[i][y_i_1][y_i][m];
				cost += _parameter->_weights[k];
			}

			#ifdef __DEBUG__
				double true_cost = 0;
				true_cost += _compute_cost_label_features(y_i_1, y_i);
				true_cost += _compute_cost_unigram_features(character_ids, character_ids_length, i, y_i_1, y_i);
				true_cost += _compute_cost_bigram_features(character_ids, character_ids_length, i, y_i_1, y_i);
				true_cost += _compute_cost_identical_1_features(character_ids, character_ids_length, i, y_i_1, y_i);
				true_cost += _compute_cost_identical_2_features(character_ids, character_ids_length, i, y_i_1, y_i);
				true_cost += _compute_cost_unigram_and_bigram_type_features(character_ids, characters, character_ids_length, i, y_i_1, y_i);
				assert(std::abs(true_cost - cost) < 1e-12);
			#endif

			return cost;
		}
		double CRF::_compute_cost_label_features(int y_i_1, int y_i){
			double cost = 0;
			cost += w_label_u(y_i);
			cost += w_label_b(y_i_1, y_i);
			return cost;
		}
		double CRF::_compute_cost_unigram_features(array<int> &character_ids, int character_ids_length, int i, int y_i_1, int y_i){
			double cost = 0;
			int r_start = std::max(1, i + _extractor->_x_unigram_start);
			int r_end = std::min(character_ids_length + 2, i + _extractor->_x_unigram_end);	// <eos>2つを考慮
			for(int r = r_start;r <= r_end;r++){
				int pos = r - i - _extractor->_x_unigram_start + 1;	// [1, _x_range_unigram]
				int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
				cost += w_unigram_u(y_i, pos, x_i);
				cost += w_unigram_b(y_i_1, y_i, pos, x_i);
			}
			return cost;
		}
		double CRF::_compute_cost_bigram_features(array<int> &character_ids, int character_ids_length, int i, int y_i_1, int y_i){
			double cost = 0;
			int r_start = std::max(2, i + _extractor->_x_bigram_start);
			int r_end = std::min(character_ids_length + 2, i + _extractor->_x_bigram_end);
			for(int r = r_start;r <= r_end;r++){
				int pos = r - i - _extractor->_x_bigram_start + 1;	// [1, _x_range_bigram]
				int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
				int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : SPECIAL_CHARACTER_END;
				cost += w_bigram_u(y_i, pos, x_i_1, x_i);
				cost += w_bigram_b(y_i_1, y_i, pos, x_i_1, x_i);
			}
			return cost;
		}
		double CRF::_compute_cost_identical_1_features(array<int> &character_ids, int character_ids_length, int i, int y_i_1, int y_i){
			double cost = 0;
			int r_start = std::max(2, i + _extractor->_x_identical_1_start);
			int r_end = std::min(character_ids_length + 2, i + _extractor->_x_identical_1_end);
			for(int r = r_start;r <= r_end;r++){
				int pos = r - i - _extractor->_x_identical_1_start + 1;	// [1, _x_range_identical_1]
				int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
				int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : SPECIAL_CHARACTER_END;
				if(x_i == x_i_1){
					cost += w_identical_1_u(y_i, pos);
					cost += w_identical_1_b(y_i_1, y_i, pos);
				}
			}
			return cost;
		}
		double CRF::_compute_cost_identical_2_features(array<int> &character_ids, int character_ids_length, int i, int y_i_1, int y_i){
			double cost = 0;
			int r_start = std::max(3, i + _extractor->_x_identical_2_start);
			int r_end = std::min(character_ids_length + 2, i + _extractor->_x_identical_2_end);
			for(int r = r_start;r <= r_end;r++){
				int pos = r - i - _extractor->_x_identical_2_start + 1;	// [1, _x_range_identical_2]
				int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
				int x_i_2 = (r - 2 <= character_ids_length) ? character_ids[r - 3] : SPECIAL_CHARACTER_END;
				if(x_i == x_i_2){
					cost += w_identical_2_u(y_i, pos);
					cost += w_identical_2_b(y_i_1, y_i, pos);
				}
			}
			return cost;
		}
		double CRF::_compute_cost_unigram_and_bigram_type_features(array<int> &character_ids, wchar_t const* characters, int character_ids_length, int i, int y_i_1, int y_i){
			assert(i > 1);
			double cost = 0;
			int type_i = (i <= character_ids_length) ? ctype::get_type(characters[i - 1]) : CTYPE_UNKNOWN;
			int type_i_1 = (i - 1 <= character_ids_length) ? ctype::get_type(characters[i - 2]) : CTYPE_UNKNOWN;
			cost += w_unigram_type_u(y_i, type_i);
			cost += w_unigram_type_b(y_i_1, y_i, type_i);
			cost += w_bigram_type_u(y_i, type_i_1, type_i);
			cost += w_bigram_type_b(y_i_1, y_i, type_i_1, type_i);
			return cost;
		}
		double CRF::compute_log_p_y_given_sentence(Sentence* sentence){
			double log_py_s = 0;
			for(int i = 2;i < sentence->get_num_segments() - 1;i++){
				int s = sentence->_start[i] + 1; // インデックスから番号へ
				int t = s + sentence->_segments[i];
				double gamma = compute_gamma(sentence, s, t);
				log_py_s += gamma;
			}
			// <eos>
			log_py_s += compute_gamma(sentence, sentence->size() + 1, sentence->size() + 2);
			return log_py_s;
		}
		FeatureIndices* CRF::extract_features(Sentence* sentence, bool generate_feature_id_if_needed){
			return _extractor->extract(sentence, generate_feature_id_if_needed);
		}
		template <class Archive>
		void CRF::serialize(Archive &ar, unsigned int version)
		{
			ar & _extractor;
			ar & _parameter;
		}
		template void CRF::serialize(boost::archive::binary_iarchive &ar, unsigned int version);
		template void CRF::serialize(boost::archive::binary_oarchive &ar, unsigned int version);
	}
}
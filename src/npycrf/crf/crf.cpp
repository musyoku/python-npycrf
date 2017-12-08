#include <iostream>
#include "../ctype.h"
#include "../sampler.h"
#include "crf.h"
#include "features.h"

namespace npycrf {
	namespace crf {
		CRF::CRF(int num_character_ids,
				 int num_character_types,
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
			_num_character_ids = num_character_ids;
			_num_character_types = num_character_types;
			_bias = 0;

			assert(feature_x_unigram_start <= feature_x_unigram_end);
			assert(feature_x_bigram_start <= feature_x_bigram_end);
			assert(feature_x_identical_1_start <= feature_x_identical_1_end);
			assert(feature_x_identical_2_start <= feature_x_identical_2_end);

			 _x_unigram_start = feature_x_unigram_start;
			 _x_unigram_end = feature_x_unigram_end;
			 _x_bigram_start = feature_x_bigram_start;
			 _x_bigram_end = feature_x_bigram_end;
			 _x_identical_1_start = feature_x_identical_1_start;
			 _x_identical_1_end = feature_x_identical_1_end;
			 _x_identical_2_start = feature_x_identical_2_start;
			 _x_identical_2_end = feature_x_identical_2_end;

			_x_range_unigram = feature_x_unigram_end - feature_x_unigram_start + 1;
			_x_range_bigram = feature_x_bigram_end - feature_x_bigram_start + 1;
			_x_range_identical_1 = feature_x_identical_1_end - feature_x_identical_1_start + 1;
			_x_range_identical_2 = feature_x_identical_2_end - feature_x_identical_2_start + 1;

			// (y_i), (y_{i-1}, y_i)
			_w_size_label_u = 2;
			_w_size_label_b = 2 * 2;
			// (y_i, i, x_i), (y_{i-1}, y_i, i, x_i)
			_w_size_unigram_u = 2 * _x_range_unigram * num_character_ids;
			_w_size_unigram_b = 2 * 2 * _x_range_unigram * num_character_ids;
			// (y_i, i, x_{i-1}, x_i), (y_{i-1}, y_i, i, x_{i-1}, x_i)
			_w_size_bigram_u = 2 * _x_range_bigram * num_character_ids * num_character_ids;
			_w_size_bigram_b = 2 * 2 * _x_range_bigram * num_character_ids * num_character_ids;
			// (y_i, i), (y_{i-1}, y_i, i)
			_w_size_identical_1_u = 2 * _x_range_identical_1;
			_w_size_identical_1_b = 2 * 2 * _x_range_identical_1;
			// (y_i, i), (y_{i-1}, y_i, i)
			_w_size_identical_2_u = 2 * _x_range_identical_2;
			_w_size_identical_2_b = 2 * 2 * _x_range_identical_2;
			// (y_i, type)
			_w_size_unigram_type_u = 2 * num_character_types;
			_w_size_unigram_type_b = 2 * 2 * num_character_types;
			// (y_i, type, type), (y_{i-1}, y_i, type, type)
			_w_size_bigram_type_u = 2 * num_character_types * num_character_types;
			_w_size_bigram_type_b = 2 * 2 * num_character_types * num_character_types;

			_weight_size = _w_size_label_u + _w_size_label_b
							 + _w_size_unigram_u + _w_size_unigram_b
							 + _w_size_bigram_u + _w_size_bigram_b 
							 + _w_size_identical_1_u + _w_size_identical_1_b 
							 + _w_size_identical_2_u + _w_size_identical_2_b 
							 + _w_size_unigram_type_u + _w_size_unigram_type_b 
							 + _w_size_bigram_type_u + _w_size_bigram_type_b;

			_weight = new double[_weight_size];
			for(int i = 0;i < _weight_size;i++){
				_weight[i] = sampler::normal(0, sigma);
				// _weight[i] = 1;
			}

			_offset_w_label_u = 0;
			_offset_w_label_b = _w_size_label_u;
			_offset_w_unigram_u = _w_size_label_u + _w_size_label_b;
			_offset_w_unigram_b = _w_size_label_u + _w_size_label_b
									+ _w_size_unigram_u;
			_offset_w_bigram_u = _w_size_label_u + _w_size_label_b
									+ _w_size_unigram_u + _w_size_unigram_b;
			_offset_w_bigram_b = _w_size_label_u + _w_size_label_b
									+ _w_size_unigram_u + _w_size_unigram_b
									+ _w_size_bigram_u;
			_offset_w_identical_1_u = _w_size_label_u + _w_size_label_b
									+ _w_size_unigram_u + _w_size_unigram_b
									+ _w_size_bigram_u + _w_size_bigram_b;
			_offset_w_identical_1_b = _w_size_label_u + _w_size_label_b
									+ _w_size_unigram_u + _w_size_unigram_b
									+ _w_size_bigram_u + _w_size_bigram_b
									+ _w_size_identical_1_u;
			_offset_w_identical_2_u = _w_size_label_u + _w_size_label_b
									+ _w_size_unigram_u + _w_size_unigram_b
									+ _w_size_bigram_u + _w_size_bigram_b
									+ _w_size_identical_1_u + _w_size_identical_1_b;
			_offset_w_identical_2_b = _w_size_label_u + _w_size_label_b
									+ _w_size_unigram_u + _w_size_unigram_b
									+ _w_size_bigram_u + _w_size_bigram_b
									+ _w_size_identical_1_u + _w_size_identical_1_b
									+ _w_size_identical_2_u;
			_offset_w_unigram_type_u = _w_size_label_u + _w_size_label_b
									+ _w_size_unigram_u + _w_size_unigram_b
									+ _w_size_bigram_u + _w_size_bigram_b
									+ _w_size_identical_1_u + _w_size_identical_1_b
									+ _w_size_identical_2_u + _w_size_identical_2_b;
			_offset_w_unigram_type_b = _w_size_label_u + _w_size_label_b
									+ _w_size_unigram_u + _w_size_unigram_b
									+ _w_size_bigram_u + _w_size_bigram_b
									+ _w_size_identical_1_u + _w_size_identical_1_b
									+ _w_size_identical_2_u + _w_size_identical_2_b
									+ _w_size_unigram_type_u;
			_offset_w_bigram_type_u = _w_size_label_u + _w_size_label_b
									+ _w_size_unigram_u + _w_size_unigram_b
									+ _w_size_bigram_u + _w_size_bigram_b
									+ _w_size_identical_1_u + _w_size_identical_1_b
									+ _w_size_identical_2_u + _w_size_identical_2_b
									+ _w_size_unigram_type_u + _w_size_unigram_type_b;
			_offset_w_bigram_type_b = _w_size_label_u + _w_size_label_b
									+ _w_size_unigram_u + _w_size_unigram_b
									+ _w_size_bigram_u + _w_size_bigram_b
									+ _w_size_identical_1_u + _w_size_identical_1_b
									+ _w_size_identical_2_u + _w_size_identical_2_b
									+ _w_size_unigram_type_u + _w_size_unigram_type_b
									+ _w_size_bigram_type_u;
		}
		CRF::~CRF(){
			delete[] _weight;
		}
		double CRF::bias(){
			return _bias;
		}
		int CRF::_index_w_label_u(int y_i){
			int index = y_i;
			assert(index < _w_size_label_u);
			return index + _offset_w_label_u;
		}
		int CRF::_index_w_label_b(int y_i_1, int y_i){
			int index =  y_i_1 * 2 + y_i;
			assert(index < _w_size_label_b);
			return index + _offset_w_label_b;
		}
		int CRF::_index_w_unigram_u(int y_i, int i, int x_i){
			assert(x_i < _num_character_ids);
			assert(1 <= i && i <= _x_range_unigram);
			int index = x_i * _x_range_unigram * 2 + (i - 1) * 2 + y_i;
			assert(index < _w_size_unigram_u);
			return index + _offset_w_unigram_u;
		}
		int CRF::_index_w_unigram_b(int y_i_1, int y_i, int i, int x_i){
			assert(x_i < _num_character_ids);
			assert(1 <= i && i <= _x_range_unigram);
			int index = x_i * _x_range_unigram * 2 * 2 + (i - 1) * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_unigram_b);
			return index + _offset_w_unigram_b;
		}
		int CRF::_index_w_bigram_u(int y_i, int i, int x_i_1, int x_i){
			assert(x_i_1 < _num_character_ids);
			assert(x_i < _num_character_ids);
			assert(1 <= i && i <= _x_range_bigram);
			int index = x_i * _num_character_ids * _x_range_bigram * 2 + x_i_1 * _x_range_bigram * 2 + (i - 1) * 2 + y_i;
			assert(index < _w_size_bigram_u);
			return index + _offset_w_bigram_u;
		}
		int CRF::_index_w_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i){
			assert(x_i_1 < _num_character_ids);
			assert(x_i < _num_character_ids);
			assert(1 <= i && i <= _x_range_bigram);
			int index = x_i * _num_character_ids * _x_range_bigram * 2 * 2 + x_i_1 * _x_range_bigram * 2 * 2 + (i - 1) * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_bigram_b);
			return index + _offset_w_bigram_b;
		}
		int CRF::_index_w_identical_1_u(int y_i, int i){
			assert(1 <= i && i <= _x_range_identical_1);
			int index = (i - 1) * 2 + y_i;
			assert(index < _w_size_identical_1_u);
			return index + _offset_w_identical_1_u;
		}
		int CRF::_index_w_identical_1_b(int y_i_1, int y_i, int i){
			assert(1 <= i && i <= _x_range_identical_1);
			int index = (i - 1) * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_identical_1_b);
			return index + _offset_w_identical_1_b;
		}
		int CRF::_index_w_identical_2_u(int y_i, int i){
			assert(1 <= i && i <= _x_range_identical_2);
			int index = (i - 1) * 2 + y_i;
			assert(index < _w_size_identical_2_u);
			return index + _offset_w_identical_2_u;
		}
		int CRF::_index_w_identical_2_b(int y_i_1, int y_i, int i){
			assert(1 <= i && i <= _x_range_identical_2);
			int index = (i - 1) * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_identical_2_b);
			return index + _offset_w_identical_2_b;
		}
		int CRF::_index_w_unigram_type_u(int y_i, int type_i){
			int index = type_i * 2 + y_i;
			assert(index < _w_size_unigram_type_u);
			return index + _offset_w_unigram_type_u;
		}
		int CRF::_index_w_unigram_type_b(int y_i_1, int y_i, int type_i){
			assert(type_i < _num_character_types);
			int index = type_i * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_unigram_type_b);
			return index + _offset_w_unigram_type_b;
		}
		int CRF::_index_w_bigram_type_u(int y_i, int type_i_1, int type_i){
			assert(type_i_1 < _num_character_types);
			assert(type_i < _num_character_types);
			int index = type_i * _num_character_types * 2 + type_i_1 * 2 + y_i;
			assert(index < _w_size_bigram_type_u);
			return index + _offset_w_bigram_type_u;
		}
		int CRF::_index_w_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i){
			assert(type_i_1 < _num_character_types);
			assert(type_i < _num_character_types);
			int index = type_i * _num_character_types * 2 * 2 + type_i_1 * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_bigram_type_b);
			return index + _offset_w_bigram_type_b;
		}
		double CRF::w_label_u(int y_i){
			int index = _index_w_label_u(y_i);
			return _weight[index];
		}
		double CRF::w_label_b(int y_i_1, int y_i){
			int index = _index_w_label_b(y_i_1, y_i);
			return _weight[index];
		}
		double CRF::w_unigram_u(int y_i, int i, int x_i){
			assert(x_i < _num_character_ids);
			int index = _index_w_unigram_u(y_i, i, x_i);
			return _weight[index];
		}
		double CRF::w_unigram_b(int y_i_1, int y_i, int i, int x_i){
			assert(x_i < _num_character_ids);
			int index = _index_w_unigram_b(y_i_1, y_i, i, x_i);
			return _weight[index];
		}
		double CRF::w_bigram_u(int y_i, int i, int x_i_1, int x_i){
			assert(x_i_1 < _num_character_ids);
			assert(x_i < _num_character_ids);
			int index = _index_w_bigram_u(y_i, i, x_i_1, x_i);
			return _weight[index];
		}
		double CRF::w_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i){
			assert(x_i_1 < _num_character_ids);
			assert(x_i < _num_character_ids);
			int index = _index_w_bigram_b(y_i_1, y_i, i, x_i_1, x_i);
			return _weight[index];
		}
		double CRF::w_identical_1_u(int y_i, int i){
			int index = _index_w_identical_1_u(y_i, i);
			return _weight[index];
		}
		double CRF::w_identical_1_b(int y_i_1, int y_i, int i){
			int index = _index_w_identical_1_b(y_i_1, y_i, i);
			return _weight[index];
		}
		double CRF::w_identical_2_u(int y_i, int i){
			int index = _index_w_identical_2_u(y_i, i);
			return _weight[index];
		}
		double CRF::w_identical_2_b(int y_i_1, int y_i, int i){
			int index = _index_w_identical_2_b(y_i_1, y_i, i);
			return _weight[index];
		}
		double CRF::w_unigram_type_u(int y_i, int type_i){
			assert(type_i < _num_character_types);
			int index = _index_w_unigram_type_u(y_i, type_i);
			return _weight[index];
		}
		double CRF::w_unigram_type_b(int y_i_1, int y_i, int type_i){
			assert(type_i < _num_character_types);
			int index = _index_w_unigram_type_b(y_i_1, y_i, type_i);
			return _weight[index];
		}
		double CRF::w_bigram_type_u(int y_i, int type_i_1, int type_i){
			assert(type_i_1 < _num_character_types);
			assert(type_i < _num_character_types);
			int index = _index_w_bigram_type_u(y_i, type_i_1, type_i);
			return _weight[index];
		}
		double CRF::w_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i){
			assert(type_i_1 < _num_character_types);
			assert(type_i < _num_character_types);
			int index = _index_w_bigram_type_b(y_i_1, y_i, type_i_1, type_i);
			return _weight[index];
		}
		void CRF::set_w_label_u(int y_i, double value){
			int index = _index_w_label_u(y_i);
			_weight[index] = value;
		}
		void CRF::set_w_label_b(int y_i_1, int y_i, double value){
			int index = _index_w_label_b(y_i_1, y_i);
			_weight[index] = value;
		}
		void CRF::set_w_unigram_u(int y_i, int i, int x_i, double value){
			int index = _index_w_unigram_u(y_i, i, x_i);
			_weight[index] = value;
		}
		void CRF::set_w_unigram_b(int y_i_1, int y_i, int i, int x_i, double value){
			int index = _index_w_unigram_b(y_i_1, y_i, i, x_i);
			_weight[index] = value;
		}
		void CRF::set_w_bigram_u(int y_i, int i, int x_i_1, int x_i, double value){
			int index = _index_w_bigram_u(y_i, i, x_i_1, x_i);
			_weight[index] = value;
		}
		void CRF::set_w_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i, double value){
			int index = _index_w_bigram_b(y_i_1, y_i, i, x_i_1, x_i);
			_weight[index] = value;
		}
		void CRF::set_w_identical_1_u(int y_i, int i, double value){
			int index = _index_w_identical_1_u(y_i, i);
			_weight[index] = value;
		}
		void CRF::set_w_identical_1_b(int y_i_1, int y_i, int i, double value){
			int index = _index_w_identical_1_b(y_i_1, y_i, i);
			_weight[index] = value;
		}
		void CRF::set_w_identical_2_u(int y_i, int i, double value){
			int index = _index_w_identical_2_u(y_i, i);
			_weight[index] = value;
		}
		void CRF::set_w_identical_2_b(int y_i_1, int y_i, int i, double value){
			int index = _index_w_identical_2_b(y_i_1, y_i, i);
			_weight[index] = value;
		}
		void CRF::set_w_unigram_type_u(int y_i, int type_i, double value){
			int index = _index_w_unigram_type_u(y_i, type_i);
			_weight[index] = value;
		}
		void CRF::set_w_unigram_type_b(int y_i_1, int y_i, int type_i, double value){
			int index = _index_w_unigram_type_b(y_i_1, y_i, type_i);
			_weight[index] = value;
		}
		void CRF::set_w_bigram_type_u(int y_i, int type_i_1, int type_i, double value){
			int index = _index_w_bigram_type_u(y_i, type_i_1, type_i);
			_weight[index] = value;
		}
		void CRF::set_w_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i, double value){
			int index = _index_w_bigram_type_b(y_i_1, y_i, type_i_1, type_i);
			_weight[index] = value;
		}
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
			int const* character_ids = sentence->_character_ids;
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
			if(sentence->_features == NULL){
				cost += _compute_cost_label_features(y_i_1, y_i);
				cost += _compute_cost_unigram_features(character_ids, character_ids_length, i, y_i_1, y_i);
				cost += _compute_cost_bigram_features(character_ids, character_ids_length, i, y_i_1, y_i);
				cost += _compute_cost_identical_1_features(character_ids, character_ids_length, i, y_i_1, y_i);
				cost += _compute_cost_identical_2_features(character_ids, character_ids_length, i, y_i_1, y_i);
				cost += _compute_cost_unigram_and_bigram_type_features(character_ids, characters, character_ids_length, i, y_i_1, y_i);
			}else{
				FeatureIndices* features = sentence->_features;
				for(int m = 0;m < features->_num_features_u[i][y_i];m++){
					int k = features->_feature_indices_u[i][y_i][m];
					cost += _weight[k];
				}
				for(int m = 0;m < features->_num_features_b[i][y_i_1][y_i];m++){
					int k = features->_feature_indices_b[i][y_i_1][y_i][m];
					cost += _weight[k];
				}
				#ifdef __DEBUG__
					double true_cost = 0;
					true_cost += _compute_cost_label_features(y_i_1, y_i);
					true_cost += _compute_cost_unigram_features(character_ids, character_ids_length, i, y_i_1, y_i);
					true_cost += _compute_cost_bigram_features(character_ids, character_ids_length, i, y_i_1, y_i);
					true_cost += _compute_cost_identical_1_features(character_ids, character_ids_length, i, y_i_1, y_i);
					true_cost += _compute_cost_identical_2_features(character_ids, character_ids_length, i, y_i_1, y_i);
					true_cost += _compute_cost_unigram_and_bigram_type_features(character_ids, characters, character_ids_length, i, y_i_1, y_i);
					std::cout << true_cost << " == " << cost << std::endl;
					assert(true_cost == cost);
				#endif
			}
			return cost;
		}
		double CRF::_compute_cost_label_features(int y_i_1, int y_i){
			double cost = 0;
			cost += w_label_u(y_i);
			cost += w_label_b(y_i_1, y_i);
			return cost;
		}
		double CRF::_compute_cost_unigram_features(int const* character_ids, int character_ids_length, int i, int y_i_1, int y_i){
			double cost = 0;
			int r_start = std::max(1, i + _x_unigram_start);
			int r_end = std::min(character_ids_length + 2, i + _x_unigram_end);	// <eos>2つを考慮
			for(int r = r_start;r <= r_end;r++){
				int pos = r - i - _x_unigram_start + 1;	// [1, _x_range_unigram]
				int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
				cost += w_unigram_u(y_i, pos, x_i);
				cost += w_unigram_b(y_i_1, y_i, pos, x_i);
			}
			return cost;
		}
		double CRF::_compute_cost_bigram_features(int const* character_ids, int character_ids_length, int i, int y_i_1, int y_i){
			double cost = 0;
			int r_start = std::max(2, i + _x_bigram_start);
			int r_end = std::min(character_ids_length + 2, i + _x_bigram_end);
			for(int r = r_start;r <= r_end;r++){
				int pos = r - i - _x_bigram_start + 1;	// [1, _x_range_bigram]
				int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
				int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : CHARACTER_ID_EOS;
				cost += w_bigram_u(y_i, pos, x_i_1, x_i);
				cost += w_bigram_b(y_i_1, y_i, pos, x_i_1, x_i);
			}
			return cost;
		}
		double CRF::_compute_cost_identical_1_features(int const* character_ids, int character_ids_length, int i, int y_i_1, int y_i){
			double cost = 0;
			int r_start = std::max(2, i + _x_identical_1_start);
			int r_end = std::min(character_ids_length + 2, i + _x_identical_1_end);
			for(int r = r_start;r <= r_end;r++){
				int pos = r - i - _x_identical_1_start + 1;	// [1, _x_range_identical_1]
				int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
				int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : CHARACTER_ID_EOS;
				if(x_i == x_i_1){
					cost += w_identical_1_u(y_i, pos);
					cost += w_identical_1_b(y_i_1, y_i, pos);
				}
			}
			return cost;
		}
		double CRF::_compute_cost_identical_2_features(int const* character_ids, int character_ids_length, int i, int y_i_1, int y_i){
			double cost = 0;
			int r_start = std::max(3, i + _x_identical_2_start);
			int r_end = std::min(character_ids_length + 2, i + _x_identical_2_end);
			for(int r = r_start;r <= r_end;r++){
				int pos = r - i - _x_identical_2_start + 1;	// [1, _x_range_identical_2]
				int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
				int x_i_2 = (r - 2 <= character_ids_length) ? character_ids[r - 3] : CHARACTER_ID_EOS;
				if(x_i == x_i_2){
					cost += w_identical_2_u(y_i, pos);
					cost += w_identical_2_b(y_i_1, y_i, pos);
				}
			}
			return cost;
		}
		double CRF::_compute_cost_unigram_and_bigram_type_features(int const* character_ids, wchar_t const* characters, int character_ids_length, int i, int y_i_1, int y_i){
			assert(i > 1);
			double cost = 0;
			wchar_t char_i = (i <= character_ids_length) ? characters[i - 1] : CHARACTER_ID_EOS;
			wchar_t char_i_1 = (i - 1 <= character_ids_length) ? characters[i - 2] : CHARACTER_ID_EOS;
			int type_i = (i <= character_ids_length) ? ctype::get_type(char_i) : CTYPE_UNKNOWN;
			int type_i_1 = (i - 1 <= character_ids_length) ? ctype::get_type(char_i_1) : CTYPE_UNKNOWN;
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
		void CRF::extract_features(Sentence* sentence){
			assert(sentence->_features == NULL);

			int const* character_ids = sentence->_character_ids;
			wchar_t const* characters = sentence->_characters;
			int character_ids_length = sentence->size();

			// CRFのラベルunigram素性の数
			int** num_crf_features_u = new int*[character_ids_length + 3];
			int*** crf_feature_indices_u = new int**[character_ids_length + 3];

			// ラベルunigram素性
			for(int i = 2;i <= character_ids_length;i++){
				num_crf_features_u[i] = new int[2];
				crf_feature_indices_u[i] = new int*[2];
				for(int y_i = 0;y_i <= 1;y_i++){
					std::vector<int> indices_u;
					int r_start, r_end;
					// ラベル素性
					indices_u.push_back(_index_w_label_u(y_i));
					// 文字unigram素性
					r_start = std::max(1, i + _x_unigram_start);
					r_end = std::min(character_ids_length + 2, i + _x_unigram_end);	// <eos>2つを考慮
					for(int r = r_start;r <= r_end;r++){
						int pos = r - i - _x_unigram_start + 1;	// [1, _x_range_unigram]
						int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
						indices_u.push_back(_index_w_unigram_u(y_i, pos, x_i));
					}
					// 文字bigram素性
					r_start = std::max(2, i + _x_bigram_start);
					r_end = std::min(character_ids_length + 2, i + _x_bigram_end);
					for(int r = r_start;r <= r_end;r++){
						int pos = r - i - _x_bigram_start + 1;	// [1, _x_range_bigram]
						int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
						int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : CHARACTER_ID_EOS;
						indices_u.push_back(_index_w_bigram_u(y_i, pos, x_i_1, x_i));
					}
					// identical_1素性
					r_start = std::max(2, i + _x_identical_1_start);
					r_end = std::min(character_ids_length + 2, i + _x_identical_1_end);
					for(int r = r_start;r <= r_end;r++){
						int pos = r - i - _x_identical_1_start + 1;	// [1, _x_range_identical_1]
						int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
						int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : CHARACTER_ID_EOS;
						if(x_i == x_i_1){
							indices_u.push_back(_index_w_identical_1_u(y_i, pos));
						}
					}
					// identical_2素性
					r_start = std::max(3, i + _x_identical_2_start);
					r_end = std::min(character_ids_length + 2, i + _x_identical_2_end);
					for(int r = r_start;r <= r_end;r++){
						int pos = r - i - _x_identical_2_start + 1;	// [1, _x_range_identical_2]
						int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
						int x_i_2 = (r - 2 <= character_ids_length) ? character_ids[r - 3] : CHARACTER_ID_EOS;
						if(x_i == x_i_2){
							indices_u.push_back(_index_w_identical_2_u(y_i, pos));
						}
					}
					// 文字種unigram・bigram素性
					wchar_t char_i = (i <= character_ids_length) ? characters[i - 1] : CHARACTER_ID_EOS;
					wchar_t char_i_1 = (i - 1 <= character_ids_length) ? characters[i - 2] : CHARACTER_ID_EOS;
					int type_i = (i <= character_ids_length) ? ctype::get_type(char_i) : CTYPE_UNKNOWN;
					int type_i_1 = (i - 1 <= character_ids_length) ? ctype::get_type(char_i_1) : CTYPE_UNKNOWN;
					indices_u.push_back(_index_w_unigram_type_u(y_i, type_i));

					num_crf_features_u[i][y_i] = indices_u.size();
					crf_feature_indices_u[i][y_i] = new int[indices_u.size()];
					for(int n = 0;n < indices_u.size();n++){
						crf_feature_indices_u[i][y_i][n] = indices_u[n];
					}
				}
			}

			// CRFのラベルbigram素性の数
			int*** num_crf_features_b = new int**[character_ids_length + 3];
			for(int i = 0;i < character_ids_length + 3;i++){
				num_crf_features_b[i] = new int*[2];
				num_crf_features_b[i][0] = new int[2];
				num_crf_features_b[i][1] = new int[2];
				num_crf_features_b[i][0][0] = 0;
				num_crf_features_b[i][0][1] = 0;
				num_crf_features_b[i][1][0] = 0;
				num_crf_features_b[i][1][1] = 0;
			}
			int**** crf_feature_indices_b = new int***[character_ids_length + 3];

			// ラベルbigram素性
			for(int i = 2;i <= character_ids_length;i++){
				crf_feature_indices_b[i] = new int**[2];
				for(int y_i_1 = 0;y_i_1 <= 1;y_i_1++){
					crf_feature_indices_b[i][y_i_1] = new int*[2];
					for(int y_i = 0;y_i <= 1;y_i++){
						std::vector<int> indices_b;
						int r_start, r_end;
						// ラベル素性
						indices_b.push_back(_index_w_label_b(y_i_1, y_i));
						// 文字unigram素性
						r_start = std::max(1, i + _x_unigram_start);
						r_end = std::min(character_ids_length + 2, i + _x_unigram_end);	// <eos>2つを考慮
						for(int r = r_start;r <= r_end;r++){
							int pos = r - i - _x_unigram_start + 1;	// [1, _x_range_unigram]
							int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
							indices_b.push_back(_index_w_unigram_b(y_i_1, y_i, pos, x_i));
						}
						// 文字bigram素性
						r_start = std::max(2, i + _x_bigram_start);
						r_end = std::min(character_ids_length + 2, i + _x_bigram_end);
						for(int r = r_start;r <= r_end;r++){
							int pos = r - i - _x_bigram_start + 1;	// [1, _x_range_bigram]
							int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
							int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : CHARACTER_ID_EOS;
							indices_b.push_back(_index_w_bigram_b(y_i_1, y_i, pos, x_i_1, x_i));
						}
						// identical_1素性
						r_start = std::max(2, i + _x_identical_1_start);
						r_end = std::min(character_ids_length + 2, i + _x_identical_1_end);
						for(int r = r_start;r <= r_end;r++){
							int pos = r - i - _x_identical_1_start + 1;	// [1, _x_range_identical_1]
							int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
							int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : CHARACTER_ID_EOS;
							if(x_i == x_i_1){
								indices_b.push_back(_index_w_identical_1_b(y_i_1, y_i, pos));
							}
						}
						// identical_2素性
						r_start = std::max(3, i + _x_identical_2_start);
						r_end = std::min(character_ids_length + 2, i + _x_identical_2_end);
						for(int r = r_start;r <= r_end;r++){
							int pos = r - i - _x_identical_2_start + 1;	// [1, _x_range_identical_2]
							int x_i = (r <= character_ids_length) ? character_ids[r - 1] : CHARACTER_ID_EOS;
							int x_i_2 = (r - 2 <= character_ids_length) ? character_ids[r - 3] : CHARACTER_ID_EOS;
							if(x_i == x_i_2){
								indices_b.push_back(_index_w_identical_2_b(y_i_1, y_i, pos));
							}
						}
						// 文字種unigram・bigram素性
						wchar_t char_i = (i <= character_ids_length) ? characters[i - 1] : CHARACTER_ID_EOS;
						wchar_t char_i_1 = (i - 1 <= character_ids_length) ? characters[i - 2] : CHARACTER_ID_EOS;
						int type_i = (i <= character_ids_length) ? ctype::get_type(char_i) : CTYPE_UNKNOWN;
						int type_i_1 = (i - 1 <= character_ids_length) ? ctype::get_type(char_i_1) : CTYPE_UNKNOWN;
						indices_b.push_back(_index_w_bigram_type_b(y_i_1, y_i, type_i_1, type_i));

						// 更新
						num_crf_features_b[i][y_i_1][y_i] = indices_b.size();
						crf_feature_indices_b[i][y_i_1][y_i] = new int[indices_b.size()];
						for(int n = 0;n < indices_b.size();n++){
							crf_feature_indices_b[i][y_i_1][y_i][n] = indices_b[n];
						}
					}
				}
			}
			FeatureIndices* features = new FeatureIndices();
			features->_num_features_u = num_crf_features_u;
			features->_num_features_b = num_crf_features_b;
			features->_feature_indices_u = crf_feature_indices_u;
			features->_feature_indices_b = crf_feature_indices_b;
			sentence->_features = features;
		}
		
		template <class Archive>
		void CRF::serialize(Archive &archive, unsigned int version)
		{
			boost::serialization::split_member(archive, *this, version);
		}
		template void CRF::serialize(boost::archive::binary_iarchive &ar, unsigned int version);
		template void CRF::serialize(boost::archive::binary_oarchive &ar, unsigned int version);
		void CRF::save(boost::archive::binary_oarchive &archive, unsigned int version) const {

		}
		void CRF::load(boost::archive::binary_iarchive &archive, unsigned int version) {
			
		}
		
	}
}
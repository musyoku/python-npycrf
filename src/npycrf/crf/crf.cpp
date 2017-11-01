#include <iostream>
#include "../ctype.h"
#include "crf.h"

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
				 int feature_x_identical_2_end)
		{
			_num_character_ids = num_character_ids;
			_num_character_types = num_character_types;
			_bias = 0;
			_x_range_unigram = feature_x_unigram_end - feature_x_unigram_start + 1;
			_x_range_bigram = feature_x_bigram_end - feature_x_bigram_start + 1;
			_x_range_identical_1 = feature_x_identical_1_end - feature_x_identical_1_start + 1;
			_x_range_identical_2 = feature_x_identical_2_end - feature_x_identical_2_start + 1;
			
			// (y_i)
			_w_size_label_u = 2;
			_w_label_u = new double[_w_size_label_u];
			for(int i = 0;i < _w_size_label_u;i++){
				_w_label_u[i] = 0;
			}
			
			// (y_{i-1}, y_i)
			_w_size_label_b = 2 * 2;
			_w_label_b = new double[_w_size_label_b];
			for(int i = 0;i < _w_size_label_b;i++){
				_w_label_b[i] = 0;
			}

			// (y_i, i, x_i)
			_w_size_unigram_u = 2 * _x_range_unigram * num_character_ids;
			_w_unigram_u = new double[_w_size_unigram_u];
			for(int i = 0;i < _w_size_unigram_u;i++){
				_w_unigram_u[i] = 0;
			}
			// (y_{i-1}, y_i, i, x_i)
			_w_size_unigram_b = 2 * 2 * _x_range_unigram * num_character_ids;
			_w_unigram_b = new double[_w_size_unigram_b];
			for(int i = 0;i < _w_size_unigram_b;i++){
				_w_unigram_b[i] = 0;
			}
			// (y_i, i, x_{i-1}, x_i);
			_w_size_bigram_u = 2 * _x_range_bigram * num_character_ids * num_character_ids;
			_w_bigram_u = new double[_w_size_bigram_u];
			for(int i = 0;i < _w_size_bigram_u;i++){
				_w_bigram_u[i] = 0;
			}
			// (y_{i-1}, y_i, i, x_{i-1}, x_i)
			_w_size_bigram_b = 2 * 2 * _x_range_bigram * num_character_ids * num_character_ids;
			_w_bigram_b = new double[_w_size_bigram_b];
			for(int i = 0;i < _w_size_bigram_b;i++){
				_w_bigram_b[i] = 0;
			}
			// (y_i, i)
			_w_size_identical_1_u = 2 * _x_range_identical_1;
			_w_identical_1_u = new double[_w_size_identical_1_u];
			for(int i = 0;i < _w_size_identical_1_u;i++){
				_w_identical_1_u[i] = 0;
			}
			// (y_{i-1}, y_i, i)
			_w_size_identical_1_b = 2 * 2 * _x_range_identical_1;
			_w_identical_1_b = new double[_w_size_identical_1_b];
			for(int i = 0;i < _w_size_identical_1_b;i++){
				_w_identical_1_b[i] = 0;
			}
			// (y_i, i)
			_w_size_identical_2_u = 2 * _x_range_identical_2;
			_w_identical_2_u = new double[_w_size_identical_2_u];
			for(int i = 0;i < _w_size_identical_2_u;i++){
				_w_identical_2_u[i] = 0;
			}
			// (y_{i-1}, y_i, i)
			_w_size_identical_2_b = 2 * 2 * _x_range_identical_2;
			_w_identical_2_b = new double[_w_size_identical_2_b];
			for(int i = 0;i < _w_size_identical_2_b;i++){
				_w_identical_2_b[i] = 0;
			}
			// (y_i, type)
			_w_size_unigram_type_u = 2 * num_character_types;
			_w_unigram_type_u = new double[_w_size_unigram_type_u];
			for(int i = 0;i < _w_size_unigram_type_u;i++){
				_w_unigram_type_u[i] = 0;
			}
			// (y_{i-1}, y_i, type)
			_w_size_unigram_type_b = 2 * 2 * num_character_types;
			_w_unigram_type_b = new double[_w_size_unigram_type_b];
			// (y_i, type, type);
			_w_size_bigram_type_u = 2 * num_character_types * num_character_types;
			_w_bigram_type_u = new double[_w_size_bigram_type_u];
			for(int i = 0;i < _w_size_bigram_type_u;i++){
				_w_bigram_type_u[i] = 0;
			}
			// (y_{i-1}, y_i, type, type)
			_w_size_bigram_type_b = 2 * 2 * num_character_types * num_character_types;
			_w_bigram_type_b = new double[_w_size_bigram_type_b];
			for(int i = 0;i < _w_size_bigram_type_b;i++){
				_w_bigram_type_b[i] = 0;
			}
		}
		CRF::~CRF(){
			delete[] _w_label_u;
			delete[] _w_label_b;
			delete[] _w_unigram_u;
			delete[] _w_unigram_b;
			delete[] _w_bigram_u;
			delete[] _w_bigram_b;
			delete[] _w_identical_1_u;
			delete[] _w_identical_1_b;
			delete[] _w_identical_2_u;
			delete[] _w_identical_2_b;
			delete[] _w_unigram_type_u;
			delete[] _w_unigram_type_b;
			delete[] _w_bigram_type_u;
			delete[] _w_bigram_type_b;
		}
		double CRF::bias(){
			return _bias;
		}
		double CRF::_index_w_label_u(int y_i){
			int index = y_i;
			assert(index < _w_size_label_u);
			return index;
		}
		double CRF::_index_w_label_b(int y_i_1, int y_i){
			int index =  y_i_1 * 2 + y_i;
			assert(index < _w_size_label_b);
			return index;
		}
		double CRF::_index_w_unigram_u(int y_i, int i, int x_i){
			assert(x_i < _num_character_ids);
			assert(1 <= i && i <= _x_range_unigram);
			int index = x_i * _x_range_unigram * 2 + (i - 1) * 2 + y_i;
			assert(index < _w_size_unigram_u);
			return index;
		}
		double CRF::_index_w_unigram_b(int y_i_1, int y_i, int i, int x_i){
			assert(x_i < _num_character_ids);
			assert(1 <= i && i <= _x_range_unigram);
			int index = x_i * _x_range_unigram * 2 * 2 + (i - 1) * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_unigram_b);
			return index;
		}
		double CRF::_index_w_bigram_u(int y_i, int i, int x_i_1, int x_i){
			assert(x_i_1 < _num_character_ids);
			assert(x_i < _num_character_ids);
			assert(1 <= i && i <= _x_range_bigram);
			int index = x_i * _num_character_ids * _x_range_bigram * 2 + x_i_1 * _x_range_bigram * 2 + (i - 1) * 2 + y_i;
			assert(index < _w_size_bigram_u);
			return index;
		}
		double CRF::_index_w_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i){
			assert(x_i_1 < _num_character_ids);
			assert(x_i < _num_character_ids);
			assert(1 <= i && i <= _x_range_bigram);
			int index = x_i * _num_character_ids * _x_range_bigram * 2 * 2 + x_i_1 * _x_range_bigram * 2 * 2 + (i - 1) * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_bigram_b);
			return index;
		}
		double CRF::_index_w_identical_1_u(int y_i, int i){
			assert(1 <= i && i <= _x_range_identical_1);
			int index = (i - 1) * 2 + y_i;
			assert(index < _w_size_identical_1_u);
			return index;
		}
		double CRF::_index_w_identical_1_b(int y_i_1, int y_i, int i){
			assert(1 <= i && i <= _x_range_identical_1);
			int index = (i - 1) * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_identical_1_b);
			return index;
		}
		double CRF::_index_w_identical_2_u(int y_i, int i){
			assert(1 <= i && i <= _x_range_identical_2);
			int index = (i - 1) * 2 + y_i;
			assert(index < _w_size_identical_2_u);
			return index;
		}
		double CRF::_index_w_identical_2_b(int y_i_1, int y_i, int i){
			assert(1 <= i && i <= _x_range_identical_2);
			int index = (i - 1) * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_identical_2_b);
			return index;
		}
		double CRF::_index_w_unigram_type_u(int y_i, int type_i){
			int index = type_i * 2 + y_i;
			assert(index < _w_size_unigram_type_u);
			return index;
		}
		double CRF::_index_w_unigram_type_b(int y_i_1, int y_i, int type_i){
			assert(type_i < _num_character_types);
			int index = type_i * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_unigram_type_b);
			return index;
		}
		double CRF::_index_w_bigram_type_u(int y_i, int type_i_1, int type_i){
			assert(type_i_1 < _num_character_types);
			assert(type_i < _num_character_types);
			int index = type_i * _num_character_types * 2 + type_i_1 * 2 + y_i;
			assert(index < _w_size_bigram_type_u);
			return index;
		}
		double CRF::_index_w_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i){
			assert(type_i_1 < _num_character_types);
			assert(type_i < _num_character_types);
			int index = type_i * _num_character_types * 2 * 2 + type_i_1 * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_bigram_type_b);
			return index;
		}
		double CRF::w_label_u(int y_i){
			int index = _index_w_label_u(y_i);
			return _w_label_u[index];
		}
		double CRF::w_label_b(int y_i_1, int y_i){
			int index = _index_w_label_b(y_i_1, y_i);
			return _w_label_b[index];
		}
		double CRF::w_unigram_u(int y_i, int i, int x_i){
			assert(x_i < _num_character_ids);
			int index = _index_w_unigram_u(y_i, i, x_i);
			return _w_unigram_u[index];
		}
		double CRF::w_unigram_b(int y_i_1, int y_i, int i, int x_i){
			assert(x_i < _num_character_ids);
			int index = _index_w_unigram_b(y_i_1, y_i, i, x_i);
			return _w_unigram_b[index];
		}
		double CRF::w_bigram_u(int y_i, int i, int x_i_1, int x_i){
			assert(x_i_1 < _num_character_ids);
			assert(x_i < _num_character_ids);
			int index = _index_w_bigram_u(y_i, i, x_i_1, x_i);
			return _w_bigram_u[index];
		}
		double CRF::w_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i){
			assert(x_i_1 < _num_character_ids);
			assert(x_i < _num_character_ids);
			int index = _index_w_bigram_b(y_i_1, y_i, i, x_i_1, x_i);
			return _w_bigram_b[index];
		}
		double CRF::w_identical_1_u(int y_i, int i){
			int index = _index_w_identical_1_u(y_i, i);
			return _w_identical_1_u[index];
		}
		double CRF::w_identical_1_b(int y_i_1, int y_i, int i){
			int index = _index_w_identical_1_b(y_i_1, y_i, i);
			return _w_identical_1_b[index];
		}
		double CRF::w_identical_2_u(int y_i, int i){
			int index = _index_w_identical_2_u(y_i, i);
			return _w_identical_2_u[index];
		}
		double CRF::w_identical_2_b(int y_i_1, int y_i, int i){
			int index = _index_w_identical_2_b(y_i_1, y_i, i);
			return _w_identical_2_b[index];
		}
		double CRF::w_unigram_type_u(int y_i, int type_i){
			assert(type_i < _num_character_types);
			int index = _index_w_unigram_type_u(y_i, type_i);
			return _w_unigram_type_u[index];
		}
		double CRF::w_unigram_type_b(int y_i_1, int y_i, int type_i){
			assert(type_i < _num_character_types);
			int index = _index_w_unigram_type_b(y_i_1, y_i, type_i);
			return _w_unigram_type_b[index];
		}
		double CRF::w_bigram_type_u(int y_i, int type_i_1, int type_i){
			assert(type_i_1 < _num_character_types);
			assert(type_i < _num_character_types);
			int index = _index_w_bigram_type_u(y_i, type_i_1, type_i);
			return _w_bigram_type_u[index];
		}
		double CRF::w_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i){
			assert(type_i_1 < _num_character_types);
			assert(type_i < _num_character_types);
			int index = _index_w_bigram_type_b(y_i_1, y_i, type_i_1, type_i);
			return _w_bigram_type_b[index];
		}
		void CRF::set_w_label_u(int y_i, double value){

		}
		void CRF::set_w_label_b(int y_i_1, int y_i, double value){

		}
		void CRF::set_w_unigram_u(int y_i, int i, int x_i, double value){
			int index = _index_w_unigram_u(y_i, i, x_i);
			_w_unigram_u[index] = value;
		}
		void CRF::set_w_unigram_b(int y_i_1, int y_i, int i, int x_i, double value){
			int index = _index_w_unigram_b(y_i_1, y_i, i, x_i);
			_w_unigram_b[index] = value;
		}
		void CRF::set_w_bigram_u(int y_i, int i, int x_i_1, int x_i, double value){
			int index = _index_w_bigram_u(y_i, i, x_i_1, x_i);
			_w_bigram_u[index] = value;
		}
		void CRF::set_w_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i, double value){
			int index = _index_w_bigram_b(y_i_1, y_i, i, x_i_1, x_i);
			_w_bigram_b[index] = value;
		}
		void CRF::set_w_identical_1_u(int y_i, int i, double value){
			int index = _index_w_identical_1_u(y_i, i);
			_w_identical_1_u[index] = value;
		}
		void CRF::set_w_identical_1_b(int y_i_1, int y_i, int i, double value){
			int index = _index_w_identical_1_b(y_i_1, y_i, i);
			_w_identical_1_b[index] = value;
		}
		void CRF::set_w_identical_2_u(int y_i, int i, double value){
			int index = _index_w_identical_2_u(y_i, i);
			_w_identical_2_u[index] = value;
		}
		void CRF::set_w_identical_2_b(int y_i_1, int y_i, int i, double value){
			int index = _index_w_identical_2_b(y_i_1, y_i, i);
			_w_identical_2_b[index] = value;
		}
		void CRF::set_w_unigram_type_u(int y_i, int type_i, double value){
			int index = _index_w_unigram_type_u(y_i, type_i);
			_w_unigram_type_u[index] = value;
		}
		void CRF::set_w_unigram_type_b(int y_i_1, int y_i, int type_i, double value){
			int index = _index_w_unigram_type_b(y_i_1, y_i, type_i);
			_w_unigram_type_b[index] = value;
		}
		void CRF::set_w_bigram_type_u(int y_i, int type_i_1, int type_i, double value){
			int index = _index_w_bigram_type_u(y_i, type_i_1, type_i);
			_w_bigram_type_u[index] = value;
		}
		void CRF::set_w_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i, double value){
			int index = _index_w_bigram_type_b(y_i_1, y_i, type_i_1, type_i);
			_w_bigram_type_b[index] = value;
		}
		// V(t, k, j) = γ(t - k + 1, t + 1) + γ(t - k - j + 1, t - k + 1)
		// tは1スタートであることに注意
		double CRF::compute_trigram_potential(int const* character_ids, wchar_t const* characters, int character_ids_length, int t, int k, int j){







































			return 0;
			
			assert(1 <= t && t <= character_ids_length + 1); // tは<eos>を指すこともある
			assert(1 <= k && k <= character_ids_length);
			assert(0 <= j && j < character_ids_length);
			assert(0 < t);
			if(j == 0){
				return compute_gamma(character_ids, characters, character_ids_length, t - k + 1, t + 1);
			}
			if(t == character_ids_length + 1){
				return compute_gamma(character_ids, characters, character_ids_length, t - k - j + 1, t - k + 1);
			}
			return compute_gamma(character_ids, characters, character_ids_length, t - k + 1, t + 1) + compute_gamma(character_ids, characters, character_ids_length, t - k - j + 1, t - k + 1);
		}
		// γ(s, t)
		// あるノードから別のノードを辿るV字型のパスのコストの合計
		double CRF::compute_gamma(int const* character_ids, wchar_t const* characters, int character_ids_length, int s, int t){
			assert(s <= character_ids_length);
			assert(t <= character_ids_length + 1);
			assert(s < t);
			if(t <= 1){
				return 0;
			}
			int repeat = t - s;
			if(repeat == 1){
				return compute_path_cost(character_ids, characters, character_ids_length, s, t, 1, 1);
			}
			double sum_cost = compute_path_cost(character_ids, characters, character_ids_length, s, s + 1, 1, 0) + compute_path_cost(character_ids, characters, character_ids_length, t - 1, t, 0, 1);
			if(repeat == 2){
				return sum_cost;
			}
			for(int i = 0;i < repeat - 2;i++){
				assert(s + i + 2 <= character_ids_length);
				sum_cost += compute_path_cost(character_ids, characters, character_ids_length, s + i + 1, s + i + 2, 0, 0);
			}
			return sum_cost;
		}
		// 隣接するノード間のパスのコストを計算
		// yはクラス（0か1）
		// iはノードの位置（1スタートなので注意。インデックスではない.ただし実際は隣接ノードが取れるi>=2のみ可）
		// i_1は本来引数にする必要はないがわかりやすさのため
		double CRF::compute_path_cost(int const* character_ids, wchar_t const* characters, int character_ids_length, int i_1, int i, int y_i_1, int y_i){
			assert(i_1 + 1 == i);
			assert(2 <= i);
			assert(i <= character_ids_length + 1);
			assert(y_i == 0 || y_i == 1);
			assert(y_i_1 == 0 || y_i_1 == 1);
			double cost = 0;
			if(i == character_ids_length + 1){	// </s>
				assert(y_i == 1);
			}
			cost += _compute_cost_unigram_features(character_ids, character_ids_length, i, y_i_1, y_i);
			cost += _compute_cost_bigram_features(character_ids, character_ids_length, i, y_i_1, y_i);
			cost += _compute_cost_identical_1_features(character_ids, character_ids_length, i, y_i_1, y_i);
			cost += _compute_cost_identical_2_features(character_ids, character_ids_length, i, y_i_1, y_i);
			cost += _compute_cost_unigram_and_bigram_type_features(character_ids, characters, character_ids_length, i, y_i_1, y_i);
			return cost;
		}
		double CRF::_compute_cost_unigram_features(int const* character_ids, int character_ids_length, int i, int y_i_1, int y_i){
			double cost = 0;
			int r_end = std::max(0, i - _x_range_unigram);
			for(int r = i;r > r_end;r--){
				int pos = i - r + 1;	// [1, _x_range_unigram]
				int x_i = character_ids[r - 1] ? r <= character_ids_length : CHARACTER_ID_EOW;
				cost += w_unigram_u(y_i, pos, x_i);
				cost += w_unigram_b(y_i_1, y_i, pos, x_i);
			}
			return cost;
		}
		double CRF::_compute_cost_bigram_features(int const* character_ids, int character_ids_length, int i, int y_i_1, int y_i){
			double cost = 0;
			int r_end = std::max(1, i - _x_range_bigram);
			for(int r = i;r > r_end;r--){
				int pos = i - r + 1;	// [1, _x_range_bigram]
				int x_i = character_ids[r - 1] ? r <= character_ids_length : CHARACTER_ID_EOW;
				int x_i_1 = character_ids[r - 2];
				cost += w_bigram_u(y_i, pos, x_i_1, x_i);
				cost += w_bigram_b(y_i_1, y_i, pos, x_i_1, x_i);
			}
			return cost;
		}
		double CRF::_compute_cost_identical_1_features(int const* character_ids, int character_ids_length, int i, int y_i_1, int y_i){
			double cost = 0;
			int pos = 1;	// [1, _x_range_identical_1]
			int r_end = std::max(1, i - _x_range_identical_1);
			for(int r = i;r > r_end;r--){
				int x_i = character_ids[r - 1] ? r <= character_ids_length : CHARACTER_ID_EOW;
				int x_i_1 = character_ids[r - 2];
				if(x_i == x_i_1){
					cost += w_identical_1_u(y_i, pos);
					cost += w_identical_1_b(y_i_1, y_i, pos);
				}
				pos++;
			}
			return cost;
		}
		double CRF::_compute_cost_identical_2_features(int const* character_ids, int character_ids_length, int i, int y_i_1, int y_i){
			double cost = 0;
			int pos = 1;	// [1, _x_range_identical_2]
			int r_end = std::max(2, i - _x_range_identical_2);
			for(int r = i;r > r_end;r--){
				int x_i = character_ids[r - 1] ? r <= character_ids_length : CHARACTER_ID_EOW;
				int x_i_2 = character_ids[r - 3];
				if(x_i == x_i_2){
					cost += w_identical_2_u(y_i, pos);
					cost += w_identical_2_b(y_i_1, y_i, pos);
				}
				pos++;
			}
			return cost;
		}
		double CRF::_compute_cost_unigram_and_bigram_type_features(int const* character_ids, wchar_t const* characters, int character_ids_length, int i, int y_i_1, int y_i){
			double cost = 0;
			wchar_t char_i = characters[i - 1] ? i <= character_ids_length : 0;
			wchar_t char_i_1 = characters[i - 2];
			int type_i = ctype::get_type(char_i) ? i <= character_ids_length : CTYPE_UNKNOWN;
			int type_i_1 = ctype::get_type(char_i_1);
			cost += w_unigram_type_u(y_i, type_i);
			cost += w_unigram_type_b(y_i_1, y_i, type_i);
			cost += w_bigram_type_u(y_i, type_i_1, type_i);
			cost += w_bigram_type_b(y_i_1, y_i, type_i_1, type_i);
			return cost;
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
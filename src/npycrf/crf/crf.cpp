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
			_x_range_bigram = feature_x_bigram_end - feature_x_bigram_end + 1;
			_x_range_identical_1 = feature_x_identical_1_end - feature_x_identical_1_start + 1;
			_x_range_identical_2 = feature_x_identical_2_end - feature_x_identical_2_start + 1;

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
		double CRF::bias(){
			return _bias;
		}
		double CRF::_index_w_unigram_u(int y_i, int i, int x_i){
			int index = x_i * _x_range_unigram * 2 + i * 2 + y_i;
			assert(index < _w_size_unigram_u);
			return index;
		}
		double CRF::_index_w_unigram_b(int y_i_1, int y_i, int i, int x_i){
			int index = x_i * _x_range_unigram * 2 * 2 + i * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_unigram_b);
			return index;
		}
		double CRF::_index_w_bigram_u(int y_i, int i, int x_i_1, int x_i){
			int index = x_i * _num_character_ids * _x_range_bigram * 2 + x_i_1 * _x_range_bigram * 2 + i * 2 + y_i;
			assert(index < _w_size_bigram_u);
			return index;
		}
		double CRF::_index_w_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i){
			int index = x_i * _num_character_ids * _x_range_bigram * 2 * 2 + x_i_1 * _x_range_bigram * 2 * 2 + i * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_bigram_b);
			return index;
		}
		double CRF::_index_w_identical_1_u(int y_i, int i){
			int index = i * 2 + y_i;
			assert(index < _w_size_identical_1_u);
			return index;
		}
		double CRF::_index_w_identical_1_b(int y_i_1, int y_i, int i){
			int index = i * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_identical_1_b);
			return index;
		}
		double CRF::_index_w_identical_2_u(int y_i, int i){
			int index = i * 2 + y_i;
			assert(index < _w_size_identical_2_u);
			return index;
		}
		double CRF::_index_w_identical_2_b(int y_i_1, int y_i, int i){
			int index = i * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_identical_2_b);
			return index;
		}
		double CRF::_index_w_unigram_type_u(int y_i, int type_i){
			int index = type_i * 2 + y_i;
			assert(index < _w_size_unigram_type_u);
			return index;
		}
		double CRF::_index_w_unigram_type_b(int y_i_1, int y_i, int type_i){
			int index = type_i * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_unigram_type_b);
			return index;
		}
		double CRF::_index_w_bigram_type_u(int y_i, int type_i_1, int type_i){
			int index = type_i * _num_character_types * 2 + type_i_1 * 2 + y_i;
			assert(index < _w_size_bigram_type_u);
			return index;
		}
		double CRF::_index_w_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i){
			int index = type_i * _num_character_types * 2 * 2 + type_i_1 * 2 * 2 + y_i * 2 + y_i_1;
			assert(index < _w_size_bigram_type_b);
			return index;
		}
		double CRF::w_unigram_u(int y_i, int i, int x_i){
			int index = _index_w_unigram_u(y_i, i, x_i);
			return _w_unigram_u[index];
		}
		double CRF::w_unigram_b(int y_i_1, int y_i, int i, int x_i){
			int index = _index_w_unigram_b(y_i_1, y_i, i, x_i);
			return _w_unigram_b[index];
		}
		double CRF::w_bigram_u(int y_i, int i, int x_i_1, int x_i){
			int index = _index_w_bigram_u(y_i, i, x_i_1, x_i);
			return _w_bigram_u[index];
		}
		double CRF::w_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i){
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
			int index = _index_w_unigram_type_u(y_i, type_i);
			return _w_unigram_type_u[index];
		}
		double CRF::w_unigram_type_b(int y_i_1, int y_i, int type_i){
			int index = _index_w_unigram_type_b(y_i_1, y_i, type_i);
			return _w_unigram_type_b[index];
		}
		double CRF::w_bigram_type_u(int y_i, int type_i_1, int type_i){
			int index = _index_w_bigram_type_u(y_i, type_i_1, type_i);
			return _w_bigram_type_u[index];
		}
		double CRF::w_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i){
			int index = _index_w_bigram_type_b(y_i_1, y_i, type_i_1, type_i);
			return _w_bigram_type_b[index];
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
	}
}
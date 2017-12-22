#include <cassert>
#include "extractor.h"
#include "../../ctype.h"

namespace npycrf {
	namespace crf {
		namespace feature {
			FeatureExtractor::FeatureExtractor(){

			}
			FeatureExtractor::FeatureExtractor(
				int num_character_ids,
				int num_character_types,
				int feature_x_unigram_start,
				int feature_x_unigram_end,
				int feature_x_bigram_start,
				int feature_x_bigram_end,
				int feature_x_identical_1_start,
				int feature_x_identical_1_end,
				int feature_x_identical_2_start,
				int feature_x_identical_2_end
			){
				_num_character_ids = num_character_ids;
				_num_character_types = num_character_types;

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
			int FeatureExtractor::function_id_label_u(int y_i){
				int index = y_i;
				assert(index < _w_size_label_u);
				return index + _offset_w_label_u;
			}
			int FeatureExtractor::function_id_label_b(int y_i_1, int y_i){
				int index =  y_i_1 * 2 + y_i;
				assert(index < _w_size_label_b);
				return index + _offset_w_label_b;
			}
			int FeatureExtractor::function_id_unigram_u(int y_i, int i, int x_i){
				assert(x_i < _num_character_ids);
				assert(1 <= i && i <= _x_range_unigram);
				int index = x_i * _x_range_unigram * 2 + (i - 1) * 2 + y_i;
				assert(index < _w_size_unigram_u);
				return index + _offset_w_unigram_u;
			}
			int FeatureExtractor::function_id_unigram_b(int y_i_1, int y_i, int i, int x_i){
				assert(x_i < _num_character_ids);
				assert(1 <= i && i <= _x_range_unigram);
				int index = x_i * _x_range_unigram * 2 * 2 + (i - 1) * 2 * 2 + y_i * 2 + y_i_1;
				assert(index < _w_size_unigram_b);
				return index + _offset_w_unigram_b;
			}
			int FeatureExtractor::function_id_bigram_u(int y_i, int i, int x_i_1, int x_i){
				assert(x_i_1 < _num_character_ids);
				assert(x_i < _num_character_ids);
				assert(1 <= i && i <= _x_range_bigram);
				int index = x_i * _num_character_ids * _x_range_bigram * 2 + x_i_1 * _x_range_bigram * 2 + (i - 1) * 2 + y_i;
				assert(index < _w_size_bigram_u);
				return index + _offset_w_bigram_u;
			}
			int FeatureExtractor::function_id_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i){
				assert(x_i_1 < _num_character_ids);
				assert(x_i < _num_character_ids);
				assert(1 <= i && i <= _x_range_bigram);
				int index = x_i * _num_character_ids * _x_range_bigram * 2 * 2 + x_i_1 * _x_range_bigram * 2 * 2 + (i - 1) * 2 * 2 + y_i * 2 + y_i_1;
				assert(index < _w_size_bigram_b);
				return index + _offset_w_bigram_b;
			}
			int FeatureExtractor::function_id_identical_1_u(int y_i, int i){
				assert(1 <= i && i <= _x_range_identical_1);
				int index = (i - 1) * 2 + y_i;
				assert(index < _w_size_identical_1_u);
				return index + _offset_w_identical_1_u;
			}
			int FeatureExtractor::function_id_identical_1_b(int y_i_1, int y_i, int i){
				assert(1 <= i && i <= _x_range_identical_1);
				int index = (i - 1) * 2 * 2 + y_i * 2 + y_i_1;
				assert(index < _w_size_identical_1_b);
				return index + _offset_w_identical_1_b;
			}
			int FeatureExtractor::function_id_identical_2_u(int y_i, int i){
				assert(1 <= i && i <= _x_range_identical_2);
				int index = (i - 1) * 2 + y_i;
				assert(index < _w_size_identical_2_u);
				return index + _offset_w_identical_2_u;
			}
			int FeatureExtractor::function_id_identical_2_b(int y_i_1, int y_i, int i){
				assert(1 <= i && i <= _x_range_identical_2);
				int index = (i - 1) * 2 * 2 + y_i * 2 + y_i_1;
				assert(index < _w_size_identical_2_b);
				return index + _offset_w_identical_2_b;
			}
			int FeatureExtractor::function_id_unigram_type_u(int y_i, int type_i){
				int index = type_i * 2 + y_i;
				assert(index < _w_size_unigram_type_u);
				return index + _offset_w_unigram_type_u;
			}
			int FeatureExtractor::function_id_unigram_type_b(int y_i_1, int y_i, int type_i){
				assert(type_i < _num_character_types);
				int index = type_i * 2 * 2 + y_i * 2 + y_i_1;
				assert(index < _w_size_unigram_type_b);
				return index + _offset_w_unigram_type_b;
			}
			int FeatureExtractor::function_id_bigram_type_u(int y_i, int type_i_1, int type_i){
				assert(type_i_1 < _num_character_types);
				assert(type_i < _num_character_types);
				int index = type_i * _num_character_types * 2 + type_i_1 * 2 + y_i;
				assert(index < _w_size_bigram_type_u);
				return index + _offset_w_bigram_type_u;
			}
			int FeatureExtractor::function_id_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i){
				assert(type_i_1 < _num_character_types);
				assert(type_i < _num_character_types);
				int index = type_i * _num_character_types * 2 * 2 + type_i_1 * 2 * 2 + y_i * 2 + y_i_1;
				assert(index < _w_size_bigram_type_b);
				return index + _offset_w_bigram_type_b;
			}
			int FeatureExtractor::function_id_to_feature_id(int function_id, bool generate_feature_id_if_needed){
				auto itr = _function_id_to_feature_id.find(function_id);
				if(itr == _function_id_to_feature_id.end()){
					if(generate_feature_id_if_needed == false){
						return -1;
					}
					int feature_id = _function_id_to_feature_id.size();
					_function_id_to_feature_id[function_id] = feature_id;
					return feature_id;
				}
				return itr->second;
			}
			int FeatureExtractor::feature_id_label_u(int y_i){
				int function_id = function_id_label_u(y_i);
				return function_id_to_feature_id(function_id, false);
			}
			int FeatureExtractor::feature_id_label_b(int y_i_1, int y_i){
				int function_id = function_id_label_b(y_i_1, y_i);
				return function_id_to_feature_id(function_id, false);
			}
			int FeatureExtractor::feature_id_unigram_u(int y_i, int i, int x_i){
				int function_id = function_id_unigram_u(y_i, i, x_i);
				return function_id_to_feature_id(function_id, false);
			}
			int FeatureExtractor::feature_id_unigram_b(int y_i_1, int y_i, int i, int x_i){
				int function_id = function_id_unigram_b(y_i_1, y_i, i, x_i);
				return function_id_to_feature_id(function_id, false);
			}
			int FeatureExtractor::feature_id_bigram_u(int y_i, int i, int x_i_1, int x_i){
				int function_id = function_id_bigram_u(y_i, i, x_i_1, x_i);
				return function_id_to_feature_id(function_id, false);
			}
			int FeatureExtractor::feature_id_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i){
				int function_id = function_id_bigram_b(y_i_1, y_i, i, x_i_1, x_i);
				return function_id_to_feature_id(function_id, false);
			}
			int FeatureExtractor::feature_id_identical_1_u(int y_i, int i){
				int function_id = function_id_identical_1_u(y_i, i);
				return function_id_to_feature_id(function_id, false);
			}
			int FeatureExtractor::feature_id_identical_1_b(int y_i_1, int y_i, int i){
				int function_id = function_id_identical_1_b(y_i_1, y_i, i);
				return function_id_to_feature_id(function_id, false);
			}
			int FeatureExtractor::feature_id_identical_2_u(int y_i, int i){
				int function_id = function_id_identical_2_u(y_i, i);
				return function_id_to_feature_id(function_id, false);
			}
			int FeatureExtractor::feature_id_identical_2_b(int y_i_1, int y_i, int i){
				int function_id = function_id_identical_2_b(y_i_1, y_i, i);
				return function_id_to_feature_id(function_id, false);
			}
			int FeatureExtractor::feature_id_unigram_type_u(int y_i, int type_i){
				int function_id = function_id_unigram_type_u(y_i, type_i);
				return function_id_to_feature_id(function_id, false);
			}
			int FeatureExtractor::feature_id_unigram_type_b(int y_i_1, int y_i, int type_i){
				int function_id = function_id_unigram_type_b(y_i_1, y_i, type_i);
				return function_id_to_feature_id(function_id, false);
			}
			int FeatureExtractor::feature_id_bigram_type_u(int y_i, int type_i_1, int type_i){
				int function_id = function_id_bigram_type_u(y_i, type_i_1, type_i);
				return function_id_to_feature_id(function_id, false);
			}
			int FeatureExtractor::feature_id_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i){
				int function_id = function_id_bigram_type_b(y_i_1, y_i, type_i_1, type_i);
				return function_id_to_feature_id(function_id, false);
			}
			FeatureIndices* FeatureExtractor::extract(Sentence* sentence, bool generate_feature_id_if_needed){
				assert(sentence->_features == NULL);

				array<int> &character_ids = sentence->_character_ids;
				wchar_t const* characters = sentence->_characters;
				int character_ids_length = sentence->size();
				int feature_id;

				// CRFのラベルunigram素性の数
				mat::bi<int> num_crf_features_u(character_ids_length + 3, 2);
				int*** crf_feature_indices_u = new int**[character_ids_length + 3];

				// ラベルunigram素性
				for(int i = 1;i <= character_ids_length + 2;i++){	// 末尾に<eos>が2つ入る
					crf_feature_indices_u[i] = new int*[2];
					for(int y_i = 0;y_i <= 1;y_i++){
						std::vector<int> indices_u;
						int r_start, r_end;
						// ラベル素性
						feature_id = function_id_to_feature_id(function_id_label_u(y_i), generate_feature_id_if_needed);
						if(feature_id != -1){
							indices_u.push_back(feature_id);
						}
						// 文字unigram素性
						r_start = std::max(1, i + _x_unigram_start);
						r_end = std::min(character_ids_length + 2, i + _x_unigram_end);	// <eos>2つを考慮
						for(int r = r_start;r <= r_end;r++){
							int pos = r - i - _x_unigram_start + 1;	// [1, _x_range_unigram]
							int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
							feature_id = function_id_to_feature_id(function_id_unigram_u(y_i, pos, x_i), generate_feature_id_if_needed);
							if(feature_id != -1){
								indices_u.push_back(feature_id);
							}
						}
						// 文字bigram素性
						r_start = std::max(2, i + _x_bigram_start);
						r_end = std::min(character_ids_length + 2, i + _x_bigram_end);
						for(int r = r_start;r <= r_end;r++){
							int pos = r - i - _x_bigram_start + 1;	// [1, _x_range_bigram]
							int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
							int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : SPECIAL_CHARACTER_END;
							feature_id = function_id_to_feature_id(function_id_bigram_u(y_i, pos, x_i_1, x_i), generate_feature_id_if_needed);
							if(feature_id != -1){
								indices_u.push_back(feature_id);
							}
						}
						// identical_1素性
						r_start = std::max(2, i + _x_identical_1_start);
						r_end = std::min(character_ids_length + 2, i + _x_identical_1_end);
						for(int r = r_start;r <= r_end;r++){
							int pos = r - i - _x_identical_1_start + 1;	// [1, _x_range_identical_1]
							int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
							int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : SPECIAL_CHARACTER_END;
							if(x_i == x_i_1){
								feature_id = function_id_to_feature_id(function_id_identical_1_u(y_i, pos), generate_feature_id_if_needed);
								if(feature_id != -1){
									indices_u.push_back(feature_id);
								}
							}
						}
						// identical_2素性
						r_start = std::max(3, i + _x_identical_2_start);
						r_end = std::min(character_ids_length + 2, i + _x_identical_2_end);
						for(int r = r_start;r <= r_end;r++){
							int pos = r - i - _x_identical_2_start + 1;	// [1, _x_range_identical_2]
							int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
							int x_i_2 = (r - 2 <= character_ids_length) ? character_ids[r - 3] : SPECIAL_CHARACTER_END;
							if(x_i == x_i_2){
								feature_id = function_id_to_feature_id(function_id_identical_2_u(y_i, pos), generate_feature_id_if_needed);
								if(feature_id != -1){
									indices_u.push_back(feature_id);
								}
							}
						}
						// 文字種unigram・bigram素性
						int type_i = (i <= character_ids_length) ? ctype::get_type(characters[i - 1]) : CTYPE_UNKNOWN;
						int type_i_1 = (i - 1 <= character_ids_length) ? ctype::get_type(characters[i - 2]) : CTYPE_UNKNOWN;
						feature_id = function_id_to_feature_id(function_id_unigram_type_u(y_i, type_i), generate_feature_id_if_needed);
						if(feature_id != -1){
							indices_u.push_back(feature_id);
						}
						feature_id = function_id_to_feature_id(function_id_bigram_type_u(y_i, type_i_1, type_i), generate_feature_id_if_needed);
						if(feature_id != -1){
							indices_u.push_back(feature_id);
						}

						int num_features = indices_u.size();
						num_crf_features_u(i, y_i) = num_features;
						if(num_features > 0){
							crf_feature_indices_u[i][y_i] = new int[num_features];
							for(int n = 0;n < num_features;n++){
								crf_feature_indices_u[i][y_i][n] = indices_u[n];
							}
						}
					}
				}

				// CRFのラベルbigram素性の数
				mat::tri<int> num_crf_features_b(character_ids_length + 3, 2, 2);
				int**** crf_feature_indices_b = new int***[character_ids_length + 3];

				// ラベルbigram素性
				for(int i = 1;i <= character_ids_length + 2;i++){	// 末尾に<eos>が2つ入る
					crf_feature_indices_b[i] = new int**[2];
					for(int y_i_1 = 0;y_i_1 <= 1;y_i_1++){
						crf_feature_indices_b[i][y_i_1] = new int*[2];
						for(int y_i = 0;y_i <= 1;y_i++){
							std::vector<int> indices_b;
							int r_start, r_end;
							// ラベル素性
							feature_id = function_id_to_feature_id(function_id_label_b(y_i_1, y_i), generate_feature_id_if_needed);
							if(feature_id != -1){
								indices_b.push_back(feature_id);
							}
							// 文字unigram素性
							r_start = std::max(1, i + _x_unigram_start);
							r_end = std::min(character_ids_length + 2, i + _x_unigram_end);	// <eos>2つを考慮
							for(int r = r_start;r <= r_end;r++){
								int pos = r - i - _x_unigram_start + 1;	// [1, _x_range_unigram]
								int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
								feature_id = function_id_to_feature_id(function_id_unigram_b(y_i_1, y_i, pos, x_i), generate_feature_id_if_needed);
								if(feature_id != -1){
									indices_b.push_back(feature_id);
								}
							}
							// 文字bigram素性
							r_start = std::max(2, i + _x_bigram_start);
							r_end = std::min(character_ids_length + 2, i + _x_bigram_end);
							for(int r = r_start;r <= r_end;r++){
								int pos = r - i - _x_bigram_start + 1;	// [1, _x_range_bigram]
								int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
								int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : SPECIAL_CHARACTER_END;
								feature_id = function_id_to_feature_id(function_id_bigram_b(y_i_1, y_i, pos, x_i_1, x_i), generate_feature_id_if_needed);
								if(feature_id != -1){
									indices_b.push_back(feature_id);
								}
							}
							// identical_1素性
							r_start = std::max(2, i + _x_identical_1_start);
							r_end = std::min(character_ids_length + 2, i + _x_identical_1_end);
							for(int r = r_start;r <= r_end;r++){
								int pos = r - i - _x_identical_1_start + 1;	// [1, _x_range_identical_1]
								int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
								int x_i_1 = (r - 1 <= character_ids_length) ? character_ids[r - 2] : SPECIAL_CHARACTER_END;
								if(x_i == x_i_1){
									feature_id = function_id_to_feature_id(function_id_identical_1_b(y_i_1, y_i, pos), generate_feature_id_if_needed);
									if(feature_id != -1){
										indices_b.push_back(feature_id);
									}
								}
							}
							// identical_2素性
							r_start = std::max(3, i + _x_identical_2_start);
							r_end = std::min(character_ids_length + 2, i + _x_identical_2_end);
							for(int r = r_start;r <= r_end;r++){
								int pos = r - i - _x_identical_2_start + 1;	// [1, _x_range_identical_2]
								int x_i = (r <= character_ids_length) ? character_ids[r - 1] : SPECIAL_CHARACTER_END;
								int x_i_2 = (r - 2 <= character_ids_length) ? character_ids[r - 3] : SPECIAL_CHARACTER_END;
								if(x_i == x_i_2){
									feature_id = function_id_to_feature_id(function_id_identical_2_b(y_i_1, y_i, pos), generate_feature_id_if_needed);
									if(feature_id != -1){
										indices_b.push_back(feature_id);
									}
								}
							}
							// 文字種unigram・bigram素性
							int type_i = (i <= character_ids_length) ? ctype::get_type(characters[i - 1]) : CTYPE_UNKNOWN;
							int type_i_1 = (i - 1 <= character_ids_length) ? ctype::get_type(characters[i - 2]) : CTYPE_UNKNOWN;
							feature_id = function_id_to_feature_id(function_id_unigram_type_b(y_i_1, y_i, type_i), generate_feature_id_if_needed);
							if(feature_id != -1){
								indices_b.push_back(feature_id);
							}
							feature_id = function_id_to_feature_id(function_id_bigram_type_b(y_i_1, y_i, type_i_1, type_i), generate_feature_id_if_needed);
							if(feature_id != -1){
								indices_b.push_back(feature_id);
							}

							// 更新
							int num_features = indices_b.size();
							num_crf_features_b(i, y_i_1, y_i) = num_features;
							if(num_features > 0){
								crf_feature_indices_b[i][y_i_1][y_i] = new int[num_features];
								for(int n = 0;n < num_features;n++){
									crf_feature_indices_b[i][y_i_1][y_i][n] = indices_b[n];
								}
							}
						}
					}
				}
				FeatureIndices* features = new FeatureIndices();
				features->_num_features_u = num_crf_features_u;
				features->_num_features_b = num_crf_features_b;
				features->_feature_indices_u = crf_feature_indices_u;
				features->_feature_indices_b = crf_feature_indices_b;
				features->_seq_length = character_ids_length + 3;
				return features;
			}
			template <class Archive>
			void FeatureExtractor::serialize(Archive &ar, unsigned int version)
			{
				ar & _num_character_ids;
				ar & _num_character_types;
				ar & _x_unigram_start;
				ar & _x_unigram_end;
				ar & _x_bigram_start;
				ar & _x_bigram_end;
				ar & _x_identical_1_start;
				ar & _x_identical_1_end;
				ar & _x_identical_2_start;
				ar & _x_identical_2_end;
				ar & _x_range_unigram;
				ar & _x_range_bigram;
				ar & _x_range_identical_1;
				ar & _x_range_identical_2;
				ar & _w_size_label_u;
				ar & _w_size_label_b;
				ar & _w_size_unigram_u;
				ar & _w_size_unigram_b;
				ar & _w_size_bigram_u;
				ar & _w_size_bigram_b;
				ar & _w_size_identical_1_u;
				ar & _w_size_identical_1_b;
				ar & _w_size_identical_2_u;
				ar & _w_size_identical_2_b;
				ar & _w_size_unigram_type_u;
				ar & _w_size_unigram_type_b;
				ar & _w_size_bigram_type_u;
				ar & _w_size_bigram_type_b;
				ar & _offset_w_label_u;
				ar & _offset_w_label_b;
				ar & _offset_w_unigram_u;
				ar & _offset_w_unigram_b;
				ar & _offset_w_bigram_u;
				ar & _offset_w_bigram_b;
				ar & _offset_w_identical_1_u;
				ar & _offset_w_identical_1_b;
				ar & _offset_w_identical_2_u;
				ar & _offset_w_identical_2_b;
				ar & _offset_w_unigram_type_u;
				ar & _offset_w_unigram_type_b;
				ar & _offset_w_bigram_type_u;
				ar & _offset_w_bigram_type_b;
			}
			template void FeatureExtractor::serialize(boost::archive::binary_iarchive &ar, unsigned int version);
			template void FeatureExtractor::serialize(boost::archive::binary_oarchive &ar, unsigned int version);
		}
	}
}
#include "features.h"

namespace npycrf {
	namespace crf {
		FeatureIndices* FeatureIndices::copy(){
			FeatureIndices* copy = new FeatureIndices();
			int size = _seq_length;
			copy->_num_features_u = new int*[size];
			copy->_feature_indices_u = new int**[size];
			copy->_num_features_b = new int**[size];
			copy->_feature_indices_b = new int***[size];

			for(int i = 2;i < size;i++){
				copy->_num_features_u[i] = new int[2];
				copy->_feature_indices_u[i] = new int*[2];
				for(int y_i = 0;y_i <= 1;y_i++){
					copy->_num_features_u[i][y_i] = _num_features_u[i][y_i];
					copy->_feature_indices_u[i][y_i] = new int[_num_features_u[i][y_i]];
					for(int n = 0;n < _num_features_u[i][y_i];n++){
						copy->_feature_indices_u[i][y_i][n] = _feature_indices_u[i][y_i][n];
					}
				}
			}

			for(int i = 2;i < size;i++){
				copy->_num_features_b[i] = new int*[2];
				copy->_num_features_b[i][0] = new int[2];
				copy->_num_features_b[i][1] = new int[2];
				copy->_feature_indices_b[i] = new int**[2];
				for(int y_i_1 = 0;y_i_1 <= 1;y_i_1++){
					copy->_feature_indices_b[i][y_i_1] = new int*[2];
					for(int y_i = 0;y_i <= 1;y_i++){
						copy->_num_features_b[i][y_i_1][y_i] = _num_features_b[i][y_i_1][y_i];
						copy->_feature_indices_b[i][y_i_1][y_i] = new int[_num_features_b[i][y_i_1][y_i]];
						for(int n = 0;n < _num_features_b[i][y_i_1][y_i];n++){
							copy->_feature_indices_b[i][y_i_1][y_i][n] = _feature_indices_b[i][y_i_1][y_i][n];
						}
					}
				}
			}

			return copy;
		}
	}
}
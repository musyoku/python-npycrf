#include "crf.h"

namespace npycrf {
	namespace crf {
		CRF::CRF(int feature_x_unigram_start = -2,
				 int feature_x_unigram_end = 2,
				 int feature_x_bigram_start = -2,
				 int feature_x_bigram_end = 1,
				 int feature_x_identical_1_start = -2,
				 int feature_x_identical_1_end = 1,
				 int feature_x_identical_2_start = -3,
				 int feature_x_identical_2_end = 1,
				 int num_character_ids)
		{
			bias = 0;
			int x_unigram_size = feature_x_unigram_end - feature_x_unigram_start + 1;
			int x_bigram_size = feature_x_bigram_end - feature_x_bigram_end + 1;
			int x_identical_1_size = feature_x_identical_1_end - feature_x_identical_1_start + 1;
			int x_identical_2_size = feature_x_identical_2_end - feature_x_identical_2_start + 1;

			// NPYCRFではy_iは0か1の2通り
			_w_unigram_u = new double**[2];
			_w_unigram_b = new double***[2];
			_w_bigram_u = new double***[2];
			_w_bigram_b = new double****[2];
			_w_identical_1_u = new double*[2];
			_w_identical_1_b = new double**[2];
			_w_identical_2_u = new double*[2];
			_w_identical_2_b = new double**[2];
			_w_unigram_type_u = new double*[2];
			_w_unigram_type_b = new double**[2];
			_w_bigram_type_u = new double**[2];
			_w_bigram_type_b = new double***[2];
			// label unigram
			for(int y_i = 0;y_i <= 1;y_i++){
				// (y_i, i, x_i)
				_w_unigram_u[y_i] = new double*[x_unigram_size];
				for(int i = 0;i < x_unigram_size;i++){
					_w_unigram_u[y_i][i] = new double[num_character_ids];
					for(int x_i = 0;x_i < num_character_ids;x_i++){
						_w_unigram_u[y_i][i][x_i] = 0;
					}
				}
				// (y_i, i, x_{i-1}, x_i)
				_w_bigram_u[y_i] = new double**[x_bigram_size];
				for(int i = 0;i < x_bigram_size;i++){
					_w_bigram_u[y_i][i] = new double*[num_character_ids];
					for(int x_i_1 = 0;x_i_1 < num_character_ids;x_i_1++){
						_w_unigram_u[y_i][i][x_i_1] = new double[num_character_ids];
						for(int x_i = 0;x_i < num_character_ids;x_i++){
							_w_unigram_u[y_i][i][x_i_1][x_i] = 0;
						}
					}
				}
				// (y_i, i)
				_w_identical_1_u[y_i] = new double[x_identical_1_size];
				for(int i = 0;i < x_identical_1_size;i++){
					_w_identical_1_u[y_i][i] = 0;
				}
				// (y_i, i)
				_w_identical_2_u[y_i] = new double[x_identical_2_size];
				for(int i = 0;i < x_identical_2_size;i++){
					_w_identical_2_u[y_i][i] = 0;
				}
			}
			// label bigram
			for(int y_i_1 = 0;y_i_1 <= 1;y_i_1++){
				_w_unigram_b[y_i_1] = new double**[2]
				_w_bigram_b[y_i_1] = new double**[2]
				_w_identical_1_b[y_i_1] = new double*[2]
				_w_identical_2_b[y_i_1] = new double[2]
				_w_unigram_type_b[y_i_1] = new double*[2]
				_w_bigram_type_b[y_i_1] = new double**[2]
				for(int y_i = 0;y_i <= 1;y_i++){
					// (y_{i-1}, y_i, i, x_i)
					_w_unigram_b[y_i_1][y_i] = new double*[x_unigram_size];
					for(int i = 0;i < x_unigram_size;i++){
						_w_unigram_u[y_i][i] = new double[num_character_ids];
						for(int x_i = 0;x_i < num_character_ids;x_i++){
							_w_unigram_u[y_i][i][x_i] = 0;
						}
					}
				}
			}
		}
	}
}
#pragma once
#include "../common.h"

// [1] A discriminative latent variable chinese segmenter with hybrid word/character information
//     https://pdfs.semanticscholar.org/0bd3/a662e19467aa21c1e1e0a51db397e9936b70.pdf

namespace npycrf {
	namespace crf {
		class CRF {
		private:
			int _x_range_unigram;
			int _x_range_bigram;
			int _x_range_identical_1;
			int _x_range_identical_2;
			int _num_character_ids;
			int _num_character_types;
			double _index_w_unigram_u(int y_i, int i, int x_i);
			double _index_w_unigram_b(int y_i_1, int y_i, int i, int x_i);
			double _index_w_bigram_u(int y_i, int i, int x_i_1, int x_i);
			double _index_w_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i);
			double _index_w_identical_1_u(int y_i, int i);
			double _index_w_identical_1_b(int y_i_1, int y_i, int i);
			double _index_w_identical_2_u(int y_i, int i);
			double _index_w_identical_2_b(int y_i_1, int y_i, int i);
			double _index_w_unigram_type_u(int y_i, int type_i);
			double _index_w_unigram_type_b(int y_i_1, int y_i, int type_i);
			double _index_w_bigram_type_u(int y_i, int type_i_1, int type_i);
			double _index_w_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i);
		public:
			// y ∈ {0,1}
			// x ∈ Z
			// i ∈ Z
			// type ∈ Z
			int _w_size_unigram_u;		// (y_i, i, x_i)
			int _w_size_unigram_b;		// (y_{i-1}, y_i, i, x_i)
			int _w_size_bigram_u;		// (y_i, i, x_{i-1}, x_i)
			int _w_size_bigram_b;		// (y_{i-1}, y_i, i, x_{i-1}, x_i)
			int _w_size_identical_1_u;	// (y_i, i)
			int _w_size_identical_1_b;	// (y_{i-1}, y_i, i)
			int _w_size_identical_2_u;	// (y_i, i)
			int _w_size_identical_2_b;	// (y_{i-1}, y_i, i)
			int _w_size_unigram_type_u;	// (y_i, type)
			int _w_size_unigram_type_b;	// (y_{i-1}, y_i, type)
			int _w_size_bigram_type_u;	// (y_i, type, type)
			int _w_size_bigram_type_b;	// (y_{i-1}, y_i, type, type)
			double _bias;
			double* _w_unigram_u;		// (y_i, i, x_i)
			double* _w_unigram_b;		// (y_{i-1}, y_i, i, x_i)
			double* _w_bigram_u;		// (y_i, i, x_{i-1}, x_i)
			double* _w_bigram_b;		// (y_{i-1}, y_i, i, x_{i-1}, x_i)
			double* _w_identical_1_u;	// (y_i, i)
			double* _w_identical_1_b;	// (y_{i-1}, y_i, i)
			double* _w_identical_2_u;	// (y_i, i)
			double* _w_identical_2_b;	// (y_{i-1}, y_i, i)
			double* _w_unigram_type_u;	// (y_i, type)
			double* _w_unigram_type_b;	// (y_{i-1}, y_i, type)
			double* _w_bigram_type_u;	// (y_i, type, type)
			double* _w_bigram_type_b;	// (y_{i-1}, y_i, type, type)
			// 用いる素性は以下の4通り（デフォルト値の例）[1]
			// i-2, i-1, i, i+1, i+2の位置のunigram文字
			// i-2, i-1, i, i+1の位置のbigram文字
			// i-2, i-1, i, i+1において、x_i == x_{i+1}
			// i-3, i-2, i-1, i, i+1において、x_i == x_{i+2}
			// これらの素性を[y_i]と[y_{i-1}, y_i]に関連づける
			CRF(int num_character_ids,		// 文字IDの総数
				int num_character_types,	// 文字種の総数
				int feature_x_unigram_start = -2,
				int feature_x_unigram_end = 2,
				int feature_x_bigram_start = -2,
				int feature_x_bigram_end = 1,
				int feature_x_identical_1_start = -2,
				int feature_x_identical_1_end = 1,
				int feature_x_identical_2_start = -3,
				int feature_x_identical_2_end = 1
			);
			~CRF();
			double bias();
			double w_unigram_u(int y_i, int i, int x_i);
			double w_unigram_b(int y_i_1, int y_i, int i, int x_i);
			double w_bigram_u(int y_i, int i, int x_i_1, int x_i);
			double w_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i);
			double w_identical_1_u(int y_i, int i);
			double w_identical_1_b(int y_i_1, int y_i, int i);
			double w_identical_2_u(int y_i, int i);
			double w_identical_2_b(int y_i_1, int y_i, int i);
			double w_unigram_type_u(int y_i, int type_i);
			double w_unigram_type_b(int y_i_1, int y_i, int type_i);
			double w_bigram_type_u(int y_i, int type_i_1, int type_i);
			double w_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i);
			void set_w_unigram_u(int y_i, int i, int x_i, double value);
			void set_w_unigram_b(int y_i_1, int y_i, int i, int x_i, double value);
			void set_w_bigram_u(int y_i, int i, int x_i_1, int x_i, double value);
			void set_w_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i, double value);
			void set_w_identical_1_u(int y_i, int i, double value);
			void set_w_identical_1_b(int y_i_1, int y_i, int i, double value);
			void set_w_identical_2_u(int y_i, int i, double value);
			void set_w_identical_2_b(int y_i_1, int y_i, int i, double value);
			void set_w_unigram_type_u(int y_i, int type_i, double value);
			void set_w_unigram_type_b(int y_i_1, int y_i, int type_i, double value);
			void set_w_bigram_type_u(int y_i, int type_i_1, int type_i, double value);
			void set_w_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i, double value);
		};
	}
}
#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "../../array.h"
#include "../../sentence.h"

namespace npycrf {
	namespace crf {
		namespace feature {
			class FeatureExtractor {
			private:
				friend class boost::serialization::access;
				template <class Archive>
				void serialize(Archive &archive, unsigned int version);
			public:
				hashmap<int, int> _function_id_to_feature_id;	// 素性関数の番号から素性IDへの変換
				int _num_character_ids;
				int _num_character_types;
				// y ∈ {0,1}
				// x ∈ Z
				// i ∈ Z
				// type ∈ Z
				int _x_range_unigram;
				int _x_range_bigram;
				int _x_range_identical_1;
				int _x_range_identical_2;
				int _x_unigram_start;
				int _x_unigram_end;
				int _x_bigram_start;
				int _x_bigram_end;
				int _x_identical_1_start;
				int _x_identical_1_end;
				int _x_identical_2_start;
				int _x_identical_2_end;
				int _w_size_label_u;		// (y_i)
				int _w_size_label_b;		// (y_{i-1}, y_i)
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
				int _weight_size;
				int _offset_w_label_u;
				int _offset_w_label_b;
				int _offset_w_unigram_u;
				int _offset_w_unigram_b;
				int _offset_w_bigram_u;
				int _offset_w_bigram_b;
				int _offset_w_identical_1_u;
				int _offset_w_identical_1_b;
				int _offset_w_identical_2_u;
				int _offset_w_identical_2_b;
				int _offset_w_unigram_type_u;
				int _offset_w_unigram_type_b;
				int _offset_w_bigram_type_u;
				int _offset_w_bigram_type_b;
				FeatureExtractor();
				FeatureExtractor(int num_character_ids,		// 文字IDの総数
								int num_character_types,	// 文字種の総数
								int feature_x_unigram_start,
								int feature_x_unigram_end,
								int feature_x_bigram_start,
								int feature_x_bigram_end,
								int feature_x_identical_1_start,
								int feature_x_identical_1_end,
								int feature_x_identical_2_start,
								int feature_x_identical_2_end);
				// 以下、iは左端を1とした番号
				// 例）
				// i=1,   2,   3, 4,   5
				//   t-2, t-1, t, t+1, t+2
				// ラベルyと入力xの組み合わせから一意な素性関数IDを生成
				// 素性IDではない
				int function_id_label_u(int y_i);
				int function_id_label_b(int y_i_1, int y_i);
				int function_id_unigram_u(int y_i, int i, int x_i);
				int function_id_unigram_b(int y_i_1, int y_i, int i, int x_i);
				int function_id_bigram_u(int y_i, int i, int x_i_1, int x_i);
				int function_id_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i);
				int function_id_identical_1_u(int y_i, int i);
				int function_id_identical_1_b(int y_i_1, int y_i, int i);
				int function_id_identical_2_u(int y_i, int i);
				int function_id_identical_2_b(int y_i_1, int y_i, int i);
				int function_id_unigram_type_u(int y_i, int type_i);
				int function_id_unigram_type_b(int y_i_1, int y_i, int type_i);
				int function_id_bigram_type_u(int y_i, int type_i_1, int type_i);
				int function_id_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i);
				int function_id_to_feature_id(int function_id, bool generate_feature_id_if_needed);
				// 素性関数IDから素性IDを返す
				// 存在しない場合は-1を返す
				int feature_id_label_u(int y_i);
				int feature_id_label_b(int y_i_1, int y_i);
				int feature_id_unigram_u(int y_i, int i, int x_i);
				int feature_id_unigram_b(int y_i_1, int y_i, int i, int x_i);
				int feature_id_bigram_u(int y_i, int i, int x_i_1, int x_i);
				int feature_id_bigram_b(int y_i_1, int y_i, int i, int x_i_1, int x_i);
				int feature_id_identical_1_u(int y_i, int i);
				int feature_id_identical_1_b(int y_i_1, int y_i, int i);
				int feature_id_identical_2_u(int y_i, int i);
				int feature_id_identical_2_b(int y_i_1, int y_i, int i);
				int feature_id_unigram_type_u(int y_i, int type_i);
				int feature_id_unigram_type_b(int y_i_1, int y_i, int type_i);
				int feature_id_bigram_type_u(int y_i, int type_i_1, int type_i);
				int feature_id_bigram_type_b(int y_i_1, int y_i, int type_i_1, int type_i);
				FeatureIndices* extract(Sentence* sentence, bool generate_feature_id_if_needed);
			};
		}
	}
}
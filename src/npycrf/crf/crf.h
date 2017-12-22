#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "../common.h"
#include "../array.h"
#include "../sentence.h"
#include "parameter.h"
#include "feature/indices.h"
#include "feature/extractor.h"

// [1] A discriminative latent variable chinese segmenter with hybrid word/character information
//     https://pdfs.semanticscholar.org/0bd3/a662e19467aa21c1e1e0a51db397e9936b70.pdf

namespace npycrf {
	namespace crf {
		using namespace feature;
		class CRF {
		private:
			friend class boost::serialization::access;
			template <class Archive>
			void serialize(Archive &archive, unsigned int version);
			void save(boost::archive::binary_oarchive &ar, unsigned int version) const;
			void load(boost::archive::binary_iarchive &ar, unsigned int version);
		public:
			Parameter* _parameter;
			FeatureExtractor* _extractor;
			// xに関する素性は以下の4通り（デフォルト値の例）[1]
			// i-2, i-1, i, i+1, i+2の位置のunigram文字
			// i-2, i-1, i, i+1の位置のbigram文字
			// i-2, i-1, i, i+1において、x_i == x_{i+1}
			// i-3, i-2, i-1, i, i+1において、x_i == x_{i+2}
			// これらの素性を[y_i]と[y_{i-1}, y_i]に関連づける
			CRF(FeatureExtractor* extractor, Parameter* parameter);
			CRF();
			~CRF();
			int get_num_features();
			double w_label_u(int y_i);
			double w_label_b(int y_i_1, int y_i);
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
			void set_w_label_u(int y_i, double value);
			void set_w_label_b(int y_i_1, int y_i, double value);
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
			double compute_gamma(Sentence* sentence, int s, int t);
			double compute_path_cost(Sentence* sentence, int i_1, int i, int y_i_1, int y_i);
			double _compute_cost_label_features(int y_i_1, int y_i);
			double _compute_cost_unigram_features(array<int> &character_ids, int character_ids_length, int i, int y_i_1, int y_i);
			double _compute_cost_bigram_features(array<int> &character_ids, int character_ids_length, int i, int y_i_1, int y_i);
			double _compute_cost_identical_1_features(array<int> &character_ids, int character_ids_length, int i, int y_i_1, int y_i);
			double _compute_cost_identical_2_features(array<int> &character_ids, int character_ids_length, int i, int y_i_1, int y_i);
			double _compute_cost_unigram_and_bigram_type_features(array<int> &character_ids, wchar_t const* characters, int character_ids_length, int i, int y_i_1, int y_i);
			double compute_log_p_y_given_sentence(Sentence* sentence);
			FeatureIndices* extract_features(Sentence* sentence, bool generate_feature_id_if_needed);
		};
	}
}
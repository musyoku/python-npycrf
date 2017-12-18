#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <vector> 
#include "../common.h"
#include "../array.h"
#include "lm/node.h"
#include "lm/vpylm.h"
#include "lm/hpylm.h"

namespace npycrf {
	namespace npylm {
		// character_idsのsubstr_char_t_startからsubstr_char_t_endまでの文字列に<eow>を付けてtoken_idsの先頭に格納
		void append_eow(npycrf::array<int> &character_ids, int substr_char_t_start, int substr_char_t_end, npycrf::array<int> &token_ids);
		double factorial(double n);
		class NPYLM {
		private:
			void _allocate_capacity(int max_sentence_length);
			void _delete_capacity();
			friend class boost::serialization::access;
			template <class Archive>
			void serialize(Archive &archive, unsigned int version);
			void save(boost::archive::binary_oarchive &archive, unsigned int version) const;
			void load(boost::archive::binary_iarchive &archive, unsigned int version);
		public:
			lm::HPYLM* _hpylm;	// 単語n-gram
			lm::VPYLM* _vpylm;	// 文字n-gram

			// 単語unigramノードで新たなテーブルが作られた時はVPYLMからその単語が生成されたと判断し、単語の文字列をVPYLMに追加する
			// その時各文字がVPYLMのどの深さに追加されたかを保存する
			// 単語unigramノードのテーブルごと、単語IDごとに保存する必要がある
			hashmap<id, std::vector<std::vector<int>>> _prev_depth_at_table_of_token;

			hashmap<id, double> _g0_cache;
			hashmap<id, double> _vpylm_g0_cache;
			npycrf::array<double> _lambda_for_type;
			npycrf::array<double> _pk_vpylm;	// 文字n-gramから長さkの単語が生成される確率
			npycrf::array<int> _token_ids;
			int _max_word_length;
			int _max_sentence_length;
			double _lambda_a;
			double _lambda_b;
			// 計算高速化用
			npycrf::array<double> _hpylm_parent_pw_cache;
			bool _fix_g0_using_poisson; // 単語の事前分布をポアソン分布により補正するかどうか
			NPYLM(){
				_fix_g0_using_poisson = true;
			}
			NPYLM(int max_word_length, 
				int max_sentence_length, 
				double g0, 
				double initial_lambda_a, 
				double initial_lambda_b, 
				double vpylm_beta_stop, 
				double vpylm_beta_pass);
			~NPYLM();
			void reserve(int max_sentence_length);
			void set_vpylm_g0(double g0);
			void set_lambda_prior(double a, double b);
			void sample_lambda_with_initial_params();
			bool add_customer_at_time_t(Sentence* sentence, int t);
			void vpylm_add_customers(array<int> &character_ids, int token_ids_length, std::vector<int> &prev_depths);
			bool remove_customer_at_time_t(Sentence* sentence, int t);
			void vpylm_remove_customers(array<int> &character_ids, int token_ids_length, std::vector<int> &prev_depths);
			lm::Node<id>* find_node_by_tracing_back_context_from_time_t(Sentence* sentence, int word_t_index, npycrf::array<double> &parent_pw_cache, bool generate_node_if_needed, bool return_middle_node);
			lm::Node<id>* find_node_by_tracing_back_context_from_time_t(npycrf::array<id> &word_ids, int word_ids_length, int word_t_index, bool generate_node_if_needed, bool return_middle_node);
			lm::Node<id>* find_node_by_tracing_back_context_from_time_t(
				array<int> &character_ids, wchar_t const* characters, int character_ids_length, 
				npycrf::array<id> &word_ids, int word_ids_length, 
				int word_t_index, int substr_char_t_start, int substr_char_t_end, 
				npycrf::array<double> &parent_pw_cache, bool generate_node_if_needed, bool return_middle_node);
			// word_idは既知なので再計算を防ぐ
			double compute_g0_substring_at_time_t(array<int> &character_ids, wchar_t const* characters, int character_ids_length, int substr_char_t_start, int substr_char_t_end, id word_t_id);
			double compute_poisson_k_lambda(unsigned int k, double lambda);
			double compute_p_k_given_vpylm(int k);
			void sample_hpylm_vpylm_hyperparameters();
			double compute_log_p_y_given_sentence(Sentence* sentence);
			double compute_p_y_given_sentence(Sentence* sentence);
			double compute_p_w_given_h(Sentence* sentence, int word_t_index);
			double compute_p_w_given_h(
				array<int> &character_ids, wchar_t const* characters, int character_ids_length, 
				npycrf::array<id> &word_ids, int word_ids_length, int word_t_index);
			double compute_p_w_given_h(
				array<int> &character_ids, wchar_t const* characters, int character_ids_length, 
				npycrf::array<id> &word_ids, int word_ids_length, 
				int word_t_index, int substr_char_t_start, int substr_char_t_end);
		};
	}
}
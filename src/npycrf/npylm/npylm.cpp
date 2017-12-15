#include <boost/serialization/split_member.hpp>
#include <algorithm>
#include <iostream> 
#include <cmath> 
#include <cassert> 
#include <cfloat>
#include "../hash.h"
#include "../sampler.h"
#include "../wordtype.h"
#include "npylm.h"

namespace npycrf {
	namespace npylm {
		using namespace lm;
		// character_idsのsubstr_char_t_startからsubstr_char_t_endまでの文字列を<bow>と<eow>で挟んでwrapped_character_idsの先頭に格納
		void append_eow(array<int> &character_ids, int substr_t_start_index, int substr_t_end_index, array<int> &token_ids){
			int i = 0;
			for(;i < substr_t_end_index - substr_t_start_index + 1;i++){
				token_ids[i] = character_ids[i + substr_t_start_index];
			}
			token_ids[i] = SPECIAL_CHARACTER_END;
		}
		double factorial(double n) {
			if (n == 0){
				return 1;
			}
			return n * factorial(n - 1);
		}
		// lambda_a, lambda_bは単語長のポアソン分布のハイパーパラメータ
		// 異なる文字種ごとに違うλを使うが、学習時に個別に推定するため事前分布は共通化する
		NPYLM::NPYLM(int max_word_length, int max_sentence_length, double g0, double initial_lambda_a, double initial_lambda_b, double vpylm_beta_stop, double vpylm_beta_pass){
			_hpylm = new HPYLM(3);		// 3-gram以外を指定すると動かないので注意
			_vpylm = new VPYLM(g0, max_sentence_length, vpylm_beta_stop, vpylm_beta_pass);
			_lambda_for_type = array<double>(WORDTYPE_NUM_TYPES + 1);	// 文字種ごとの単語長のポアソン分布のハイパーパラメータ
			_hpylm_parent_pw_cache = array<double>(3);		// 3-gram
			set_lambda_prior(initial_lambda_a, initial_lambda_b);

			_max_sentence_length = max_sentence_length;
			_max_word_length = max_word_length;
			_token_ids = array<int>(max_sentence_length + 1); 	// <eow>を含める
			_pk_vpylm = array<double>(max_word_length + 2);			// kが1スタート、かつk > max_word_length用の領域も必要なので+2
			for(int k = 1;k < max_word_length + 2;k++){
				_pk_vpylm[k] = 1.0 / (max_word_length + 2);
			}
			_fix_g0_using_poisson = true;
			#ifdef __DEBUG__
			std::cout << "Warning: Debug mode enabled!" << std::endl;
			#endif
		}
		NPYLM::~NPYLM(){
			delete _hpylm;
			delete _vpylm;
		}
		void NPYLM::reserve(int max_sentence_length){
			if(max_sentence_length <= _max_sentence_length){
				return;
			}
			_delete_capacity();
			_allocate_capacity(max_sentence_length);
			_max_sentence_length = max_sentence_length;
		}
		void NPYLM::_allocate_capacity(int max_sentence_length){
			_max_sentence_length = max_sentence_length;
			_token_ids = array<int>(max_sentence_length + 1);
		}
		void NPYLM::_delete_capacity(){

		}
		void NPYLM::set_vpylm_g0(double g0){
			_vpylm->set_g0(g0);
		}
		void NPYLM::set_lambda_prior(double a, double b){
			_lambda_a = a;
			_lambda_b = b;
			sample_lambda_with_initial_params();
		}
		void NPYLM::sample_lambda_with_initial_params(){
			for(int type = 1;type <= WORDTYPE_NUM_TYPES;type++){
				_lambda_for_type[type] = sampler::gamma(_lambda_a, _lambda_b);
			}
		}
		bool NPYLM::add_customer_at_time_t(Sentence* sentence, int t){
			assert(t >= 2);
			id token_t = sentence->get_word_id_at(t);
			Node<id>* node = find_node_by_tracing_back_context_from_time_t(sentence, t, _hpylm_parent_pw_cache, true, false);
			assert(node != NULL);
			int num_tables_before = _hpylm->_root->_num_tables;
			int added_table_k = -1;
			int substr_t_start_index = sentence->_start[t];
			int substr_t_end_index = sentence->_start[t] + sentence->_segments[t] - 1;
			node->add_customer(token_t, _hpylm_parent_pw_cache, _hpylm->_d_m, _hpylm->_theta_m, true, added_table_k);
			int num_tables_after = _hpylm->_root->_num_tables;
			// 単語unigramノードでテーブル数が増えた場合VPYLMに追加
			if(num_tables_before < num_tables_after){
				_g0_cache.clear();
				if(token_t == SPECIAL_CHARACTER_END){
					_vpylm->_root->add_customer(token_t, _vpylm->_g0, _vpylm->_d_m, _vpylm->_theta_m, true, added_table_k);
					return true;
				}
				assert(added_table_k != -1);
				std::vector<std::vector<int>> &depths = _prev_depth_at_table_of_token[token_t];
				assert(depths.size() <= added_table_k);	// 存在してはいけない
				std::vector<int> prev_depths;
				assert(substr_t_end_index >= substr_t_start_index);
				assert(substr_t_end_index < _max_sentence_length);
				int token_ids_length = substr_t_end_index - substr_t_start_index + 2;	// <eow>を考慮
				append_eow(sentence->_character_ids, substr_t_start_index, substr_t_end_index, _token_ids);
				vpylm_add_customers(_token_ids, token_ids_length, prev_depths);
				depths.push_back(prev_depths);
			}
			return true;
		}
		void NPYLM::vpylm_add_customers(array<int> &token_ids, int token_ids_length, std::vector<int> &prev_depths){
			assert(prev_depths.size() == 0);
			// 客を追加
			for(int t = 0;t < token_ids_length;t++){
				int depth_t = _vpylm->sample_depth_at_time_t(token_ids, t, _vpylm->_parent_pw_cache, _vpylm->_path_nodes);
				_vpylm->add_customer_at_time_t(token_ids, t, depth_t, _vpylm->_parent_pw_cache, _vpylm->_path_nodes);	// キャッシュを使って追加
				prev_depths.push_back(depth_t);
			}
			assert(prev_depths.size() == token_ids_length);
		}
		bool NPYLM::remove_customer_at_time_t(Sentence* sentence, int t){
			assert(t >= 2);
			id word_t = sentence->get_word_id_at(t);
			Node<id>* node = find_node_by_tracing_back_context_from_time_t(sentence->_word_ids, sentence->get_num_segments(), t, false, false);
			assert(node != NULL);
			int num_tables_before = _hpylm->_root->_num_tables;
			int removed_from_table_k = -1;
			int substr_t_start_index = sentence->_start[t];
			int substr_t_end_index = sentence->_start[t] + sentence->_segments[t] - 1;
			node->remove_customer(word_t, true, removed_from_table_k);

			// 単語unigramノードでテーブル数が増えた場合VPYLMから削除
			int num_tables_after = _hpylm->_root->_num_tables;
			if(num_tables_before > num_tables_after){
				_g0_cache.clear();
				if(word_t == SPECIAL_CHARACTER_END){
					// <eos>は文字列に分解できないので常にVPYLMのルートノードに追加されている
					_vpylm->_root->remove_customer(word_t, true, removed_from_table_k);
					return true;
				}
				assert(removed_from_table_k != -1);
				auto itr = _prev_depth_at_table_of_token.find(word_t);
				assert(itr != _prev_depth_at_table_of_token.end());
				std::vector<std::vector<int>> &depths = itr->second;
				assert(removed_from_table_k < depths.size());
				// 客を除外
				std::vector<int> &prev_depths = depths[removed_from_table_k];
				assert(prev_depths.size() > 0);
				assert(substr_t_end_index >= substr_t_start_index);
				assert(substr_t_end_index < _max_sentence_length);
				int token_ids_length = substr_t_end_index - substr_t_start_index + 2;	// <eow>を考慮
				append_eow(sentence->_character_ids, substr_t_start_index, substr_t_end_index, _token_ids);
				vpylm_remove_customers(_token_ids, token_ids_length, prev_depths);
				// シフト
				depths.erase(depths.begin() + removed_from_table_k);
			}
			if(node->need_to_remove_from_parent()){
				node->remove_from_parent();
			}
			return true;
		}
		void NPYLM::vpylm_remove_customers(array<int> &token_ids, int token_ids_length, std::vector<int> &prev_depths){
			// 客を除外
			assert(prev_depths.size() == token_ids_length);
			auto prev_depth_t = prev_depths.begin();
			for(int t = 0;t < token_ids_length;t++){
				_vpylm->remove_customer_at_time_t(token_ids, t, *prev_depth_t);
				prev_depth_t++;
			}
		}
		Node<id>* NPYLM::find_node_by_tracing_back_context_from_time_t(
				id const* word_ids, int word_ids_length, int word_t_index, 
				bool generate_node_if_needed, bool return_middle_node)
		{
			assert(word_t_index >= 2);
			assert(word_t_index < word_ids_length);
			Node<id>* node = _hpylm->_root;
			for(int depth = 1;depth <= 2;depth++){
				id context_id = SPECIAL_CHARACTER_BEGIN;
				if(word_t_index - depth >= 0){
					context_id = word_ids[word_t_index - depth];
				}
				Node<id>* child = node->find_child_node(context_id, generate_node_if_needed);
				if(child == NULL){
					if(return_middle_node){
						return node;
					}
					return NULL;
				}
				node = child;
			}
			assert(node->_depth == 2);
			return node;
		}
		// add_customer用
		Node<id>* NPYLM::find_node_by_tracing_back_context_from_time_t(
				Sentence* sentence, int word_t_index, array<double> &parent_pw_cache, 
				bool generate_node_if_needed, bool return_middle_node)
		{
			assert(word_t_index >= 2);
			assert(word_t_index < sentence->get_num_segments());
			assert(sentence->_segments[word_t_index] > 0);
			int substr_t_start_index = sentence->_start[word_t_index];
			int substr_t_end_index = sentence->_start[word_t_index] + sentence->_segments[word_t_index] - 1;
			return find_node_by_tracing_back_context_from_time_t(
				sentence->_character_ids, sentence->_characters, sentence->size(), 
				sentence->_word_ids, sentence->get_num_segments(), 
				word_t_index, substr_t_start_index, substr_t_end_index, 
				parent_pw_cache, generate_node_if_needed, return_middle_node);
		}
		// 効率のためノードを探しながら確率も計算する
		Node<id>* NPYLM::find_node_by_tracing_back_context_from_time_t(
				array<int> &character_ids, wchar_t const* characters, int character_ids_length, 
				id const* word_ids, int word_ids_length, 
				int word_t_index, int substr_t_start_index, int substr_t_end_index, 
				array<double> &parent_pw_cache, bool generate_node_if_needed, bool return_middle_node)
		{
			assert(word_t_index >= 2);
			assert(word_t_index < word_ids_length);
			Node<id>* node = _hpylm->_root;
			id word_t_id = word_ids[word_t_index];
			int word_length = substr_t_end_index - substr_t_start_index + 1;
			double parent_pw = 0;
			if(word_t_id == SPECIAL_CHARACTER_END){
				parent_pw = _vpylm->_g0;
			}else{
				assert(0 <= substr_t_start_index && substr_t_start_index <= substr_t_end_index);
				assert(substr_t_start_index <= substr_t_end_index);
				if(word_length > _max_word_length){
					parent_pw = 0;
				}else{
					parent_pw = compute_g0_substring_at_time_t(character_ids, characters, character_ids_length, substr_t_start_index, substr_t_end_index, word_t_id);
				}
			}
			if(word_length > _max_word_length){
				assert(parent_pw >= 0);
			}else{
				assert(parent_pw > 0);
			}
			parent_pw_cache[0] = parent_pw;
			for(int depth = 1;depth <= 2;depth++){
				id context_id = SPECIAL_CHARACTER_BEGIN;
				if(word_t_index - depth >= 0){
					context_id = word_ids[word_t_index - depth];
				}
				// 事前に確率を計算
				double pw = node->compute_p_w_with_parent_p_w(word_t_id, parent_pw, _hpylm->_d_m, _hpylm->_theta_m);
				parent_pw_cache[depth] = pw;
				Node<id>* child = node->find_child_node(context_id, generate_node_if_needed);
				if(child == NULL && return_middle_node == true){
					return node;
				}
				assert(child != NULL);
				parent_pw = pw;
				node = child;
			}
			assert(node->_depth == 2);
			return node;
		}
		// word_idは既知なので再計算を防ぐ
		double NPYLM::compute_g0_substring_at_time_t(
				array<int> &character_ids, wchar_t const* characters, int character_ids_length, 
				int substr_t_start_index, int substr_t_end_index, id word_t_id)
		{
			if(word_t_id == SPECIAL_CHARACTER_END){
				return _vpylm->_g0;
			}

			#ifdef __DEBUG__
				id a = hash_substring_ptr(characters, substr_t_start_index, substr_t_end_index);
				assert(a == word_t_id);
			#endif

			assert(substr_t_end_index < _max_sentence_length);
			assert(substr_t_start_index >= 0);
			assert(substr_t_end_index >= substr_t_start_index);
			int word_length = substr_t_end_index - substr_t_start_index + 1;
			assert(word_length <= character_ids_length);
			assert(word_length <= _max_word_length);
			auto itr = _g0_cache.find(word_t_id);
			if(itr == _g0_cache.end()){
				// 先頭に<bow>をつける
				// wchar_t* token_ids = _characters;
				// append_eow(characters, substr_t_start_index, substr_t_end_index, token_ids);
				// int token_ids_length = substr_t_end_index - substr_t_start_index + 3;
				// g0を計算
				double pw = std::max(_vpylm->compute_p_w(character_ids, substr_t_start_index, substr_t_end_index), std::numeric_limits<double>::min());

				// 学習の最初のイテレーションでは文が丸ごと1単語になるので補正する意味はない
				if(_fix_g0_using_poisson == false){
					_g0_cache[word_t_id] = pw;
					return pw;
				}

				double p_k_given_vpylm = compute_p_k_given_vpylm(word_length);
				int type = wordtype::detect_word_type_substr(characters, substr_t_start_index, substr_t_end_index);
				assert(type <= WORDTYPE_NUM_TYPES);
				assert(type > 0);
				double lambda = _lambda_for_type[type];
				double poisson = compute_poisson_k_lambda(word_length, lambda);
				assert(poisson > 0);
				double g0 = pw * poisson / p_k_given_vpylm;

				// ごく稀にポアソン補正で1を超えることがある
				if((0 < g0 && g0 < 1) == false){
					for(int u = 0;u < character_ids_length;u++){
						std::wcout << characters[u];
					}
					std::wcout << std::endl;
					for(int u = 0;u < character_ids_length;u++){
						std::cout << character_ids[u] << ", ";
					}
					std::cout << std::endl;
					std::cout << "pw = " << pw << std::endl;
					std::cout << "poisson = " << poisson << std::endl;
					std::cout << "p_k_given_vpylm = " << p_k_given_vpylm << std::endl;
					std::cout << "g0 = " << g0 << std::endl;
					std::cout << "word_length = " << word_length << std::endl;
				}
				assert(0 < g0 && g0 < 1);
				_g0_cache[word_t_id] = g0;
				return g0;
			}
			return itr->second;
		}
		double NPYLM::compute_poisson_k_lambda(unsigned int k, double lambda){
			return pow(lambda, k) * exp(-lambda) / factorial(k);
		}
		double NPYLM::compute_p_k_given_vpylm(int k){
			assert(0 < k && k <= _max_word_length);
			return _pk_vpylm[k];
		}
		void NPYLM::sample_hpylm_vpylm_hyperparameters(){
			_hpylm->sample_hyperparams();
			_vpylm->sample_hyperparams();
		}
		double NPYLM::compute_log_p_y_given_sentence(Sentence* sentence){
			double pw = 0;
			for(int t = 2;t < sentence->get_num_segments();t++){
				pw += log(compute_p_w_given_h(sentence, t));
			}
			return pw;
		}
		double NPYLM::compute_p_y_given_sentence(Sentence* sentence){
			double pw = 1;
			for(int t = 2;t < sentence->get_num_segments();t++){
				pw *= compute_p_w_given_h(sentence, t);
			}
			return pw;
		}
		double NPYLM::compute_p_w_given_h(Sentence* sentence, int word_t_index){
			assert(word_t_index >= 2);
			assert(word_t_index < sentence->get_num_segments());
			assert(sentence->_segments[word_t_index] > 0);
			int substr_t_start_index = sentence->_start[word_t_index];
			int substr_t_end_index = sentence->_start[word_t_index] + sentence->_segments[word_t_index] - 1;
			return compute_p_w_given_h(sentence->_character_ids, sentence->_characters, sentence->size(), sentence->_word_ids, sentence->get_num_segments(), word_t_index, substr_t_start_index, substr_t_end_index);
		}
		// word_t_index, substr_t_start_index, substr_char_t_endはインデックス
		double NPYLM::compute_p_w_given_h(
				array<int> &character_ids, wchar_t const* characters, int character_ids_length, 
				id const* word_ids, int word_ids_length, int word_t_index)
		{
			return compute_p_w_given_h(character_ids, characters, character_ids_length, word_ids, word_ids_length, word_t_index, -1, -1);
		}
		double NPYLM::compute_p_w_given_h(
				array<int> &character_ids, wchar_t const* characters, int character_ids_length, 
				id const* word_ids, int word_ids_length, 
				int word_t_index, int substr_t_start_index, int substr_t_end_index)
		{
			assert(0 <= word_t_index && word_t_index < word_ids_length);
			id word_id = word_ids[word_t_index];

			if(word_id != SPECIAL_CHARACTER_END){
				assert(0 <= substr_t_start_index && substr_t_start_index <= substr_t_end_index);
				assert(0 <= substr_t_end_index && substr_t_end_index < character_ids_length);
				#ifdef __DEBUG__
					id a = hash_substring_ptr(characters, substr_t_start_index, substr_t_end_index);
					if(a != word_id){
						for(int i = substr_t_start_index;i <= substr_t_end_index;i++){
							std::wcout << characters[i];
						}
						std::cout << std::endl;
						std::cout << "character_ids_length: " << character_ids_length << std::endl;
						std::cout << "substr_t_start_index: " << substr_t_start_index << std::endl;
						std::cout << "substr_t_end_index: " << substr_t_end_index << std::endl;
						std::cout << word_ids[0] << ", " << word_ids[1] << ", " << word_ids[2] << ", " << std::endl;
						std::cout << word_id << std::endl;
						std::cout << a << std::endl;
					}
					assert(a == word_id);
				#endif
			}
			// ノードを探しながら_hpylm_parent_pw_cacheをセット
			Node<id>* node = find_node_by_tracing_back_context_from_time_t(character_ids, characters, character_ids_length, word_ids, word_ids_length, word_t_index, substr_t_start_index, substr_t_end_index, _hpylm_parent_pw_cache, false, true);
			assert(node != NULL);
			double parent_pw = _hpylm_parent_pw_cache[node->_depth];
			// 効率のため親の確率のキャッシュから計算
			return node->compute_p_w_with_parent_p_w(word_id, parent_pw, _hpylm->_d_m, _hpylm->_theta_m);
		}
		template <class Archive>
		void NPYLM::serialize(Archive &archive, unsigned int version)
		{
			boost::serialization::split_member(archive, *this, version);
		}
		template void NPYLM::serialize(boost::archive::binary_iarchive &ar, unsigned int version);
		template void NPYLM::serialize(boost::archive::binary_oarchive &ar, unsigned int version);
		void NPYLM::save(boost::archive::binary_oarchive &archive, unsigned int version) const {
			archive & _hpylm;
			archive & _vpylm;
			archive & _max_word_length;
			archive & _max_sentence_length;
			archive & _lambda_a;
			archive & _lambda_b;

			for(int type = 1;type <= WORDTYPE_NUM_TYPES;type++){
				archive & _lambda_for_type[type];
			}
			for(int k = 0;k <= _max_word_length + 1;k++){
				archive & _pk_vpylm[k];
			}
		}
		void NPYLM::load(boost::archive::binary_iarchive &archive, unsigned int version) {
			archive & _hpylm;
			archive & _vpylm;
			archive & _max_word_length;
			archive & _max_sentence_length;
			archive & _lambda_a;
			archive & _lambda_b;

			_pk_vpylm = npycrf::array<double>(_max_word_length + 2);
			_lambda_for_type = array<double>(WORDTYPE_NUM_TYPES + 1);

			_hpylm_parent_pw_cache = array<double>(3);
			_token_ids = array<int>(_max_sentence_length + 1);

			for(int type = 1;type <= WORDTYPE_NUM_TYPES;type++){
				archive & _lambda_for_type[type];
			}
			for(int k = 0;k <= _max_word_length + 1;k++){
				archive & _pk_vpylm[k];
			}
		}
	}
}
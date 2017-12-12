#include <boost/serialization/split_member.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <iostream>
#include <cassert>
#include <fstream>
#include "../../sampler.h"
#include "vpylm.h"

namespace npycrf {
	namespace npylm {
		namespace lm {
			VPYLM::VPYLM(double g0, int max_possible_depth, double beta_stop, double beta_pass){
				assert(g0 > 0);
				_root = new Node<int>(0);
				_root->_depth = 0;	// ルートは深さ0
				// http://www.ism.ac.jp/~daichi/paper/ipsj07vpylm.pdfによると初期値は(4, 1)
				// しかしVPYLMは初期値にあまり依存しないらしい
				_beta_stop = beta_stop;
				_beta_pass = beta_pass;
				_depth = 0;
				_g0 = g0;
				_max_depth = max_possible_depth;	// 訓練データ中の最大長の文の文字数が可能な最大深さになる
				_parent_pw_cache = new double[max_possible_depth + 1];
				_sampling_table = new double[max_possible_depth + 1];
				_path_nodes = new Node<int>*[max_possible_depth + 1];
			}
			VPYLM::~VPYLM(){
				_delete_node(_root);
				delete[] _sampling_table;
				delete[] _parent_pw_cache;
				delete[] _path_nodes;
			}
			bool VPYLM::add_customer_at_time_t(array<int> &character_ids, int t, int depth_t){
				assert(_parent_pw_cache != NULL);
				assert(0 <= depth_t && depth_t <= t);
				Node<int>* node = find_node_by_tracing_back_context(character_ids, t, depth_t, _parent_pw_cache);
				assert(node != NULL);
				if(depth_t > 0){	// ルートノードは特殊なので無視
					assert(node->_token_id == character_ids[t - depth_t]);
				}
				assert(node->_depth == depth_t);
				int token_t = character_ids[t];
				int tabke_k;
				return node->add_customer(token_t, _parent_pw_cache, _d_m, _theta_m, true, tabke_k);
			}
			// parent_pw_cacheがすでにセットされていてpath_nodesを更新する
			// NPYLMから呼ぶ用
			bool VPYLM::add_customer_at_time_t(array<int> &character_ids, int t, int depth_t, double* parent_pw_cache, Node<int>** path_nodes){
				assert(path_nodes != NULL);
				assert(0 <= depth_t && depth_t <= t);
				Node<int>* node = find_node_by_tracing_back_context(character_ids, t, depth_t, path_nodes);
				assert(node != NULL);
				if(depth_t > 0){	// ルートノードは特殊なので無視
					// if(node->_token_id != character_ids[t - depth_t]){
					// 	for(int i = 0;i <= t;i++){
					// 		std::wcout << character_ids[i];
					// 	}
					// 	std::wcout << std::endl;
					// }
					assert(node->_token_id == character_ids[t - depth_t]);
				}
				assert(node->_depth == depth_t);
				int token_t = character_ids[t];
				int tabke_k;
				return node->add_customer(token_t, parent_pw_cache, _d_m, _theta_m, true, tabke_k);
			}
			bool VPYLM::remove_customer_at_time_t(array<int> &character_ids, int t, int depth_t){
				assert(0 <= depth_t && depth_t <= t);
				Node<int>* node = find_node_by_tracing_back_context(character_ids, t, depth_t, false, false);
				assert(node != NULL);
				if(depth_t > 0){
					assert(node->_token_id == character_ids[t - depth_t]);
				}
				assert(node->_depth == depth_t);
				int token_t = character_ids[t];
				int table_k;
				node->remove_customer(token_t, true, table_k);
				// 客が一人もいなくなったらノードを削除する
				if(node->need_to_remove_from_parent()){
					node->remove_from_parent();
				}
				return true;
			}
			// 文字列の位置tからorderだけ遡る
			// character_ids:       [2, 4, 7, 1, 9, 10]
			// t: 3                     ^     ^
			// depth_t: 2               |<- <-|
			Node<int>* VPYLM::find_node_by_tracing_back_context(array<int> &character_ids, int t, int depth_t, bool generate_node_if_needed, bool return_middle_node){
				if(t - depth_t < 0){
					return NULL;
				}
				Node<int>* node = _root;
				for(int depth = 1;depth <= depth_t;depth++){
					int context_token_id = character_ids[t - depth];
					Node<int>* child = node->find_child_node(context_token_id, generate_node_if_needed);
					if(child == NULL){
						if(return_middle_node){
							return node;
						}
						return NULL;
					}
					node = child;
				}
				assert(node->_depth == depth_t);
				if(depth_t > 0){
					assert(node->_token_id == character_ids[t - depth_t]);
				}
				return node;
			}
			// add_customer用
			// 辿りながら確率をキャッシュ
			Node<int>* VPYLM::find_node_by_tracing_back_context(array<int> &character_ids, int t, int depth_t, double* parent_pw_cache){
				assert(parent_pw_cache != NULL);
				if(t - depth_t < 0){
					return NULL;
				}
				int token_t = character_ids[t];
				Node<int>* node = _root;
				double parent_pw = _g0;
				parent_pw_cache[0] = _g0;
				for(int depth = 1;depth <= depth_t;depth++){
					int context_token_id = character_ids[t - depth];
					// 事前に確率を計算
					double pw = node->compute_p_w_with_parent_p_w(token_t, parent_pw, _d_m, _theta_m);
					assert(pw > 0);
					parent_pw_cache[depth] = pw;
					Node<int>* child = node->find_child_node(context_token_id, true);
					assert(child != NULL);
					parent_pw = pw;
					node = child;
				}
				assert(node->_depth == depth_t);
				if(depth_t > 0){
					assert(node->_token_id == character_ids[t - depth_t]);
				}
				return node;
			}
			// すでに辿ったノードのキャッシュを使いながら辿る
			Node<int>* VPYLM::find_node_by_tracing_back_context(array<int> &character_ids, int t, int depth_t, Node<int>** path_nodes_cache){
				assert(path_nodes_cache != NULL);
				if(t - depth_t < 0){
					return NULL;
				}
				Node<int>* node = _root;
				int depth = 0;
				for(;depth < depth_t;depth++){
					if(path_nodes_cache[depth + 1] != NULL){
						node = path_nodes_cache[depth + 1];
						assert(node->_depth == depth + 1);
					}else{
						int context_token_id = character_ids[t - depth - 1];
						Node<int>* child = node->find_child_node(context_token_id, true);
						assert(child != NULL);
						node = child;
					}
				}
				assert(node != NULL);
				if(depth_t > 0){
					assert(node->_token_id == character_ids[t - depth_t]);
				}
				return node;
			}
			double VPYLM::compute_p_w(array<int> &character_ids, int substr_start, int substr_end){
				int token_t = character_ids[substr_start];
				double pw = _root->compute_p_w(token_t, _g0, _d_m, _theta_m);
				for(int t = substr_start;t < substr_end;t++){
					pw *= compute_p_w_given_h(character_ids, substr_start, t);
				}
				return pw;
			}
			double VPYLM::compute_log_p_w(array<int> &character_ids, int substr_start, int substr_end){
				int token_t = character_ids[substr_start];
				double log_pw = log(_root->compute_p_w(token_t, _g0, _d_m, _theta_m));
				for(int t = substr_start;t < substr_end;t++){
					log_pw += log(compute_p_w_given_h(character_ids, substr_start, t));
				}
				return log_pw;
			}
			// 文字列のcontext_substr_startからcontext_substr_endまでの部分文字列を文脈として、context_substr_end+1の文字が生成される確率
			double VPYLM::compute_p_w_given_h(array<int> &character_ids, int context_substr_start, int context_substr_end){
				assert(context_substr_start >= 0);
				assert(context_substr_end >= context_substr_start);
				int target_id = character_ids[context_substr_end + 1];
				return compute_p_w_given_h(target_id, character_ids, context_substr_start, context_substr_end);
			}
			// 単語のサンプリングなどで任意のtarget_idの確率を計算することがあるため一般化
			// 文字列のcontext_substr_startからcontext_substr_endまでの部分文字列を文脈として、target_idが生成される確率
			double VPYLM::compute_p_w_given_h(int target_id, array<int> &character_ids, int context_substr_start, int context_substr_end){
				assert(context_substr_start >= 0);
				assert(context_substr_end >= context_substr_start);
				Node<int>* node = _root;
				assert(node != NULL);
				double parent_pass_probability = 1;
				double p = 0;
				double parent_pw = _g0;
				double eps = VPYLM_EPS;		// 停止確率がこの値を下回れば打ち切り
				double p_stop = 1;
				int depth = 0;

				// 無限の深さまで考える
				// 実際のコンテキスト長を超えて確率を計算することもある
				while(p_stop > eps){
					// ノードがない場合親の確率とベータ事前分布から計算
					if(node == NULL){
						p_stop = (_beta_stop) / (_beta_pass + _beta_stop) * parent_pass_probability;
						p += parent_pw * p_stop;
						parent_pass_probability *= (_beta_pass) / (_beta_pass + _beta_stop);
					}else{
						assert(context_substr_end - depth >= 0);
						assert(node->_depth == depth);
						double pw = node->compute_p_w_with_parent_p_w(target_id, parent_pw, _d_m, _theta_m);
						p_stop = node->stop_probability(_beta_stop, _beta_pass, false) * parent_pass_probability;
						p += pw * p_stop;
						parent_pass_probability *= node->pass_probability(_beta_stop, _beta_pass, false);
						parent_pw = pw;
						if(context_substr_end - depth <= context_substr_start){
							node = NULL;
						}else{
							int context_token_id = character_ids[context_substr_end - depth];
							Node<int>* child = node->find_child_node(context_token_id);
							node = child;
							if(depth > 0 && node){
								assert(node->_token_id == context_token_id);
							}
						}
					}
					depth++;
				}
				assert(p > 0);
				return p;
			}
			// 辿ったノードとそれぞれのノードからの出力確率をキャッシュしながらオーダーをサンプリング
			int VPYLM::sample_depth_at_time_t(array<int> &character_ids, int t, double* parent_pw_cache, Node<int>** path_nodes){
				assert(path_nodes != NULL);
				assert(parent_pw_cache != NULL);
				if(t == 0){
					return 0;
				}
				// VPYLMは本来無限の深さを考えるが、計算量的な問題から以下の値を下回れば打ち切り
				double eps = VPYLM_EPS;
				
				int token_t = character_ids[t];
				double sum = 0;
				double parent_pw = _g0;
				double parent_pass_probability = 1;
				parent_pw_cache[0] = _g0;
				int sampling_table_size = 0;
				Node<int>* node = _root;
				for(int n = 0;n <= t;n++){
					if(node){
						assert(n == node->_depth);
						double pw = node->compute_p_w_with_parent_p_w(token_t, parent_pw, _d_m, _theta_m);
						double p_stop = node->stop_probability(_beta_stop, _beta_pass, false);
						double p = pw * p_stop * parent_pass_probability;
						parent_pw = pw;
						parent_pw_cache[n + 1] = pw;
						_sampling_table[n] = p;
						path_nodes[n] = node;
						sampling_table_size += 1;
						parent_pass_probability *= node->pass_probability(_beta_stop, _beta_pass, false);
						sum += p;
						if(p_stop < eps){
							break;
						}
						if(n < t){
							assert(t - n - 1 >= 0);
							int context_token_id = character_ids[t - n - 1];
							node = node->find_child_node(context_token_id);
						}
					}else{
						double p_stop = (_beta_stop) / (_beta_pass + _beta_stop) * parent_pass_probability;
						double p = parent_pw * p_stop;	// ノードがない場合親の確率をそのまま使う
						parent_pw_cache[n + 1] = parent_pw;
						_sampling_table[n] = p;
						path_nodes[n] = NULL;
						sampling_table_size += 1;
						sum += p;
						parent_pass_probability *= (_beta_pass) / (_beta_pass + _beta_stop);
						if(p_stop < eps){
							break;
						}
					}
				}
				assert(sampling_table_size <= t + 1);
				double normalizer = 1.0 / sum;
				double bernoulli = sampler::uniform(0, 1);
				double stack = 0;
				for(int n = 0;n < sampling_table_size;n++){
					stack += _sampling_table[n] * normalizer;
					if(bernoulli < stack){
						return n;
					}
				}
				return _sampling_table[sampling_table_size - 1];
			}
			template <class Archive>
			void VPYLM::serialize(Archive &archive, unsigned int version)
			{
				boost::serialization::split_member(archive, *this, version);
			}
			template void VPYLM::serialize(boost::archive::binary_iarchive &ar, unsigned int version);
			template void VPYLM::serialize(boost::archive::binary_oarchive &ar, unsigned int version);
			void VPYLM::save(boost::archive::binary_oarchive &archive, unsigned int version) const {
				archive & _root;
				archive & _depth;
				archive & _max_depth;
				archive & _beta_stop;
				archive & _beta_pass;
				archive & _g0;
				archive & _d_m;
				archive & _theta_m;
				archive & _a_m;
				archive & _b_m;
				archive & _alpha_m;
				archive & _beta_m;
			}
			void VPYLM::load(boost::archive::binary_iarchive &archive, unsigned int version) {
				archive & _root;
				archive & _depth;
				archive & _max_depth;
				archive & _beta_stop;
				archive & _beta_pass;
				archive & _g0;
				archive & _d_m;
				archive & _theta_m;
				archive & _a_m;
				archive & _b_m;
				archive & _alpha_m;
				archive & _beta_m;
				_parent_pw_cache = new double[_max_depth + 1];
				_sampling_table = new double[_max_depth + 1];
				_path_nodes = new Node<int>*[_max_depth + 1];
			}
		};
	}
}
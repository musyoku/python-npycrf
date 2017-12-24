#pragma once
#include <vector>
#include "common.h"
#include "array.h"
#include "array.h"
#include "crf/crf.h"
#include "npylm/npylm.h"

namespace npycrf {
	class Lattice {
	private:
		bool _pure_crf_mode;	// NPYLMを無視
		bool _pure_npylm_mode;	// CRFを無視
		void _allocate_capacity(int max_word_length, int max_sentence_length);
		void _sum_alpha_t_k_j(Sentence* sentence, int t, int k, int j, mat::tri<double> &alpha, mat::quad<double> &pw_h_tkji, mat::quad<double> &p_transition_tkji, double prod_scaling, double crf_potential);
		void _sum_beta_t_k_j(Sentence* sentence, int t, int k, int j, mat::tri<double> &beta, mat::quad<double> &p_transition_tkji, npycrf::array<double> &scaling, bool use_scaling);
		void _backward_sampling(Sentence* sentence, mat::tri<double> &alpha, mat::quad<double> &p_transition_tkji, std::vector<int> &segments);
		void _sample_backward_k_and_j(Sentence* sentence, mat::tri<double> &alpha, mat::quad<double> &p_transition_tkji, int t, int next_word_length, int &sampled_k, int &sampled_j);
		double _lambda_0();
	public:
		npylm::NPYLM* _npylm;
		crf::CRF* _crf;
		array<id> _word_ids;			// 3-gram
		mat::bi<id> _substring_word_id_cache;
		mat::bi<double> _pc_s;			// 文の部分文字列が単語になる条件付き確率
		mat::tri<double> _alpha;		// 前向き確率
		mat::tri<double> _beta;			// 後向き確率
		mat::tri<double> _pz_s;			// Markov-CRFの周辺確率
		mat::tri<int> _viterbi_backward;
		mat::quad<double> _pw_h_tkji;	// n-gram確率のキャッシュ
		mat::quad<double> _p_transition_tkji;	// exp(lamda_0 * p(・) + potential)のキャッシュ
		mat::quad<double> _p_conc_tkji;
		array<double> _scaling;			// スケーリング係数
		array<double> _backward_sampling_table;
		int _max_word_length;
		int _max_sentence_length;
		Lattice(npylm::NPYLM* npylm, crf::CRF* crf);
		~Lattice();
		void set_pure_crf_mode(bool enabled);
		bool get_pure_crf_mode();
		void set_pure_npylm_mode(bool enabled);
		bool get_pure_npylm_mode();
		void set_npycrf_mode();
		id get_substring_word_id_at_t_k(Sentence* sentence, int t, int k);
		void reserve(int max_word_length, int max_sentence_length);
		void forward_filtering(Sentence* sentence, bool use_scaling);
		void backward_sampling(Sentence* sentence, std::vector<int> &segments);
		void blocked_gibbs(Sentence* sentence, std::vector<int> &segments, bool use_scaling = true);
		void viterbi_argmax_alpha_t_k_j(Sentence* sentence, int t, int k, int j);
		void viterbi_forward(Sentence* sentence);
		void viterbi_argmax_backward_k_and_j_to_eos(Sentence* sentence, int t, int &argmax_k, int &argmax_j);
		void viterbi_backward(Sentence* sentence, std::vector<int> &segments);
		void viterbi_decode(Sentence* sentence, std::vector<int> &segments);
		double compute_normalizing_constant(Sentence* sentence, bool use_scaling = true);
		double compute_log_normalizing_constant(Sentence* sentence, bool use_scaling = true);
		double _compute_normalizing_constant_backward(Sentence* sentence, mat::tri<double> &beta, mat::quad<double> &p_transition_tkji);
		void enumerate_marginal_p_trigram_given_sentence(Sentence* sentence, mat::quad<double> &p_conc_tkji, mat::quad<double> &pw_h_tkji, bool use_scaling = true);
		void _enumerate_marginal_p_trigram_given_sentence(Sentence* sentence, mat::quad<double> &p_conc_tkji, mat::tri<double> &alpha, mat::tri<double> &beta, mat::quad<double> &p_transition_tkji, npycrf::array<double> &scaling, bool use_scaling = true);
		void enumerate_marginal_p_z_given_sentence(Sentence* sentence, mat::tri<double> &pz_s);
		void _enumerate_marginal_p_z_given_sentence(Sentence* sentence, mat::tri<double> &pz_s, mat::tri<double> &alpha, mat::tri<double> &beta);
		void _enumerate_marginal_p_z_given_sentence_using_p_substring(mat::tri<double> &pz_s, int sentence_length, mat::bi<double> &pc_s);
		void enumerate_marginal_p_z_and_trigram_given_sentence(Sentence* sentence, mat::quad<double> &p_conc_tkji, mat::quad<double> &pw_h_tkji, mat::tri<double> &pz_s);
		double _compute_p_z_case_1_1(int sentence_length, int t, mat::bi<double> &pc_s);
		double _compute_p_z_case_1_0(int sentence_length, int t, mat::bi<double> &pc_s);
		double _compute_p_z_case_0_1(int sentence_length, int t, mat::bi<double> &pc_s);
		double _compute_p_z_case_0_0(int sentence_length, int t, mat::bi<double> &pc_s);
		void _enumerate_marginal_p_substring_given_sentence(mat::bi<double> &pc_s, int sentence_length, mat::tri<double> &alpha, mat::tri<double> &beta);
		void _enumerate_forward_variables(Sentence* sentence, mat::tri<double> &alpha, mat::quad<double> &pw_h_tkji, mat::quad<double> &p_transition_tkji, array<double> &scaling, bool use_scaling = true);
		void _enumerate_backward_variables(Sentence* sentence, mat::tri<double> &beta, mat::quad<double> &p_transition_tkji, array<double> &scaling, bool use_scaling = true);
		void _clear_p_tkji(int N);
		void _clear_word_id_cache(int N);
	};
} // namespace npylm
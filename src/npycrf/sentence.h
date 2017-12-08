#pragma once
#include <string>
#include <vector>
#include "crf/features.h"
#include "common.h"

// <bos>と<eos>は長さが0文字であることに注意
// NPYLMでは先頭に<bos>が2つ、末尾に<eos>が1つ
// CRFでは先頭に<bos>が1つ、末尾に<eos>が2つ

namespace npycrf {
	class Sentence {
	public:
		int _num_segments;	// <bos>2つと<eos>1つを含める
		int* _segments;		// 各単語の長さが入る. <bos>2つが先頭に来る
		int* _start;		// <bos>2つが先頭に来る
		wchar_t const* _characters; // _sentence_strの各文字. 実際には使わない
		int* _character_ids;// _sentence_strの各文字のid. 実際に使われるのはこっち
		id* _word_ids;		// <bos>2つと<eos>1つを含める
		int* _labels;		// CRFのラベル. <bos>が1つ先頭に入り、<eos>が末尾に2つ入る. CRFに合わせて1スタート、[0]は<bos>
		crf::FeatureIndices* _features;	// CRFの素性ID. 不変なのであらかじめ計算しておく.
		std::wstring _sentence_str;	// 生の文データ
		Sentence(std::wstring sentence, int* character_ids);
		~Sentence();
		Sentence* copy();
		int size();
		int get_num_segments();
		int get_num_segments_without_special_tokens();
		int get_word_length_at(int t);
		id get_word_id_at(int t);
		id get_substr_word_id(int start_index, int end_index);				// end_indexを含む
		int get_crf_label_at(int t);	// tは1から
		std::wstring get_substr_word_str(int start_index, int end_index);	// endを含む
		std::wstring get_word_str_at(int t);	// t=0,1の時は<bos>が返る
		void dump_characters();
		void dump_words();
		// num_segmentsには<bos>や<eos>の数は含めない
		void split(int* segments_without_special_tokens, int num_segments_without_special_tokens);
		void split(std::vector<int> &segments_without_special_tokens);
	};
} // namespace npycrf
#include <boost/python.hpp>
#include <iostream>
#include <fstream>
#include "dataset.h"
#include "../npycrf/sampler.h"
#include "../npycrf/array.h"

namespace npycrf {
	namespace python {
		Dataset::Dataset(Corpus* corpus, Dictionary* dict, double train_dev_split, int seed){
			sampler::set_seed(seed);
			_corpus = corpus;
			_max_sentence_length = 0;
			_avg_sentence_length = 0;
			int sum_sentence_length = 0;

			// データをtrain/devに振り分ける
			train_dev_split = std::min(1.0, std::max(0.0, train_dev_split));
			int num_data = corpus->get_num_data();
			std::vector<int> rand_indices;
			shuffle(rand_indices.begin(), rand_indices.end(), sampler::mt);	// データをシャッフル
			int num_train_data = num_data * train_dev_split;
			for(int i = 0;i < num_data;i++){
				rand_indices.push_back(i);
			}
			for(int i = 0;i < rand_indices.size();i++){
				int data_index = rand_indices[i];
				// 分割から元の文を復元
				std::vector<std::wstring> &words = corpus->_word_sequences[data_index];
				std::vector<int> segmentation;
				std::wstring sentence_str;
				for(auto word_str: words){
					assert(word_str.size() > 0);
					sentence_str += word_str;
					segmentation.push_back(word_str.size());
				}
				// 構成文字を辞書に追加し、文字IDに変換
				array<int> character_ids = array<int>(sentence_str.size());
				for(int i = 0;i < sentence_str.size();i++){
					wchar_t character = sentence_str[i];
					int character_id = dict->add_character(character);
					character_ids[i] = character_id;
				}
				// データセットに追加
				Sentence* sentence = new Sentence(sentence_str, character_ids);
				sentence->split(segmentation);		// 分割
				if(i < num_train_data){
					_sentences_train.push_back(sentence);
				}else{
					_sentences_dev.push_back(sentence);
				}
				// 統計
				if(_max_sentence_length == 0 || sentence_str.size() > _max_sentence_length){
					_max_sentence_length = sentence_str.size();
				}
				sum_sentence_length += sentence_str.size();
			}
			_avg_sentence_length = sum_sentence_length / (double)num_data;
		}
		Dataset::~Dataset(){
			for(int n = 0;n < _sentences_train.size();n++){
				Sentence* sentence = _sentences_train[n];
				delete sentence;
			}
			for(int n = 0;n < _sentences_dev.size();n++){
				Sentence* sentence = _sentences_dev[n];
				delete sentence;
			}
		}
		int Dataset::get_size_train(){
			return _sentences_train.size();
		}
		int Dataset::get_size_dev(){
			return _sentences_dev.size();
		}
		int Dataset::get_max_sentence_length(){
			return _max_sentence_length;
		}
		int Dataset::get_average_sentence_length(){
			return _avg_sentence_length;
		}
	}
}
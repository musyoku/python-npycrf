#pragma once
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../npycrf/common.h"
#include "../npycrf/sentence.h"
#include "corpus.h"
#include "dictionary.h"

namespace npycrf {
	namespace python {
		class Dataset{
		private:
			Corpus* _corpus;
			void _add_words_to_dataset(std::wstring &sentence_str, std::vector<Sentence*> &dataset, Dictionary* dict);
			void _detect_collision_of_sentence(Sentence* sentence, std::unordered_map<id, std::wstring> &pool, int max_word_length);
		public:
			int _max_sentence_length;
			int _avg_sentence_length;
			int _num_unsupervised_data;
			int _num_supervised_data;
			std::vector<Sentence*> _sentence_sequences_train;
			std::vector<Sentence*> _sentence_sequences_dev;
			Dataset(Corpus* corpus, Dictionary* dict, double train_split, int seed);
			~Dataset();
			int get_num_training_data();
			int get_num_validation_data();
			int get_max_sentence_length();
			int get_average_sentence_length();
			int detect_hash_collision(int max_word_length);
		};
	}
}
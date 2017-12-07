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
		public:
			int _max_sentence_length;
			int _avg_sentence_length;
			std::vector<Sentence*> _sentences_train;
			std::vector<Sentence*> _sentences_dev;
			Dataset(Corpus* corpus, Dictionary* dict, double train_dev_split, int seed);
			~Dataset();
			int get_size_train();
			int get_size_dev();
			int get_max_sentence_length();
			int get_average_sentence_length();
		};
	}
}
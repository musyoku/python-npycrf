#pragma once
#include <boost/python.hpp>
#include <vector>

namespace npycrf {
	namespace python {
		class Corpus{
		private:
		  void _before_python_add_words(boost::python::list &py_word_str_list, std::vector<std::wstring> &word_str_vec);
		public:
			std::vector<std::vector<std::wstring>> _word_sequences;
			Corpus(){}
			void add_words(std::vector<std::wstring> &word_str_vec);		// 正解の分割を追加する
			void python_add_words(boost::python::list py_word_str_list);
			int get_num_data();
		};
	}
}
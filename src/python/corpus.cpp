#include <boost/python.hpp>
#include <cassert>
#include <fstream>
#include "corpus.h"

namespace npycrf {
	namespace python {
		void Corpus::_before_python_add_words(boost::python::list &py_word_str_list, std::vector<std::wstring> &word_str_vec){
			int num_words = boost::python::len(py_word_str_list);
			for(int i = 0;i < num_words;i++){
				std::wstring word = boost::python::extract<std::wstring>(py_word_str_list[i]);
				word_str_vec.push_back(word);
			}
		}
		void Corpus::python_add_words(boost::python::list py_word_str_list){
			std::vector<std::wstring> word_str_vec;
			_before_python_add_words(py_word_str_list, word_str_vec);
			assert(word_str_vec.size() > 0);
			add_words(word_str_vec);
		}
		void Corpus::add_words(std::vector<std::wstring> &word_str_vec){
			assert(word_str_vec.size() >= 1);
			_word_sequences.push_back(word_str_vec);
		}
		int Corpus::get_num_data(){
			return _word_sequences.size();
		}
	}
}
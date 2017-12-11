#include <iostream>
#include "hash.h"
#include "sentence.h"

// <bos>と<eos>は長さが0文字であることに注意

namespace npycrf {
	Sentence::Sentence(std::wstring sentence, int* character_ids){
		_sentence_str = sentence;
		_characters = _sentence_str.data();
		_character_ids = new int[size()];
		for(int i = 0;i < size();i++){
			_character_ids[i] = character_ids[i];
		}
		_word_ids = new id[size() + 3];
		_segments = new int[size() + 3];
		_start = new int[size() + 3];
		_labels = new int[size() + 3];
		_features = NULL;
		for(int i = 0;i < size() + 3;i++){
			_word_ids[i] = 0;
			_segments[i] = 0;
			_labels[i] = 0;
		}
		_word_ids[0] = ID_BOS;
		_word_ids[1] = ID_BOS;
		_word_ids[2] = get_substr_word_id(0, size() - 1);
		_word_ids[3] = ID_EOS;
		_segments[0] = 1;
		_segments[1] = 1;
		_segments[2] = _sentence_str.size();
		_segments[3] = 1;
		_start[0] = 0;
		_start[1] = 0;
		_start[2] = 0;
		_start[3] = _sentence_str.size();
		_num_segments = 4;	// <bos>2つと<eos>1つを含む単語数. 4以上の値になる.
	}
	Sentence::~Sentence(){
		delete[] _character_ids;
		delete[] _segments;
		delete[] _start;
		delete[] _word_ids;
		if(_features != NULL){
			delete _features;
		}
	}
	Sentence* Sentence::copy(){
		Sentence* sentence = new Sentence(_sentence_str, _character_ids);
		if(_features != NULL){
			sentence->_features = _features->copy();
		}
		sentence->_num_segments = _num_segments;
		for(int n = 0;n < size() + 3;n++){
			sentence->_start[n] = _start[n];
			sentence->_word_ids[n] = _word_ids[n];
			sentence->_segments[n] = _segments[n];
		}
		return sentence;
	}
	// 文字数を返す
	// <bos>と<eos>は含まない
	int Sentence::size(){
		return _sentence_str.size();
	}
	// <bos>と<eos>を含む単語数を返す
	int Sentence::get_num_segments(){
		return _num_segments;
	}
	int Sentence::get_num_segments_without_special_tokens(){
		return _num_segments - 3;
	}
	int Sentence::get_word_length_at(int t){
		assert(t < _num_segments);
		return _segments[t];
	}
	id Sentence::get_word_id_at(int t){
		assert(t < _num_segments);
		return _word_ids[t];
	}
	id Sentence::get_substr_word_id(int start_index, int end_index){
		return hash_substring_ptr(_characters, start_index, end_index);
	}
	std::wstring Sentence::get_substr_word_str(int start_index, int end_index){
		std::wstring str(_sentence_str.begin() + start_index, _sentence_str.begin() + end_index + 1);
		return str;
	}
	// <bos>を考慮
	std::wstring Sentence::get_word_str_at(int t){
		assert(t < _num_segments);
		if(t < 2){
			return L"<bos>";
		}
		assert(t < _num_segments - 1);
		std::wstring str(_sentence_str.begin() + _start[t], _sentence_str.begin() + _start[t] + _segments[t]);
		return str;
	}
	void Sentence::dump_characters(){
		for(int i = 0;i < size();i++){
			std::cout << _characters[i] << ",";
		}
		std::cout << std::endl;
	}
	void Sentence::dump_words(){
		std::wcout << L" / ";
		for(int i = 2;i < _num_segments - 1;i++){
			for(int j = 0;j < _segments[i];j++){
				std::wcout << _characters[j + _start[i]];
			}
			std::wcout << L" / ";
		}
		std::wcout << std::endl;
	}
	// num_segmentsには<bos>や<eos>の数は含めない
	void Sentence::split(int* segments_without_special_tokens, int num_segments_without_special_tokens){
		int start = 0;
		int n = 0;
		int sum = 0;
		for(;n < num_segments_without_special_tokens;n++){
			if(segments_without_special_tokens[n] == 0){
				assert(n > 0);
				break;
			}
			sum += segments_without_special_tokens[n];
			_segments[n + 2] = segments_without_special_tokens[n];
			_word_ids[n + 2] = get_substr_word_id(start, start + segments_without_special_tokens[n] - 1);
			_start[n + 2] = start;
			start += segments_without_special_tokens[n];
		}
		assert(sum == _sentence_str.size());
		_segments[n + 2] = 1;
		_word_ids[n + 2] = ID_EOS;
		_start[n + 2] = _start[n + 1];
		n++;
		for(;n < _sentence_str.size();n++){
			_segments[n + 2] = 0;
			_start[n + 2] = 0;
		}
		_num_segments = num_segments_without_special_tokens + 3;
	}
	void Sentence::split(std::vector<int> &segments_without_special_tokens){
		int num_segments_without_special_tokens = segments_without_special_tokens.size();
		int start = 0;
		int n = 0;
		int sum = 0;
		for(;n < num_segments_without_special_tokens;n++){
			assert(segments_without_special_tokens[n] > 0);
			sum += segments_without_special_tokens[n];
			_segments[n + 2] = segments_without_special_tokens[n];
			_word_ids[n + 2] = get_substr_word_id(start, start + segments_without_special_tokens[n] - 1);
			_start[n + 2] = start;
			start += segments_without_special_tokens[n];
		}
		assert(sum == _sentence_str.size());
		_segments[n + 2] = 1;
		_word_ids[n + 2] = ID_EOS;
		_start[n + 2] = _sentence_str.size() - 1;
		n++;
		for(;n < _sentence_str.size();n++){
			_segments[n + 2] = 0;
			_start[n + 2] = 0;
		}
		_num_segments = num_segments_without_special_tokens + 3;
		
		// CRFのラベルを設定
		// 文字位置は1スタート、<bos>は1つ、<eos>は2つと考える
		_labels[0] = 1;
		int yt_1 = 1;
		int yt = 1;
		int i = 2;
		for(int t = 1;t <= size();t++){
			int s = (i < _num_segments - 1) ? _start[i] + 1 : size() + 1;
			yt_1 = yt;
			yt = 0;
			if(t == s){
				i = (i < _num_segments - 1) ? i + 1 : _num_segments - 1;
				yt = 1;
			}
			_labels[t] = yt;
		}
		_labels[size() + 1] = 1;	// <eos>
		_labels[size() + 2] = 1;	// <eos>
	}
	int Sentence::get_crf_label_at(int t){
		assert(t <= size() + 2);
		return _labels[t];
	}

} // namespace npycrf
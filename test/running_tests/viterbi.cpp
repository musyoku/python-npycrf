#include <iostream>
#include <chrono>
#include <sstream>
#include <utility>
#include "../../src/npycrf/sampler.h"
#include "../../src/npycrf/ctype.h"
#include "../../src/python/npycrf.h"
#include "../../src/python/dataset.h"
#include "../../src/python/dictionary.h"
#include "../../src/python/trainer.h"
#include "../../src/python/model/crf.h"
#include "../../src/python/model/npylm.h"
using namespace npycrf;
using namespace npycrf::python;
using std::cout;
using std::flush;
using std::endl;

void run_viterbi_decoding(){
	setlocale(LC_CTYPE, "ja_JP.UTF-8");
	std::ios_base::sync_with_stdio(false);
	std::locale default_loc("ja_JP.UTF-8");
	std::locale::global(default_loc);
	std::locale ctype_default(std::locale::classic(), default_loc, std::locale::ctype); //※
	std::wcout.imbue(ctype_default);
	std::wcin.imbue(ctype_default);
	
	Corpus* corpus = new Corpus();
	std::string filename_u = "../../dataset/aozora/kokoro.txt";
	std::wifstream ifs_u(filename_u.c_str());
	std::wstring sentence_str;
	assert(ifs_u.fail() == false);
	while(getline(ifs_u, sentence_str)){
		if (sentence_str.empty()){
			continue;
		}
		// std::wcout << sentence_str << std::endl;
		std::vector<std::wstring> words = {sentence_str};
		corpus->add_words(words);
	}
	ifs_u.close();

	model::NPYLM* py_npylm = new model::NPYLM("../../run/separate_files/out/npylm.model");
	model::CRF* py_crf = new model::CRF("../../run/separate_files/out/crf.model");
	NPYCRF* npycrf = new NPYCRF(py_npylm, py_crf);
	Dictionary* dict = new Dictionary("../../run/separate_files/out/char.dict");
	
	for(auto &words: corpus->_word_sequences){
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
			int character_id = dict->get_character_id(character);
			character_ids[i] = character_id;
		}
		// データセットに追加
		Sentence* sentence = new Sentence(sentence_str, character_ids);
		npycrf->parse(sentence);
		sentence->dump_words();
		delete sentence;
	}

	delete corpus;
	delete dict;
	delete npycrf;
}

int main(int argc, char *argv[]){
	for(int i = 0;i < 1000;i++){
		run_viterbi_decoding();
	}
}
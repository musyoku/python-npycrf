#pragma once
#include "../npycrf/common.h"

namespace npycrf {
	namespace python {
		class Dictionary{
		public:
			hashmap<wchar_t, int> _map_character_ids;	// すべての文字
			Dictionary(){
				_map_character_ids[CHARACTER_ID_UNK] = 0;	// <unk>
				_map_character_ids[CHARACTER_ID_BOS] = 1;	// <s>
				_map_character_ids[CHARACTER_ID_EOS] = 2;	// </s>
			}
			Dictionary(std::string filename);
			int add_character(wchar_t character);
			int get_character_id(wchar_t character);
			int get_num_characters();
			bool load(std::string filename);
			bool save(std::string filename);
		};
	}
}
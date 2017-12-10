#pragma once
#include "../npycrf/common.h"

namespace npycrf {
	namespace python {
		class Dictionary{
		public:
			hashmap<wchar_t, int> _map_character_ids;	// すべての文字
			Dictionary(){
				_map_character_ids[CHARACTER_ID_UNK] = CHARACTER_ID_UNK;	// <unk>
				_map_character_ids[CHARACTER_ID_BOS] = CHARACTER_ID_BOS;	// <s>
				_map_character_ids[CHARACTER_ID_EOS] = CHARACTER_ID_EOS;	// </s>
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
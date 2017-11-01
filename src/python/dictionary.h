#pragma once
#include "../npycrf/common.h"

namespace npycrf {
	namespace python {
		class Dictionary{
		public:
			hashmap<wchar_t, int> _map_character_ids;	// すべての文字
			Dictionary(){
				_map_character_ids[CHARACTER_ID_UNK] = CHARACTER_ID_UNK;	// <unk>
				_map_character_ids[CHARACTER_ID_BOW] = CHARACTER_ID_BOW;	// <s>
				_map_character_ids[CHARACTER_ID_EOW] = CHARACTER_ID_EOW;	// </s>
			}
			int add_character(wchar_t character);
			int get_character_id(wchar_t character);
			int get_num_characters();
			bool load(std::string filename);
			bool save(std::string filename);
		};
	}
}
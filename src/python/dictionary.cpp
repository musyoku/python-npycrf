#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <fstream>
#include "dictionary.h"

namespace npycrf {
	namespace python {
		int Dictionary::add_character(wchar_t character){
			auto itr = _map_character_ids.find(character);
			if(itr == _map_character_ids.end()){
				int character_id = _map_character_ids.size();
				_map_character_ids[character] = character_id;
				return character_id;
			}
			return itr->second;
		}
		int Dictionary::get_num_characters(){
			return _map_character_ids.size();
		}
		int Dictionary::get_character_id(wchar_t character){
			auto itr = _map_character_ids.find(character);
			if(itr == _map_character_ids.end()){
				return 0;
			}
			return itr->second;
		}
		bool Dictionary::load(std::string filename){
			std::string dictionary_filename = filename;
			std::ifstream ifs(dictionary_filename);
			if(ifs.good()){
				boost::archive::binary_iarchive iarchive(ifs);
				iarchive >> _map_character_ids;
				ifs.close();
				return true;
			}
			ifs.close();
			return false;
		}
		bool Dictionary::save(std::string filename){
			std::ofstream ofs(filename);
			boost::archive::binary_oarchive oarchive(ofs);
			oarchive << _map_character_ids;
			ofs.close();
			return true;
		}
	}
}
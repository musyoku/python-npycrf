#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <fstream>
#include <iostream>
#include "dictionary.h"

namespace npycrf {
	namespace python {
		Dictionary::Dictionary(std::string filename){
			if(load(filename) == false){
				std::cout << filename << " not found." << std::endl;
				exit(0);
			}
		}
		int Dictionary::add_character(wchar_t character){
			auto itr = _map_character_to_id.find(character);
			if(itr == _map_character_to_id.end()){
				int character_id = _map_character_to_id.size();
				_map_character_to_id[character] = character_id;
				_map_id_to_character[character_id] = character;
				return character_id;
			}
			return itr->second;
		}
		int Dictionary::get_num_characters(){
			return _map_character_to_id.size();
		}
		int Dictionary::get_character_id(wchar_t character){
			auto itr = _map_character_to_id.find(character);
			if(itr == _map_character_to_id.end()){
				return SPECIAL_CHARACTER_UNK;
			}
			return itr->second;
		}
		bool Dictionary::load(std::string filename){
			std::string dictionary_filename = filename;
			std::ifstream ifs(dictionary_filename);
			if(ifs.good()){
				boost::archive::binary_iarchive iarchive(ifs);
				iarchive >> _map_character_to_id;
				iarchive >> _map_id_to_character;
				ifs.close();
				return true;
			}
			ifs.close();
			return false;
		}
		bool Dictionary::save(std::string filename){
			std::ofstream ofs(filename);
			boost::archive::binary_oarchive oarchive(ofs);
			oarchive << _map_character_to_id;
			oarchive << _map_id_to_character;
			ofs.close();
			return true;
		}
	}
}
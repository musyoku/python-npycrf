#pragma once
#include <cassert>

namespace npycrf {
	namespace lattice {
		namespace array {
			template<typename T>
			class dual {
			private:
				void _delete(){
					for(int t = 0;t < _t_size;t++){
						delete[] _array[t];
					}
					delete[] _array;
				}
				void _alloc(){
					_array = new T*[_t_size];
					for(int t = 0;t < _t_size;t++){
						_array[t] = new T[_k_size];
						for(int k = 0;k < _k_size;k++){
							_array[t][k] = 0;
						}
					}
				}
			public:
				T** _array;
				int _t_size;
				int _k_size;
				dual(){
					_array = NULL;
					_t_size = 0;
					_k_size = 0;
				}
				dual(int t_size, int k_size){
					_alloc();
				}
				dual(const dual &a){
					_t_size = a._t_size;
					_k_size = a._k_size;
					if(_array != NULL){
						_delete();
					}
					std::cout << "lattice::array::dual allocated." << std::endl;
					_alloc();
					for(int t = 0;t < _t_size;t++){
						for(int k = 0;k < _k_size;k++){
							_array[t][k] = a._array[t][k];
						}
					}
				}
				~dual(){
					if(_array != NULL){
						_delete();
					}
				}
				dual &operator=(const dual &a){
					_t_size = a._t_size;
					_k_size = a._k_size;
					if(_array != NULL){
						_delete();
					}
					std::cout << "lattice::array::dual allocated." << std::endl;
					_alloc();
					for(int t = 0;t < _t_size;t++){
						for(int k = 0;k < _k_size;k++){
							_array[t][k] = a._array[t][k];
						}
					}
					return *this;
				}
				int size_t(){
					return _t_size;
				}
				int size_k(){
					return _k_size;
				}
				int operator()(int t, int k) {
					assert(t < _t_size);
					assert(k < _k_size);
					return _array[t][k];
				}
			};
			template<typename T>
			class triple {
			private:
				void _delete(){
					for(int t = 0;t < _t_size;t++){
						for(int k = 0;k < _k_size;k++){
							delete[] _array[t][k];
						}
						delete[] _array[t];
					}
					delete[] _array;
				}
				void _alloc(){
					_array = new T**[_t_size];
					for(int t = 0;t < _t_size;t++){
						_array[t] = new T*[_k_size];
						for(int k = 0;k < _k_size;k++){
							_array[t][k] = new T*[_j_size];
							for(int j = 0;j < _k_size;j++){
								_array[t][k][j] = 0;
							}
						}
					}
				}
			public:
				T*** _array;
				int _t_size;
				int _k_size;
				int _j_size;
				triple(){
					_array = NULL;
					_t_size = 0;
					_k_size = 0;
					_j_size = 0;
				}
				triple(int t_size, int k_size, int j_size){
					_alloc();
				}
				triple(const triple &a){
					_t_size = a._t_size;
					_k_size = a._k_size;
					_j_size = a._j_size;
					if(_array != NULL){
						_delete();
					}
					std::cout << "lattice::array::triple allocated." << std::endl;
					_alloc();
					for(int t = 0;t < _t_size;t++){
						for(int k = 0;k < _k_size;k++){
							for(int j = 0;j < _k_size;j++){
								_array[t][k][j] = a._array[t][k][j];
							}
						}
					}
				}
				~triple(){
					if(_array != NULL){
						_delete();
					}
				}
				triple &operator=(const triple &a){
					_t_size = a._t_size;
					_k_size = a._k_size;
					if(_array != NULL){
						_delete();
					}
					std::cout << "lattice::array::triple allocated." << std::endl;
					_alloc();
					for(int t = 0;t < _t_size;t++){
						for(int k = 0;k < _k_size;k++){
							for(int j = 0;j < _k_size;j++){
								_array[t][k][j] = a._array[t][k][j];
							}
						}
					}
					return *this;
				}
				int size_t(){
					return _t_size;
				}
				int size_k(){
					return _k_size;
				}
				int size_j(){
					return _j_size;
				}
				int operator()(int t, int k, int j) {
					assert(t < _t_size);
					assert(k < _k_size);
					assert(j < _j_size);
					return _array[t][k][j];
				}
			};
		}
	}
	template<typename T>
	class array {
	private:
		T* _array;
		int _size;
	public:
		array(){
			_array = NULL;
			_size = 0;
		}
		array(int size){
			_array = new T[size];
			_size = size;
		}
		array(const array &a){
			_size = a._size;
			if(_array != NULL){
				delete[] _array;
			}
			_array = new T[_size];
			for(int i = 0;i < _size;i++){
				_array[i] = a._array[i];
			}
		}
		~array(){
			if(_array != NULL){
				delete[] _array;
			}
		}
		array &operator=(const array &a){
			if(_array != NULL){
				delete[] _array;
			}
			_size = a._size;
			_array = new T[_size];
			for(int i = 0;i < _size;i++){
				_array[i] = a._array[i];
			}
			return *this;
		}
		T &operator[](int i){    // [] 演算子の多重定義
			assert(0 <= i && i < _size);
			return _array[i];
		}
		const T &operator[](int i) const {    // [] 演算子の多重定義
			assert(0 <= i && i < _size);
			return _array[i];
		}
		int size(){
			return _size;
		}
	};
}
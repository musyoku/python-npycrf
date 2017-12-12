#pragma once
namespace npycrf {
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
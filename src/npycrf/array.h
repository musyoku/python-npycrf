#pragma once
namespace npycrf {
	template<typename T>
	class array {
	private:
		T* _array;
		int _size;
	public:
		array(int size){
			_array = new T[size];
			_size = size;
		}
	};
}
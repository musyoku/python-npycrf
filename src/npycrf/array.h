#pragma once
#include <iostream>
#include <boost/format.hpp>
#include <boost/assert.hpp>

namespace npycrf {
	namespace mat {
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
				_t_size = t_size;
				_k_size = k_size;
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
			void fill(T value){
				for(int t = 0;t < _t_size;t++){
					for(int k = 0;k < _k_size;k++){
						_array[t][k] = value;
					}
				}
			}
			T &operator()(int t, int k) {
				BOOST_ASSERT_MSG(0 <= t && t < _t_size, (boost::format("0 <= %d < %d") % t % _t_size).str().c_str());
				BOOST_ASSERT_MSG(0 <= k && k < _k_size, (boost::format("0 <= %d < %d") % k % _k_size).str().c_str());
				return _array[t][k];
			}
			const T &operator()(int t, int k) const {
				BOOST_ASSERT_MSG(0 <= t && t < _t_size, (boost::format("0 <= %d < %d") % t % _t_size).str().c_str());
				BOOST_ASSERT_MSG(0 <= k && k < _k_size, (boost::format("0 <= %d < %d") % k % _k_size).str().c_str());
				return _array[t][k];
			}
		};
		template<typename T>
		class tri {
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
						_array[t][k] = new T[_j_size];
						for(int j = 0;j < _j_size;j++){
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
			tri(){
				_array = NULL;
				_t_size = 0;
				_k_size = 0;
				_j_size = 0;
			}
			tri(int t_size, int k_size, int j_size){
				_t_size = t_size;
				_k_size = k_size;
				_j_size = j_size;
				_alloc();
			}
			tri(const tri &a){
				_t_size = a._t_size;
				_k_size = a._k_size;
				_j_size = a._j_size;
				if(_array != NULL){
					_delete();
				}
				std::cout << "lattice::array::tri allocated." << std::endl;
				_alloc();
				for(int t = 0;t < _t_size;t++){
					for(int k = 0;k < _k_size;k++){
						for(int j = 0;j < _j_size;j++){
							_array[t][k][j] = a._array[t][k][j];
						}
					}
				}
			}
			~tri(){
				if(_array != NULL){
					_delete();
				}
			}
			tri &operator=(const tri &a){
				_t_size = a._t_size;
				_k_size = a._k_size;
				_j_size = a._j_size;
				if(_array != NULL){
					_delete();
				}
				std::cout << "lattice::array::tri allocated." << std::endl;
				_alloc();
				for(int t = 0;t < _t_size;t++){
					for(int k = 0;k < _k_size;k++){
						for(int j = 0;j < _j_size;j++){
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
			void fill(T value){
				for(int t = 0;t < _t_size;t++){
					for(int k = 0;k < _k_size;k++){
						for(int j = 0;j < _j_size;j++){
							_array[t][k][j] = value;
						}
					}
				}
			}
			T &operator()(int t, int k, int j) {
				BOOST_ASSERT_MSG(0 <= t && t < _t_size, (boost::format("0 <= %d < %d") % t % _t_size).str().c_str());
				BOOST_ASSERT_MSG(0 <= k && k < _k_size, (boost::format("0 <= %d < %d") % k % _k_size).str().c_str());
				BOOST_ASSERT_MSG(0 <= j && j < _j_size, (boost::format("0 <= %d < %d") % j % _j_size).str().c_str());
				return _array[t][k][j];
			}
			const T &operator()(int t, int k, int j) const {
				BOOST_ASSERT_MSG(0 <= t && t < _t_size, (boost::format("0 <= %d < %d") % t % _t_size).str().c_str());
				BOOST_ASSERT_MSG(0 <= k && k < _k_size, (boost::format("0 <= %d < %d") % k % _k_size).str().c_str());
				BOOST_ASSERT_MSG(0 <= j && j < _j_size, (boost::format("0 <= %d < %d") % j % _j_size).str().c_str());
				return _array[t][k][j];
			}
		};
		template<typename T>
		class quad {
		private:
			void _delete(){
				for(int t = 0;t < _t_size;t++){
					for(int k = 0;k < _k_size;k++){
						for(int j = 0;j < _j_size;j++){
							delete[] _array[t][k][j];
						}
						delete[] _array[t][k];
					}
					delete[] _array[t];
				}
				delete[] _array;
			}
			void _alloc(){
				_array = new T***[_t_size];
				for(int t = 0;t < _t_size;t++){
					_array[t] = new T**[_k_size];
					for(int k = 0;k < _k_size;k++){
						_array[t][k] = new T*[_j_size];
						for(int j = 0;j < _j_size;j++){
							_array[t][k][j] = new T[_i_size];
							for(int i = 0;i < _i_size;i++){
								_array[t][k][j][i] = 0;
							}
						}
					}
				}
			}
		public:
			T**** _array;
			int _t_size;
			int _k_size;
			int _j_size;
			int _i_size;
			quad(){
				_array = NULL;
				_t_size = 0;
				_k_size = 0;
				_j_size = 0;
				_i_size = 0;
			}
			quad(int t_size, int k_size, int j_size, int i_size){
				_t_size = t_size;
				_k_size = k_size;
				_j_size = j_size;
				_i_size = i_size;
				_alloc();
			}
			quad(const quad &a){
				_t_size = a._t_size;
				_k_size = a._k_size;
				_j_size = a._j_size;
				_i_size = a._i_size;
				if(_array != NULL){
					_delete();
				}
				std::cout << "lattice::array::quad allocated." << std::endl;
				_alloc();
				for(int t = 0;t < _t_size;t++){
					for(int k = 0;k < _k_size;k++){
						for(int j = 0;j < _j_size;j++){
							for(int i = 0;i < _i_size;i++){
								_array[t][k][j][i] = a._array[t][k][j][i];
							}
						}
					}
				}
			}
			~quad(){
				if(_array != NULL){
					_delete();
				}
			}
			quad &operator=(const quad &a){
				_t_size = a._t_size;
				_k_size = a._k_size;
				_j_size = a._j_size;
				_i_size = a._i_size;
				if(_array != NULL){
					_delete();
				}
				std::cout << "lattice::array::quad allocated." << std::endl;
				_alloc();
				for(int t = 0;t < _t_size;t++){
					for(int k = 0;k < _k_size;k++){
						for(int j = 0;j < _j_size;j++){
							for(int i = 0;i < _i_size;i++){
								_array[t][k][j][i] = a._array[t][k][j][i];
							}
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
			int size_i(){
				return _i_size;
			}
			void fill(T value){
				for(int t = 0;t < _t_size;t++){
					for(int k = 0;k < _k_size;k++){
						for(int j = 0;j < _j_size;j++){
							for(int i = 0;i < _i_size;i++){
								_array[t][k][j][i] = value;
							}
						}
					}
				}
			}
			T &operator()(int t, int k, int j, int i) {
				BOOST_ASSERT_MSG(0 <= t && t < _t_size, (boost::format("0 <= %d < %d") % t % _t_size).str().c_str());
				BOOST_ASSERT_MSG(0 <= k && k < _k_size, (boost::format("0 <= %d < %d") % k % _k_size).str().c_str());
				BOOST_ASSERT_MSG(0 <= j && j < _j_size, (boost::format("0 <= %d < %d") % j % _j_size).str().c_str());
				BOOST_ASSERT_MSG(0 <= i && i < _i_size, (boost::format("0 <= %d < %d") % i % _i_size).str().c_str());
				return _array[t][k][j][i];
			}
			const T &operator()(int t, int k, int j, int i) const {
				BOOST_ASSERT_MSG(0 <= t && t < _t_size, (boost::format("0 <= %d < %d") % t % _t_size).str().c_str());
				BOOST_ASSERT_MSG(0 <= k && k < _k_size, (boost::format("0 <= %d < %d") % k % _k_size).str().c_str());
				BOOST_ASSERT_MSG(0 <= j && j < _j_size, (boost::format("0 <= %d < %d") % j % _j_size).str().c_str());
				BOOST_ASSERT_MSG(0 <= i && i < _i_size, (boost::format("0 <= %d < %d") % i % _i_size).str().c_str());
				return _array[t][k][j][i];
			}
		};
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
			BOOST_ASSERT_MSG(0 <= i && i < _size, (boost::format("0 <= %d < %d") % i % _size).str().c_str());
			return _array[i];
		}
		const T &operator[](int i) const {    // [] 演算子の多重定義
			BOOST_ASSERT_MSG(0 <= i && i < _size, (boost::format("0 <= %d < %d") % i % _size).str().c_str());
			return _array[i];
		}
		int size(){
			return _size;
		}
	};
}
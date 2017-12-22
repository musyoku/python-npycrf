#include <boost/serialization/split_member.hpp>
#include "parameter.h"
#include "../sampler.h"

namespace npycrf {
	namespace crf {
		Parameter::Parameter(){

		}
		Parameter::~Parameter(){

		}
		Parameter::Parameter(double weight_size, double lambda_0, double sigma){
			_bias = 0;
			_pruned = false;
			_weights = array<double>(weight_size);
			for(int i = 0;i < weight_size;i++){
				_weights[i] = sampler::uniform(-0.0001, 0.0001);
			}
			_lambda_0 = lambda_0;
			_sigma = sigma;
		}
		double Parameter::weight_at_index(int index){
			return _weights[index];
		}
		const double &Parameter::operator[](int i) const {    // [] 演算子の多重定義
			return _weights[i];
		}
		int Parameter::get_num_features(){
			return _weights.size();
		}
		template <class Archive>
		void Parameter::serialize(Archive &ar, unsigned int version)
		{
			boost::serialization::split_member(ar, *this, version);
		}
		template void Parameter::serialize(boost::archive::binary_iarchive &ar, unsigned int version);
		template void Parameter::serialize(boost::archive::binary_oarchive &ar, unsigned int version);
		void Parameter::save(boost::archive::binary_oarchive &ar, unsigned int version) const {
			int size = _weights.size();
			ar & size;
			for(int k = 0;k < size;k++){
				ar & k;
				ar & _weights[k];
			}
			ar & _bias;
			ar & _lambda_0;
		}
		void Parameter::load(boost::archive::binary_iarchive &ar, unsigned int version) {
			int size = 0;
			ar & size;
			_weights = array<double>(size);
			for(int n = 0;n < size;n++){
				int k;
				double value;
				ar & k;
				ar & value;
				_weights[k] = value;
			}
			ar & _bias;
			ar & _lambda_0;
		}
	}
}
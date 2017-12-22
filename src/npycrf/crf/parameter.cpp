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
			_all_weights = array<double>(weight_size);
			for(int i = 0;i < weight_size;i++){
				_all_weights[i] = sampler::uniform(-0.0001, 0.0001);
			}

			_num_updates = array<int>(weight_size);
			for(int i = 0;i < weight_size;i++){
				_num_updates[i] = 0;
			}

			_lambda_0 = lambda_0;
			_sigma = sigma;
		}
		double Parameter::weight_at_index(int index){
			if(_pruned){
				auto itr = _effective_weights.find(index);
				if(itr == _effective_weights.end()){
					return 0;
				}
				return itr->second;
			}
			return _all_weights[index];
		}
		void Parameter::set_weight_at_index(int index, double value){
			if(_pruned){
				_effective_weights[index] = value;
			}else{
				_all_weights[index] = value;
			}
		}
		int Parameter::get_num_features(){
			if(_pruned){
				return _effective_weights.size();
			}
			return _all_weights.size();
		}
	}
}
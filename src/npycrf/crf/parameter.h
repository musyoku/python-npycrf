#include "../common.h"
#include "../array.h"

namespace npycrf {
	namespace crf {
		class Parameter {
		public:
			double _bias;
			array<double> _all_weights;		// 全ての重み
			hashmap<int, double> _effective_weights;	// 枝刈りされた重み
			array<int> _num_updates;
			int _weight_size;
			double _lambda_0;	// モデル補完重み
			double _sigma;		// パラメータの事前分布の標準偏差
			double weight_at_index(int index);
			bool _pruned;
			void set_weight_at_index(int index, double value);
			Parameter(double weight_size, double lambda_0, double sigma);
			~Parameter();
			int get_num_features();
		};
	}
}
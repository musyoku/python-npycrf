#pragma once
namespace npycrf {
	namespace crf {
		class FeatureIndices{
		public:
			int** _num_features_u;		// 位置tにおけるy_tに関する全ての素性IDの数. CRFに合わせて1スタート.
			int*** _num_features_b;		// 位置tにおけるy_{t-1}, y_tに関する全ての素性IDの数. CRFに合わせて1スタート.
			int*** _feature_indices_u;	// 位置tにおけるy_tに関する全ての素性ID. CRFに合わせて1スタート.
			int**** _feature_indices_b;	// 位置tにおけるy_{t-1}, y_tに関する全ての素性ID. CRFに合わせて1スタート.
		};
	}
}
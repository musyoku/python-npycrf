#pragma once
#include "../../array.h"

namespace npycrf {
	namespace crf {
		namespace feature {
			class FeatureIndices {
			public:
				int _seq_length;
				mat::bi<int> _num_features_u;	// 位置tにおけるy_tに関する全ての素性IDの数. CRFに合わせてtは1スタート.
				mat::tri<int> _num_features_b;	// 位置tにおけるy_{t-1}, y_tに関する全ての素性IDの数. CRFに合わせてtは1スタート.
				int*** _feature_indices_u;		// 位置tにおけるy_tに関する全ての素性ID. CRFに合わせてtは1スタート.
				int**** _feature_indices_b;		// 位置tにおけるy_{t-1}, y_tに関する全ての素性ID. CRFに合わせてtは1スタート.
				FeatureIndices* copy();
			};
		}
	}
}
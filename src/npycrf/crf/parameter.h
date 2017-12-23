#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "../common.h"
#include "../array.h"

namespace npycrf {
	namespace crf {
		class Parameter {
		private:
			friend class boost::serialization::access;
			template <class Archive>
			void serialize(Archive &archive, unsigned int version);
			void save(boost::archive::binary_oarchive &ar, unsigned int version) const;
			void load(boost::archive::binary_iarchive &ar, unsigned int version);
		public:
			double _bias;
			array<double> _weights;		// 重み
			double _lambda_0;	// モデル補完重み
			double _sigma;		// パラメータの事前分布の標準偏差
			Parameter();
			Parameter(double weight_size, double lambda_0, double sigma);
			~Parameter();
			int get_num_features();
		};
	}
}
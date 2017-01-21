// Copyright 2016 Husky Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <random>

#include "lib/vector.hpp"

namespace husky {
namespace lib {
namespace ml {

// extends LabeledPoint, using Vector to represent features
template <typename FeatureT, typename LabelT, bool is_sparse>
class LabeledPointHObj : public LabeledPoint<Vector<FeatureT, is_sparse>, LabelT> {
   public:
    using FeatureV = Vector<FeatureT, is_sparse>;
    // Object key
    using KeyT = int;
    KeyT key;
    const KeyT& id() const { return key; }

    // constructors
    LabeledPointHObj() : LabeledPoint<FeatureV, LabelT>() { init(); }
    explicit LabeledPointHObj(int feature_num) : LabeledPoint<FeatureV, LabelT>() { this->x = FeatureV(feature_num); init(); }
    LabeledPointHObj(FeatureV& x, LabelT& y) : LabeledPoint<FeatureV, LabelT>(x, y) { init(); }
    LabeledPointHObj(FeatureV&& x, LabelT&& y) : LabeledPoint<FeatureV, LabelT>(x, y) { init(); }

    void init() {
		std::mt19937 rng;
		rng.seed(std::random_device()());
		std::uniform_int_distribution<std::mt19937::result_type> dist(1, 1 << 31);
		key = dist(rng);
    }

    friend BinStream& operator<<(husky::BinStream& stream, LabeledPointHObj<FeatureT, LabelT, is_sparse>& obj) {
        stream << obj.key << obj.x << obj.y;
        return stream;
    }

    friend BinStream& operator>>(husky::BinStream& stream, LabeledPointHObj<FeatureT, LabelT, is_sparse>& obj) {
        stream >> obj.key >> obj.x >> obj.y;
        return stream;
    }
};  // LabeledPointHObj

}  // namespace ml
}  // namespace lib
}  // namespace husky

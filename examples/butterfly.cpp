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
//
//
// master_host=master
// master_port=15444
// comm_port=19832
// hdfs_namenode=master
// hdfs_namenode_port=9000
// train=hdfs:///datasets/classification/a9
// test=hdfs:///datasets/classification/a9
// is_sparse=true
// format=libsvm
// n_iter=500
// alpha=0.5
// 
// # For Master
// serve=1
// 
// # Session for worker information
// [worker]
// info=w1:4
// info=w2:4
// info=w3:4
// info=w5:4

#include <boost/thread/thread.hpp>
#include <cmath>
#include <random>
#include <utility>
#include <string>
#include <vector>

#include "base/serialization.hpp"
#include "base/log.hpp"
#include "base/serialization.hpp"
#include "core/engine.hpp"
#include "core/mailbox.hpp"
#include "core/utils.hpp"
#include "lib/vector.hpp"
#include "lib/ml/data_loader.hpp"
#include "lib/ml/feature_label.hpp"
#include "lib/ml/parameter.hpp"

std::string vector_to_string(std::vector<int> v) {
    std::ostringstream oss;
    std::copy(v.begin(), v.end()-1, std::ostream_iterator<int>(oss, ","));
    oss << v.back();
    return oss.str();
}

int butterfly_idx(int node_idx, int t, int n_node) {
	auto intceil = [&](double x) { return (int) (ceil(x) + 0.5); };
	auto group = [&](int k) { return intceil((k - 0.5) / 2); };
	auto diff = [&](int a, int b) { 
		int res = a - b;
		return (res >= 0) ? res : -res;
	};
	auto fly = [&](int pow) { return (pow < 0) ? 0 : (1 << pow); };
	int logN = int(log(n_node)/log(2) + 0.5);
	int i = (t % logN);
	int j = node_idx + (1 << i);
	int next_i = (i + 1);
	if (diff(group(node_idx), group(j)) > fly(i-1)) {
		j -= (1 << next_i);
	}
	j = ((j + n_node - 1) % n_node) + 1;
	return j;
}

void send_to_fly(husky::base::BinStream bin, husky::LocalMailbox* mailbox) {
    static thread_local int time = 0;
    int sender_tid = husky::Context::get_global_tid();
    bin << sender_tid;
    int target_tid = butterfly_idx(sender_tid + 1, time++, husky::Context::get_num_workers()) - 1;
    mailbox->send(target_tid, 0, 0, bin);
    // husky::LOG_I << "sender: " << sender_tid << ", going to: " << target_tid << ", time: " << time;
    mailbox->send_complete(0, 0, husky::Context::get_worker_info().get_local_tids(), husky::Context::get_worker_info().get_pids());
}
husky::base::BinStream receiving() {
    husky::LocalMailbox* mailbox = husky::Context::get_mailbox();
    while (!mailbox->poll(0, 0)) { boost::this_thread::sleep(boost::posix_time::milliseconds(100)); };
    return mailbox->recv(0, 0);
}

void butterfly() {
    int N = 8;
    husky::LocalMailbox* mailbox = husky::Context::get_mailbox();
    std::vector<int> weight = {1, 2, 3, 4, 5};
    husky::base::BinStream bin;
    bin << weight;
    send_to_fly(bin, mailbox);
    husky::base::BinStream recv_bin = receiving();
    std::vector<int> ans;
    recv_bin >> ans;
    husky::LOG_I << "recv tid: " << husky::Context::get_global_tid() << ", recv msg: " << vector_to_string(ans);
}

void regression() {
    using LabeledPointHObj = husky::lib::ml::LabeledPointHObj<double, double, true>;
    auto& train_set = husky::ObjListStore::create_objlist<LabeledPointHObj>("train_set");
    auto& test_set = husky::ObjListStore::create_objlist<LabeledPointHObj>("test_set");

    // load data
    auto format_str = husky::Context::get_param("format");
    husky::lib::ml::DataFormat format;
    if (format_str == "libsvm") {
        format = husky::lib::ml::kLIBSVMFormat;
    } else if (format_str == "tsv") {
        format = husky::lib::ml::kTSVFormat;
    }

    int num_features = husky::lib::ml::load_data(husky::Context::get_param("train"), train_set, format);
    husky::globalize(train_set);
    husky::lib::ml::load_data(husky::Context::get_param("test"), test_set, format, num_features);
    // husky::globalize(test_set);

    // processing labels
    husky::list_execute(train_set, [](auto& this_obj) {
        if (this_obj.y < 0)
            this_obj.y = 0;
    });
    husky::list_execute(test_set, [](auto& this_obj) {
        if (this_obj.y < 0)
            this_obj.y = 0;
    });

    double alpha = std::stod(husky::Context::get_param("alpha"));
    int num_iter = std::stoi(husky::Context::get_param("n_iter"));

    // initialize logistic regression model
    int num_feature_ = num_features;

    // gradient function: X * (true_y - innerproduct(xv, pv))
    using ObjT = LabeledPointHObj;
    using FeatureT = double;
    using LabelT = double;
    using ParamT = husky::lib::ml::ParameterBucket<double>;

    auto gradient_func_ = [](ObjT& this_obj, husky::lib::Vector<FeatureT, false>& param) {
        auto vec_X = this_obj.x;
        auto pred_y = param.dot_with_intcpt(vec_X);
        pred_y = static_cast<FeatureT>(1. / (1. + exp(-pred_y)));
        auto delta = static_cast<FeatureT>(this_obj.y) - pred_y;
        vec_X *= delta;
        int num_param = param.get_feature_num();
        vec_X.resize(num_param);
        vec_X.set(num_param - 1, delta);  // intercept factor
        return vec_X;                     // gradient vector
    };

    auto error_func_ = [](ObjT& this_obj, husky::lib::Vector<FeatureT, false>& param) {
        auto vec_X = this_obj.x;
        auto pred_y = param.dot_with_intcpt(vec_X);
        // husky::LOG_I << pred_y;
        // pred_y = static_cast<FeatureT>(1. / (1. + exp(-pred_y)));
        // return ((0.0 - pred_y) <= 1e-6);
        int output = (pred_y >= 0.0) ? 1 : 0;
        return (output == this_obj.y) ? 0 : 1;
    };

    // train the model 
    std::vector<ObjT> vec_ObjL_train;

    list_execute(train_set, [&](auto& obj) {
        vec_ObjL_train.push_back(obj);
    });

    husky::lib::Aggregator<int> num_samples_agg(0, [](int& a, const int& b) { a += b; });
    int num_train_obj =  vec_ObjL_train.size();
    auto& ac = husky::lib::AggregatorFactory::get_channel();
    list_execute(train_set, {}, {&ac}, [&](ObjT& Obj) {
        num_samples_agg.update(1);
    });
    int num_global_samples = num_samples_agg.get_value();
    // husky::LOG_I << "size of train: " << num_train_obj;

    husky::lib::Vector<FeatureT, false> param(num_features + 1);
    husky::LocalMailbox* mailbox = husky::Context::get_mailbox();

    auto show_vec = [&](husky::lib::Vector<FeatureT, false> vec) {
        int param_idx = 0;
        for (auto it = vec.begin_feaval(); it != vec.end_feaval(); ++it) {
            const auto& w = *it;
            husky::LOG_I << param_idx++ << ": " << w.val;
        }
    };

    for (size_t round_t = 0; round_t < num_iter; round_t++) {
        husky::lib::Vector<FeatureT, false> sum_grad(num_features + 1);
        for (size_t i = 0; i < num_train_obj; i++) {
            auto grad = gradient_func_(vec_ObjL_train[i], param);
            sum_grad += grad;
        }
        // show_vec(sum_grad);
        sum_grad *= alpha / num_train_obj;
        param += sum_grad;
        // send the param out
        husky::base::BinStream send_bin;
        send_bin << sum_grad;
        send_to_fly(send_bin, mailbox);
        husky::base::BinStream recv_bin = receiving();
        husky::lib::Vector<FeatureT, false> recv_vec_grad;
        recv_bin >> recv_vec_grad;
        param += ((sum_grad + recv_vec_grad) / 2.0);

        // husky::lib::Vector<FeatureT, false> zero_grad(num_features + 1);
        // husky::LOG_I << "zero: " << zero_grad.get_feature_num();
        // husky::lib::Aggregator<husky::lib::Vector<FeatureT, false>> sum_grad(zero_grad, 
        //         [&](husky::lib::Vector<FeatureT, false>& a, const husky::lib::Vector<FeatureT, false>& b) { 
        //             husky::LOG_I << "a: " << a.get_feature_num() << ", b: " << b.get_feature_num();
        //             a += b; 
        //         });
        // list_execute(train_set, {}, {&ac}, [&](ObjT& Obj) {
        //     auto grad = gradient_func_(Obj, param);
        //     sum_grad.update(grad);
        // });
        // param += alpha * sum_grad.get_value() / num_global_samples;
    }

    husky::lib::Aggregator<int> num_test_sample(0, [](int& a, const int& b) { a += b; });
    husky::lib::Aggregator<int> error_agg(0, [](int& a, const int& b) { a += b; });
    list_execute(train_set, {}, {&ac}, [&](ObjT& Obj) {
        num_test_sample.update(1);
        error_agg.update(error_func_(Obj, param));
    });
    list_execute(train_set, {}, {&ac}, [&](ObjT& Obj) {});
    if (husky::Context::get_global_tid() == 0) {
        show_vec(param);
        husky::LOG_I << "wrong: " << error_agg.get_value();
        husky::LOG_I << "all: " << num_test_sample.get_value();
        husky::LOG_I << "ratio: " << static_cast<double>(error_agg.get_value())/num_test_sample.get_value();
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args(8);
    args[0] = "hdfs_namenode";
    args[1] = "hdfs_namenode_port";
    args[2] = "train";
    args[3] = "test";
    args[4] = "n_iter";
    args[5] = "alpha";
    args[6] = "format";
    args[7] = "is_sparse";
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(regression);
        return 0;
    }
    return 1;
}

#include <functional>
#include <utility>
#include <vector>

#include "caffe/ex_layers/statistics_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StatisticLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void StatisticLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (current_iter % this->layer_param_.slice_param().axis() == 1 ) {
    Dtype sum = 0;
    const int count = bottom[0]->count();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    for (int index = 0; index < count; index++) {
      if (bottom_data[index] < Dtype(1e-8))
          sum ++;
    }
    // LOG(INFO) << "Statistic: " << accuracy;
    top[0]->mutable_cpu_data()[0] = sum / count;
    // Statistic layer should not be used as a loss function.
  }
  current_iter ++;
}

INSTANTIATE_CLASS(StatisticLayer);
REGISTER_LAYER_CLASS(Statistic);

}  // namespace caffe

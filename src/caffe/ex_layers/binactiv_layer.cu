#include <vector>

#include "caffe/ex_layers/binactiv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#define sign(x) ((x)>=0?1:-1)

template <typename Dtype>
__global__ void BinActiv_FW_Kernal(const int count, const int C, const int map_size, const Dtype* bottom_data, Dtype* sumA, Dtype* signA) {
  CUDA_KERNEL_LOOP(index, count) {
    const int batch_id = index / map_size;
    const int coordinate = index % map_size;
    sumA[index] = 0;
    for (int _c = 0; _c < C; _c++) {
      sumA[index] += bottom_data[ batch_id*C*map_size + _c*map_size + coordinate ] / C;
      signA[ batch_id*C*map_size + _c*map_size + coordinate ] =
        sign( bottom_data[ batch_id*C*map_size + _c*map_size + coordinate ] );
    }
  }
}

template <typename Dtype>
void BinActivLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ",,, Forward_gpu start ";
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* sumA = convolution_bottom_vec_[0]->mutable_gpu_data();
  Dtype* signA = top[0]->mutable_gpu_data();
  const int C = bottom[0]->channels();
  const int count = convolution_bottom_vec_[0]->count();
  const int map_size = bottom[0]->height()*bottom[0]->width();
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ",,, Forward_gpu C : " << C << ", count : " << count << ", map_size : " << map_size;
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ",,, signA : " << top[0]->num() << ", " << top[0]->channels() << ", " << top[0]->height() << ", " << top[0]->width();
  BinActiv_FW_Kernal<Dtype>
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, C, map_size, bottom_data, sumA, signA);

  DLOG(INFO) << "-----> " << this->layer_param_.name() << ",,, BinActiv_FW_Kernal Done ";
  CUDA_POST_KERNEL_CHECK;
  //cudaDeviceSynchronize();
  convolution_layer_->Forward(convolution_bottom_vec_, convolution_top_vec_);

  DLOG(INFO) << "-----> " << this->layer_param_.name() << ",,, convolution_layer Forward";
  const int size_kernal = this->layer_param_.convolution_param().kernel_size(0)
        * this->layer_param_.convolution_param().kernel_size(0);
  CHECK_EQ(top[1]->count(), convolution_top_vec_[0]->count());
  caffe_copy(top[1]->count(), convolution_top_vec_[0]->gpu_data(), top[1]->mutable_gpu_data());
  caffe_gpu_scale(top[1]->count(), Dtype(1)/size_kernal, top[1]->gpu_data(),  top[1]->mutable_gpu_data());
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ",,, Forward_gpu left ";
}

template <typename Dtype>
__global__ void BinActiv_BP_Kernal(const int count, const Dtype* bottom_data, const Dtype* top_diff, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, count) {
    if ( std::abs(bottom_data[index]) <= Dtype(1) ) {
      bottom_diff[ index ] = top_diff[ index ];
    } else {
      bottom_diff[ index ] = Dtype(0);
    }
  }
}

template <typename Dtype>
void BinActivLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if ( propagate_down[0] == false ) return;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = top[0]->count();
  BinActiv_BP_Kernal<Dtype>
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_diff, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(BinActivLayer);

}  // namespace caffe

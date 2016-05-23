#include <algorithm>
#include <vector>

#include "caffe/ex_layers/binactiv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#define sign(x) ((x)>=0?1:-1)

template <typename Dtype>
void BinActivLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), 2);
  CHECK_EQ(bottom.size(), 1);
  LayerParameter convolution_param(this->layer_param_);
  convolution_param.set_type("Convolution");
  convolution_param.mutable_convolution_param()->CopyFrom(
    this->layer_param_.convolution_param() );
  convolution_layer_ = LayerRegistry<Dtype>::CreateLayer(convolution_param);
  convolution_bottom_shared_.reset(new Blob<Dtype>());
  convolution_bottom_vec_.clear();
  convolution_bottom_vec_.push_back(convolution_bottom_shared_.get());
  convolution_bottom_vec_[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  convolution_top_shared_.reset(new Blob<Dtype>());
  convolution_top_vec_.clear();
  convolution_top_vec_.push_back(convolution_top_shared_.get());
  convolution_layer_->SetUp(convolution_bottom_vec_, convolution_top_vec_);
}

template <typename Dtype>
void BinActivLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ",,, enter Reshape";
  CHECK_EQ(top.size(), 2);
  CHECK_EQ(bottom.size(), 1);
  top[0]->ReshapeLike(*bottom[0]);
  convolution_bottom_vec_[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  top[1]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  convolution_layer_->Reshape(convolution_bottom_vec_, convolution_top_vec_);
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ", bottom[0] shape : " << bottom[0]->num() << ", " << bottom[0]->channels() << ", " << bottom[0]->height() << ", " << bottom[0]->width();
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ", top[0] shape : " << top[0]->num() << ", " << top[0]->channels() << ", " << top[0]->height() << ", " << top[0]->width();
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ", top[1] shape : " << top[1]->num() << ", " << top[1]->channels() << ", " << top[1]->height() << ", " << top[1]->width();
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ", conv_bottom_vec[0] shape : " << convolution_bottom_vec_[0]->num() << ", " << convolution_bottom_vec_[0]->channels() << ", " << convolution_bottom_vec_[0]->height() << ", " << convolution_bottom_vec_[0]->width();
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ", conv_top_vec[0] shape : " << convolution_top_vec_[0]->num() << ", " << convolution_top_vec_[0]->channels() << ", " << convolution_top_vec_[0]->height() << ", " << convolution_top_vec_[0]->width();
}

template <typename Dtype>
void BinActivLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* sumA = convolution_bottom_vec_[0]->mutable_cpu_data();
  Dtype* signA = top[0]->mutable_cpu_data();
  const int C = bottom[0]->channels();
  for (int index = 0; index < bottom[0]->num(); index++ ) {
    for (int _h = 0; _h < bottom[0]->height(); _h++ ) {
      for (int _w = 0; _w < bottom[0]->width(); _w++ ) {
        sumA[ top[1]->offset(index,0,_h,_w) ] = 0;
        for (int _c = 0; _c < bottom[0]->channels(); _c++ ) {
          sumA[ top[1]->offset(index,0,_h,_w) ] += 
            bottom_data[ bottom[0]->offset(index,_c,_h,_w) ] / C;
          signA[ top[0]->offset(index,_c,_h,_w)] =
            sign(bottom_data[ bottom[0]->offset(index,_c,_h,_w) ]);
        }
      }
    }
  }
  convolution_layer_->Forward(convolution_bottom_vec_, convolution_top_vec_);
  const int size_kernal = this->layer_param_.convolution_param().kernel_size(0)
        * this->layer_param_.convolution_param().kernel_size(0);
  CHECK_EQ(top[1]->count(), convolution_top_vec_[0]->count());
  caffe_copy(top[1]->count(), convolution_top_vec_[0]->cpu_data(), top[1]->mutable_cpu_data());
  caffe_cpu_scale(top[1]->count(), Dtype(1)/size_kernal, top[1]->cpu_data(),  top[1]->mutable_cpu_data());
}

template <typename Dtype>
void BinActivLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if ( propagate_down[0] == false ) return;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int index = 0; index < bottom[0]->count(); index++) {
    if ( std::abs(bottom_data[index]) <= Dtype(1) ) {
      bottom_diff[ index ] = top_diff[ index ];
    } else {
      bottom_diff[ index ] = Dtype(0);
    }
  } 
}

#ifdef CPU_ONLY
STUB_GPU(BinActivLayer);
#endif

INSTANTIATE_CLASS(BinActivLayer);
REGISTER_LAYER_CLASS(BinActiv);

}  // namespace caffe

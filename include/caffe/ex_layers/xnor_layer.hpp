#ifndef CAFFE_XNORNET_LAYER_HPP_
#define CAFFE_XNORNET_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/ex_layers/binary_conv_layer.hpp"
#include "caffe/ex_layers/split_concat_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"

namespace caffe {

/**
 * @brief Merge the XnorNet -> SplitConcat | BinaryConvolution -> Eltwise layer,
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 * Input Real Value, binarize the input and use binary weight to do convolution.
 */
template <typename Dtype>
class XnorNetLayer : public Layer<Dtype> {
 public:
  explicit XnorNetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "XnorNet"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// BinActiv : norm1 -> [0]B-norm1 , [1]K-norm1-single
  shared_ptr<Layer<Dtype> > binactiv_layer_;
  vector<Blob<Dtype>*> binactiv_top_vec_;
  shared_ptr<Blob<Dtype> >  binactiv_top_vec_shared_;
  
  /// SplitConcat : K-norm1-single -> K-norm1 , channels multi
  shared_ptr<Layer<Dtype> > splitconcat_layer_;
  vector<Blob<Dtype>*> splitconcat_bottom_vec_;
  vector<Blob<Dtype>*> splitconcat_top_vec_;

  /// BinaryConvolution : X * W : all BinaryConvolution
  shared_ptr<Layer<Dtype> > binaryconvolution_layer_;
  vector<Blob<Dtype>*> binaryconvolution_bottom_vec_;
  vector<Blob<Dtype>*> binaryconvolution_top_vec_;
  shared_ptr<Blob<Dtype> >  binaryconvolution_top_vec_shared_;

  // Eltwise (PROD) : K-norm1 * B-conv2 -> conv2
  shared_ptr<Layer<Dtype> > eltwise_layer_;
  vector<Blob<Dtype>*> eltwise_bottom_vec_;
  vector<Blob<Dtype>*> eltwise_top_vec_;
  
};

}  // namespace caffe

#endif  // CAFFE_XNORNET_LAYER_HPP_

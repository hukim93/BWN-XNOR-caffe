clear;clc;
current_dir = pwd;
caffe_dir = '../../'; cd(caffe_dir); caffe_dir = pwd;
cd(current_dir);
addpath(fullfile(caffe_dir,'matlab'));
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(4);

rand('seed',0);
cifar10_train_data = load(fullfile(caffe_dir,'examples','cifar10','cifar10_train_lmdb.mat'));
cifar10_train_data.image = single(permute(cifar10_train_data.image,[3,2,4,1]));
cifar10_test_data = load(fullfile(caffe_dir,'examples','cifar10','cifar10_test_lmdb.mat'));
cifar10_test_data.image = single(permute(cifar10_test_data.image,[3,2,4,1]));
train_num = size(cifar10_train_data.label, 1);
test_num = size(cifar10_test_data.label, 1);

solver_file = fullfile(caffe_dir,'matlab','Binary','cifar10_full.matlab.solver');
solver = caffe.Solver(solver_file);
caffemodel = fullfile(caffe_dir,'examples','cifar10','cifar10_full_iter_90000.caffemodel');
solver.net.copy_from(caffemodel);

mean_cifar10 = load('mean_cifar10.mat');
mean_cifar10 = mean_cifar10.mean;
mean_cifar10 = permute(mean_cifar10,[3,2,1]);

%sub mean
cifar10_train_data.image = cifar10_train_data.image - single(repmat(mean_cifar10,1,1,1,train_num));
cifar10_test_data.image = cifar10_test_data.image - single(repmat(mean_cifar10,1,1,1,test_num));

minbatch = 256;
iter = solver.iter();
max_iter = solver.max_iter();
test_iterval = 500;
display_iter = 50;
assert(train_num == 50000);
assert(test_num == 10000);
cifar10_train_data.label = reshape(cifar10_train_data.label, [1,1,1,numel(cifar10_train_data.label)]);
cifar10_test_data.label = reshape(cifar10_test_data.label, [1,1,1,numel(cifar10_test_data.label)]);

while (iter < max_iter)
    solver.set_iter(iter);
    inds = randperm(train_num, minbatch);
    data = cifar10_train_data.image(:,:,:,inds);
    label = cifar10_train_data.label(:,:,:,inds);
    solver.net.blobs('data').set_data(data);
    solver.net.blobs('label').set_data(label);
    solver.net.forward_prefilled();
    solver.net.clear_param_diff();
    solver.net.backward_prefilled();
    solver.update();
    iter = iter + 1; 
    if (rem(iter, display_iter) == 1 )
        res = solver.net.forward({data,label});
        fprintf('[Train] | iter : %4d / %4d , loss : %.4f , accuracy : %.2f\n', iter, max_iter, res{2}, res{1});
    end 
    
    if (rem(iter, test_iterval) == 1)
        test_res = cell(2,1);
        test_res{1} = 0; test_res{2} = 0;
        test_iter = 0;
        for index = 1:minbatch:test_num-minbatch+1
            inds = index:index+minbatch-1;
            test_data = cifar10_test_data.image(:,:,:,inds);
            test_label = cifar10_test_data.label(:,:,:,inds);
            res = solver.net.forward( {test_data, test_label} );
            test_res{1} = test_res{1} + res{1};
            test_res{2} = test_res{2} + res{2};
            test_iter = test_iter + 1;
        end
        fprintf('[Test] : loss : %.4f , accuracy : %.2f\n', test_res{1} / test_iter , test_res{1} / test_iter);
    end
end
caffe.reset_all();

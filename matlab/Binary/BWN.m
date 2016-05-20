function BWN()

clear;clc;
current_dir = pwd;
caffe_dir = '../..'; cd(caffe_dir); caffe_dir = pwd;
cd(current_dir);
addpath(fullfile(caffe_dir,'matlab'));
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(5);

rand('seed',0);
cifar10_train_data = load(fullfile(caffe_dir,'examples','cifar10','cifar10_train_lmdb.mat'));
cifar10_train_data.image = permute(cifar10_train_data.image,[3,2,4,1]);
cifar10_test_data = load(fullfile(caffe_dir,'examples','cifar10','cifar10_test_lmdb.mat'));
cifar10_test_data.image = permute(cifar10_test_data.image,[3,2,4,1]);

solver_file = fullfile(caffe_dir,'examples','cifar10','cifar10_full.matlab.solver');
solver = caffe.Solver(solver_file);
caffemodel = fullfile(caffe_dir,'examples','cifar10','cifar10_full_iter_70000.caffemodel');
solver.net.copy_from(caffemodel);

minbatch = 100;

iter = solver.iter();
max_iter = 30000;
test_iterval = 500;
test_iter = 100;
display_iter = 100;
train_num = size(cifar10_train_data.label, 1);
test_num = size(cifar10_test_data.label, 1);
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

    W_Ori = ConverToBinary( solver );

    solver.net.clear_param_diff();
    solver.net.forward_prefilled();
    solver.net.backward_prefilled();
    
    RecoverFromBinary( solver , W_Ori );

    solver.update();
    iter = iter + 1; 

    if (rem(iter, display_iter) == 1 )
        loss = solver.net.blobs('loss').get_data();
        accuracy = solver.net.blobs('accuracy').get_data();
        fprintf('[Train] | iter : %4d / %4d , loss : %.4f , accuracy : %.2f\n', iter, max_iter, loss, accuracy);
    end 
    
    if (rem(iter, test_iterval) == 1)
        W_Ori = ConverToBinary( solver );
        test_res = cell(2,1);
        test_res{1} = 0; test_res{2} = 0;
        for index = 1:test_iter
            inds = randperm(test_num, minbatch);
            test_data = cifar10_test_data.image(:,:,:,inds);
            test_label = cifar10_test_data.label(:,:,:,inds);
            res = solver.net.forward( {test_data, test_label} );
            test_res{1} = test_res{1} + res{1};
            test_res{2} = test_res{2} + res{2};
        end
        fprintf('[Test] : loss : %.4f , accuracy : %.2f\n', test_res{1} / test_iter , test_res{1} / test_iter);
        RecoverFromBinary( solver , W_Ori );
    end
end
caffe.reset_all();

end

function W_Ori = ConverToBinary( solver )
    layers_num = size(solver.net.layer_vec, 2);
    conv_num = 0;
    for i = 1 : layers_num
        if(strcmp(solver.net.layer_vec(i).type, 'Convolution')==1)
            conv_num = conv_num + 1;
            W(conv_num) = {solver.net.layer_vec(i).params(1).get_data()};
            n(conv_num) = size(W{conv_num}, 1) * size(W{conv_num}, 2) * size(W{conv_num}, 3);
            W_estimate = W{conv_num};
            A = sum(sum(sum(abs(W{conv_num})/n(conv_num))));
            W_estimate = repmat(A, size(W_estimate, 1), size(W_estimate, 2), size(W_estimate, 3), 1) .* sign(W_estimate);
            solver.net.layer_vec(i).params(1).set_data(W_estimate);
        end
    end
    W_Ori = W;
end

function RecoverFromBinary( solver , W_Ori )
    layers_num = size(solver.net.layer_vec, 2);
    conv_num = 0;
    for i = 1 : layers_num
        if(strcmp(solver.net.layer_vec(i).type, 'Convolution')==1)
            conv_num = conv_num + 1;
            solver.net.layer_vec(i).params(1).set_data(W_Ori{conv_num});
        end
    end
    assert(conv_num == numel(W_Ori));
end

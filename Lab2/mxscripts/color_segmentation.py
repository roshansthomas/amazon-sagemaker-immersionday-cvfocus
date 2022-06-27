from __future__ import print_function
import mxnet as mx
from mxnet import ndarray as F
from mxnet.io import DataBatch, DataDesc
import os
import numpy as np
import logging
import urllib
import zipfile
import tarfile
import shutil
import gzip
from glob import glob
import random
import json
import argparse

###############################
###     Loss Functions      ###
###############################

def dice_coef(y_true, y_pred):
    intersection = mx.sym.sum(mx.sym.broadcast_mul(y_true, y_pred), axis=(1, 2, 3))
    return mx.sym.broadcast_div((2. * intersection + 1.),(mx.sym.sum(y_true, axis=(1, 2, 3)) + mx.sym.sum(y_pred, axis=(1, 2, 3)) + 1.))

def dice_coef_loss(y_true, y_pred):
    intersection = mx.sym.sum(mx.sym.broadcast_mul(y_true, y_pred), axis=1, )
    return -mx.sym.broadcast_div((2. * intersection + 1.),(mx.sym.broadcast_add(mx.sym.sum(y_true, axis=1), mx.sym.sum(y_pred, axis=1)) + 1.))

###############################
###     UNet Architecture   ###
###############################

def conv_block(inp, num_filter, kernel, pad, block, conv_block):
    conv = mx.sym.Convolution(inp, num_filter=num_filter, kernel=kernel, pad=pad, name='conv%i_%i' % (block, conv_block))
    conv = mx.sym.BatchNorm(conv, fix_gamma=True, name='bn%i_%i' % (block, conv_block))
    conv = mx.sym.Activation(conv, act_type='relu', name='relu%i_%i' % (block, conv_block))
    return conv

def down_block(inp, num_filter, kernel, pad, block, pool=True):
    conv = conv_block(inp, num_filter, kernel, pad, block, 1)
    conv = conv_block(conv, num_filter, kernel, pad, block, 2)
    if pool:
        pool = mx.sym.Pooling(conv, kernel=(2,2), stride=(2,2), pool_type='max', name='pool_%i' % block)
        return pool, conv
    return conv

def down_branch(inp):
    pool1, conv1 = down_block(inp, num_filter=32, kernel=(3,3), pad=(1,1), block=1)
    pool2, conv2 = down_block(pool1, num_filter=64, kernel=(3,3), pad=(1,1), block=2)
    pool3, conv3 = down_block(pool2, num_filter=128, kernel=(3,3), pad=(1,1), block=3)
    pool4, conv4 = down_block(pool3, num_filter=256, kernel=(3,3), pad=(1,1), block=4)
    conv5 = down_block(pool4, num_filter=512, kernel=(3,3), pad=(1,1), block=5, pool=False)
    return [conv5, conv4, conv3, conv2, conv1]

def up_block(inp, down_feature, num_filter, kernel, pad, block):
    trans_conv = mx.sym.Deconvolution(inp, num_filter=num_filter, kernel=(2,2), stride=(2,2), no_bias=True,
                                      name='trans_conv_%i' % block)
    up = mx.sym.concat(*[trans_conv, down_feature], dim=1, name='concat_%i' % block)
    conv = conv_block(up, num_filter, kernel, pad, block, 1)
    conv = conv_block(conv, num_filter, kernel, pad, block, 2)
    return conv

def up_branch(down_features):
    conv6 = up_block(down_features[0], down_features[1], num_filter=256, kernel=(3,3), pad=(1,1), block=6)
    conv7 = up_block(conv6, down_features[2], num_filter=128, kernel=(3,3), pad=(1,1), block=7)
    conv8 = up_block(conv7, down_features[3], num_filter=64, kernel=(3,3), pad=(1,1), block=8)
    conv9 = up_block(conv8, down_features[4], num_filter=64, kernel=(3,3), pad=(1,1), block=9)
    conv10 = mx.sym.Convolution(conv9, num_filter=1, kernel=(1,1), name='conv10_1')
    return conv10

def dice_coef_loss(y_true, y_pred):
    intersection = mx.sym.sum(mx.sym.broadcast_mul(y_true, y_pred), axis=1, )
    return -mx.sym.broadcast_div((2. * intersection + 1.),(mx.sym.broadcast_add(mx.sym.sum(y_true, axis=1), mx.sym.sum(y_pred, axis=1)) + 1.))

def build_unet(inference=False):
    data = mx.sym.Variable(name='data')
    down_features = down_branch(data)
    decoded = up_branch(down_features)
    decoded = mx.sym.sigmoid(decoded, name='softmax')
    if inference:
        return decoded
    else:
        
        net = mx.sym.Flatten(decoded)
        label = mx.sym.Variable(name='label')
        label = mx.sym.Flatten(label, name='label_flatten')
        loss = mx.sym.MakeLoss(dice_coef_loss(label, net), normalization='batch')
        mask_output = mx.sym.BlockGrad(decoded, 'mask')
        out = mx.sym.Group([loss, mask_output])
        return out

###############################
###     Training Script     ###
###############################

def get_data(f_path):
    train_X = np.load(os.path.join(f_path,'train_X_crops.npy'))
    train_Y = np.load(os.path.join(f_path,'train_Y_crops.npy'))
    validation_X = np.load(os.path.join(f_path,'validation_X_crops.npy'))
    validation_Y = np.load(os.path.join(f_path,'validation_Y_crops.npy'))
    return train_X, train_Y, validation_X, validation_Y

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.1)

    #     parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
       
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
#     parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TESTING'])

#     parser.add_argument("--channels", type=str, default=os.environ["SM_CHANNELS"])


    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--burn_in", type=int, default=5)

    return parser.parse_args()

# def train(channel_input_dirs, hyperparameters, hosts, **kwargs
          
def train(
    batch_size,
    epochs,
    learning_rate,
    num_gpus,
    training_channel,
#     testing_channel,
    hosts,
    current_host,
    model_dir,
    beta1,
    beta2,
    burn_in
):
    # retrieve the hyperparameters we set in notebook (with some defaults)
#     batch_size = hyperparameters.get('batch_size', 128)
#     epochs = hyperparameters.get('epochs', 100)
#     learning_rate = hyperparameters.get('learning_rate', 0.1)
#     beta1 = hyperparameters.get('beta1', 0.9)
#     beta2 = hyperparameters.get('beta2', 0.99)
#     num_gpus = hyperparameters.get('num_gpus', 0)
#     burn_in = hyperparameters.get('burn_in', 5)
    # set logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    CHECKPOINTS_DIR = "/opt/ml/checkpoints"
    checkpoints_enabled = os.path.exists(CHECKPOINTS_DIR)

    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'

    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    print (ctx)
    
    print(training_channel)
#     f_path = channel_input_dirs['training']
#     train_X, train_Y, validation_X, validation_Y = get_data(f_path)
    train_X, train_Y, validation_X, validation_Y = get_data(training_channel)

    print ('loaded data')
    
    train_iter = mx.io.NDArrayIter(data = train_X, label=train_Y, batch_size=batch_size, shuffle=True)
    validation_iter = mx.io.NDArrayIter(data = validation_X, label=validation_Y, batch_size=batch_size, shuffle=False)
    data_shape = (batch_size,) + train_X.shape[1:]
    label_shape = (batch_size,) + train_Y.shape[1:]

    print ('created iters')
   
    sym = build_unet()
    net = mx.mod.Module(sym, context=ctx, data_names=('data',), label_names=('label',))
    net.bind(data_shapes=[['data', data_shape]], label_shapes=[['label', label_shape]])
    net.init_params(mx.initializer.Xavier(magnitude=6))
    net.init_optimizer(optimizer = 'adam', 
                               optimizer_params=(
                                   ('learning_rate', learning_rate),
                                   ('beta1', beta1),
                                   ('beta2', beta2)
                              ))
    print ('start training')
    smoothing_constant = .01
    curr_losses = []
    moving_losses = []
    i = 0
    best_val_loss = np.inf
    for e in range(epochs):
        while True:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter.reset()
                break
            net.forward_backward(batch)
            loss = net.get_outputs()[0]
            net.update()
            curr_loss = F.mean(loss).asscalar()
            curr_losses.append(curr_loss)
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                                   else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
            moving_losses.append(moving_loss)
            i += 1
        val_losses = []
        for batch in validation_iter:
            net.forward(batch)
            loss = net.get_outputs()[0]
            val_losses.append(F.mean(loss).asscalar())
        validation_iter.reset()
        # early stopping
        val_loss = np.mean(val_losses)
        print("e:", e)
        print("Burn in:", burn_in)
        print("val_loss:", val_loss)
        print("best_val_loss:", best_val_loss)
#         if e > burn_in and val_loss < best_val_loss:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving the model, params and optimizer state")
#             net.save_checkpoint('best_net', 0)
            net.save_params(os.path.join('/tmp', "best-0000.params"))
#             net.save_checkpoint(CHECKPOINTS_DIR + "/%.4f-unet" % (best_val_loss), e)
            print("Best model at Epoch %i" %(e+1))
        print("Epoch %i: Moving Training Loss %0.5f, Validation Loss %0.5f" % (e+1, moving_loss, val_loss))
    inference_sym = build_unet(inference=True)
    net = mx.mod.Module(inference_sym, context=ctx, data_names=('data',))
    net.bind(data_shapes=[['data', data_shape]])
    net.load_params(os.path.join('/tmp', "best-0000.params"))
#     return net
    
    save(model_dir, net)

def save(model_dir, model):
    print("Saving model in %s" % (model_dir))
    model.symbol.save(os.path.join(model_dir, "model-symbol.json"))
    model.save_params(os.path.join(model_dir, "model-0000.params"))

    signature = [
        {"name": data_desc.name, "shape": [dim for dim in data_desc.shape]}
        for data_desc in model.data_shapes
    ]
    with open(os.path.join(model_dir, "model-shapes.json"), "w") as f:
        json.dump(signature, f)
    
    # verify write
    model_dir_list = os.listdir(model_dir)
    print("Verifying files in model dir")
    for file in model_dir_list:
        print(file)

if __name__ == "__main__":
    args = parse_args()
    num_gpus = int(os.environ["SM_NUM_GPUS"])

    train(
        args.batch_size,
        args.epochs,
        args.learning_rate,
        num_gpus,
        args.train,
#         args.test,
        args.hosts,
        args.current_host,
        args.model_dir,
        args.beta1,
        args.beta2,
        args.burn_in
    )
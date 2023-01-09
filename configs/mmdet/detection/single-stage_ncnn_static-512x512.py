_base_ = ['./single-stage_ncnn_static-416x416.py']

onnx_config = dict(input_shape=[512, 512])

_base_ = ['../_base_/base_tensorrt_static-300x300.py']

onnx_config = dict(input_shape=(512, 512))

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 512, 512],
                    opt_shape=[1, 3, 512, 512],
                    max_shape=[1, 3, 512, 512])))
    ])

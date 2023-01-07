_base_ = ['./detection_openvino_dynamic-608x608.py']

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 512, 512]))])

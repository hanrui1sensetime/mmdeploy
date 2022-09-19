# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import re
from typing import Dict, List, Tuple, Union

import mmengine

from mmdeploy.apis import build_task_processor
from mmdeploy.utils import (Task, get_backend, get_codebase, get_ir_config,
                            get_precision, get_root_logger, get_task_type,
                            is_dynamic_batch, load_config)
from mmdeploy.utils.constants import SDK_TASK_MAP as task_map


def get_mmdpeloy_version() -> str:
    """Return the version of MMDeploy."""
    import mmdeploy
    version = mmdeploy.__version__
    return version


def get_task(deploy_cfg: mmengine.Config) -> Dict:
    """Get the task info for mmdeploy.json. The task info is composed of
    task_name, the codebase name and the codebase version.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.

    Return:
        dict: The task info.
    """
    task_name = get_task_type(deploy_cfg).value
    codebase_name = get_codebase(deploy_cfg).value
    try:
        codebase = importlib.import_module(codebase_name)
    except ModuleNotFoundError:
        logger = get_root_logger()
        logger.warning(f'can not import the module: {codebase_name}')
    codebase_version = codebase.__version__
    return dict(
        task=task_name, codebase=codebase_name, version=codebase_version)


def get_model_name_customs(deploy_cfg: mmengine.Config,
                           model_cfg: mmengine.Config, work_dir: str) -> Tuple:
    """Get the model name and dump custom file.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        work_dir (str): Work dir to save json files.

    Return:
        tuple(): Composed of the model name and the custom info.
    """
    task = get_task_type(deploy_cfg)
    task_processor = build_task_processor(
        model_cfg=model_cfg, deploy_cfg=deploy_cfg, device='cpu')
    name = task_processor.get_model_name()
    customs = []
    if task == Task.TEXT_RECOGNITION:
        customs.append('dict_file.txt')
    return name, customs


def get_models(deploy_cfg: Union[str, mmengine.Config],
               model_cfg: Union[str, mmengine.Config], work_dir: str) -> List:
    """Get the output model informantion for deploy.json.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        work_dir (str): Work dir to save json files.

    Return:
        list[dict]: The list contains dicts composed of the model name, net,
            weghts, backend, precision batchsize and dynamic_shape.
    """
    name, _ = get_model_name_customs(deploy_cfg, model_cfg, work_dir)
    precision = 'FP32'
    ir_name = get_ir_config(deploy_cfg)['save_file']
    weights = ''
    backend = get_backend(deploy_cfg=deploy_cfg).value

    backend_net = dict(
        tensorrt=lambda file: re.sub(r'\.[a-z]+', '.engine', file),
        openvino=lambda file: re.sub(r'\.[a-z]+', '.xml', file),
        ncnn=lambda file: re.sub(r'\.[a-z]+', '.param', file),
        snpe=lambda file: re.sub(r'\.[a-z]+', '.dlc', file))
    backend_weights = dict(
        pplnn=lambda file: re.sub(r'\.[a-z]+', '.json', file),
        openvino=lambda file: re.sub(r'\.[a-z]+', '.bin', file),
        ncnn=lambda file: re.sub(r'\.[a-z]+', '.bin', file))
    net = backend_net.get(backend, lambda x: x)(ir_name)
    weights = backend_weights.get(backend, lambda x: weights)(ir_name)

    precision = get_precision(deploy_cfg)
    dynamic_shape = is_dynamic_batch(deploy_cfg, input_name='input')
    return [
        dict(
            name=name,
            net=net,
            weights=weights,
            backend=backend,
            precision=precision,
            batch_size=1,
            dynamic_shape=dynamic_shape)
    ]


def get_inference_info(deploy_cfg: mmengine.Config, model_cfg: mmengine.Config,
                       work_dir: str) -> Dict:
    """Get the inference information for pipeline.json.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        work_dir (str): Work dir to save json files.

    Return:
        dict: Composed of the model name, type, module, input, output and
            input_map.
    """
    name, _ = get_model_name_customs(deploy_cfg, model_cfg, work_dir)
    ir_config = get_ir_config(deploy_cfg)
    input_names = ir_config.get('input_names', None)
    input_name = input_names[0] if input_names else 'input'
    input_map = dict(img=input_name)
    return_dict = dict(
        name=name,
        type='Task',
        module='Net',
        input=['prep_output'],
        output=['infer_output'],
        input_map=input_map)
    if 'use_vulkan' in deploy_cfg['backend_config']:
        return_dict['use_vulkan'] = deploy_cfg['backend_config']['use_vulkan']
    return return_dict


def get_preprocess(deploy_cfg: mmengine.Config, model_cfg: mmengine.Config):
    task_processor = build_task_processor(
        model_cfg=model_cfg, deploy_cfg=deploy_cfg, device='cpu')
    pipeline = task_processor.get_preprocess()
    meta_keys = [
        'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
        'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'valid_ratio'
    ]
    if 'transforms' in pipeline[-1]:
        transforms = pipeline[-1]['transforms']
        transforms.insert(0, pipeline[0])
        for transform in transforms:
            if transform['type'] == 'Resize':
                transform['size'] = pipeline[-1].img_scale[::-1]
                if 'img_scale' in transform:
                    transform.pop('img_scale')
    else:
        pipeline = [
            item for item in pipeline if item['type'] != 'MultiScaleFilpAug'
        ]
        transforms = pipeline
    transforms = [
        item for item in transforms if 'Random' not in item['type']
        and 'RescaleToZeroOne' not in item['type']
    ]
    if model_cfg.default_scope == 'mmedit':
        transforms.insert(1, model_cfg.model.data_preprocessor)
    for i, transform in enumerate(transforms):
        if 'keys' in transform and transform['keys'] == ['lq']:
            transform['keys'] = ['img']
        if 'key' in transform and transform['key'] == 'lq':
            transform['key'] = 'img'
        if transform['type'] == 'ToTensor':
            transform['type'] = 'ImageToTensor'
        if transform['type'] == 'EditDataPreprocessor':
            transform['type'] = 'Normalize'
        if transform['type'] == 'PackTextDetInputs':
            meta_keys += transform[
                'meta_keys'] if 'meta_keys' in transform else []
            transform['meta_keys'] = list(set(meta_keys))
            transforms[i]['type'] = 'Collect'
        if transform['type'] == 'PackEditInputs':
            meta_keys += transform[
                'meta_keys'] if 'meta_keys' in transform else []
            transform['meta_keys'] = list(set(meta_keys))
            transform['keys'] = ['img']
            transforms[i]['type'] = 'Collect'
    assert transforms[0]['type'] == 'LoadImageFromFile', 'The first item type'\
        ' of pipeline should be LoadImageFromFile'

    return dict(
        type='Task',
        module='Transform',
        name='Preprocess',
        input=['img'],
        output=['prep_output'],
        transforms=transforms)


def get_postprocess(deploy_cfg: mmengine.Config, model_cfg: mmengine.Config,
                    work_dir: str) -> Dict:
    """Get the post process information for pipeline.json.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        work_dir (str): Work dir to save json files.

    Return:
        dict: Composed of the model name, type, module, input, params and
            output.
    """
    task_processor = build_task_processor(
        model_cfg=model_cfg, deploy_cfg=deploy_cfg, device='cpu')
    post_processor = task_processor.get_postprocess(work_dir)

    return dict(
        type='Task',
        module=get_codebase(deploy_cfg).value,
        name='postprocess',
        component=post_processor['type'],
        params=post_processor.get('params', dict()),
        output=['post_output'])


def get_deploy(deploy_cfg: mmengine.Config, model_cfg: mmengine.Config,
               work_dir: str) -> Dict:
    """Get the inference information for pipeline.json.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        work_dir (str): Work dir to save json files.

    Return:
        dict: Composed of version, task, models and customs.
    """

    task = get_task_type(deploy_cfg)
    cls_name = task_map[task]['cls_name']
    _, customs = get_model_name_customs(
        deploy_cfg, model_cfg, work_dir=work_dir)
    version = get_mmdpeloy_version()
    models = get_models(deploy_cfg, model_cfg, work_dir=work_dir)
    return dict(version=version, task=cls_name, models=models, customs=customs)


def get_pipeline(deploy_cfg: mmengine.Config, model_cfg: mmengine.Config,
                 work_dir: str) -> Dict:
    """Get the inference information for pipeline.json.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        work_dir (str): Work dir to save json files.

    Return:
        dict: Composed of input node name, output node name and the tasks.
    """
    preprocess = get_preprocess(deploy_cfg, model_cfg)
    infer_info = get_inference_info(deploy_cfg, model_cfg, work_dir=work_dir)
    postprocess = get_postprocess(deploy_cfg, model_cfg, work_dir)
    task = get_task_type(deploy_cfg)
    input_names = preprocess['input']
    output_names = postprocess['output']
    if task == Task.CLASSIFICATION or task == Task.SUPER_RESOLUTION:
        postprocess['input'] = infer_info['output']
    else:
        postprocess['input'] = preprocess['output'] + infer_info['output']

    return dict(
        pipeline=dict(
            input=input_names,
            output=output_names,
            tasks=[preprocess, infer_info, postprocess]))


def get_detail(deploy_cfg: mmengine.Config, model_cfg: mmengine.Config,
               pth: str) -> Dict:
    """Get the detail information for detail.json.

    Args:
        deploy_cfg (mmengine.Config): Deploy config dict.
        model_cfg (mmengine.Config): The model config dict.
        pth (str): The checkpoint weight of pytorch model.

    Return:
        dict: Composed of version, codebase, codebase_config, onnx_config,
            backend_config and calib_config.
    """
    version = get_mmdpeloy_version()
    codebase = get_task(deploy_cfg)
    codebase['pth'] = pth
    codebase['config'] = model_cfg.filename
    codebase_config = deploy_cfg.get('codebase_config', dict())
    ir_config = get_ir_config(deploy_cfg)
    backend_config = deploy_cfg.get('backend_config', dict())
    calib_config = deploy_cfg.get('calib_config', dict())
    return dict(
        version=version,
        codebase=codebase,
        codebase_config=codebase_config,
        onnx_config=ir_config,
        backend_config=backend_config,
        calib_config=calib_config)


def export2SDK(deploy_cfg: Union[str, mmengine.Config],
               model_cfg: Union[str,
                                mmengine.Config], work_dir: str, pth: str):
    """Export information to SDK. This function dump `deploy.json`,
    `pipeline.json` and `detail.json` to work dir.

    Args:
        deploy_cfg (str | mmengine.Config): Deploy config file or dict.
        model_cfg (str | mmengine.Config): Model config file or dict.
        work_dir (str): Work dir to save json files.
        pth (str): The path of the model checkpoint weights.
    """
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    deploy_info = get_deploy(deploy_cfg, model_cfg, work_dir=work_dir)
    pipeline_info = get_pipeline(deploy_cfg, model_cfg, work_dir=work_dir)
    detail_info = get_detail(deploy_cfg, model_cfg, pth=pth)
    mmengine.dump(
        deploy_info,
        '{}/deploy.json'.format(work_dir),
        sort_keys=False,
        indent=4)
    mmengine.dump(
        pipeline_info,
        '{}/pipeline.json'.format(work_dir),
        sort_keys=False,
        indent=4)
    mmengine.dump(
        detail_info,
        '{}/detail.json'.format(work_dir),
        sort_keys=False,
        indent=4)

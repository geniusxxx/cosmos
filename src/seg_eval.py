import os
import sys
import argparse
import training.clip_segmentor
import training.custom_datasets
from training.params import parse_args

from mmengine.config import Config
from mmengine.runner import Runner


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main(args):
    args = parse_args(args)
    if args.seg_w_background: # with background
        conf_files = ['cfg_voc21.py', 'cfg_context60.py', 'cfg_coco_object.py']
    else:
        conf_files = ['cfg_voc20.py', 'cfg_city_scapes.py', 'cfg_context59.py', 'cfg_ade20k.py', 'cfg_coco_stuff164k.py']

    for conf_f in conf_files:
        cfg = Config.fromfile(f'./training/seg_configs/{conf_f}')
        cfg.launcher = 'none'
        cfg.work_dir = './work_logs/'

        openclip_args = {}
        for arg in vars(args):
            openclip_args[arg] = getattr(args, arg)
        cfg.model.update(openclip_args)

        runner = Runner.from_cfg(cfg)
        runner.test()

    
if __name__ == "__main__":
    main(sys.argv[1:])
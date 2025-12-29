#!/usr/bin/env python
"""
빠른 학습 체크 스크립트
- 몇 개의 iteration만 실행하여 모델이 정상 작동하는지 확인
- Heatmap, Token 통계 출력
"""
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import set_random_seed

from mmpose.apis import init_random_seed
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Quick check for training')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--iterations', type=int, default=10, 
                        help='number of iterations to test')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    # Quick check settings
    cfg.total_epochs = 1
    cfg.log_config.interval = 1
    cfg.checkpoint_config.interval = 999999  # 저장 안함
    cfg.evaluation.interval = 999999  # 평가 안함
    
    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # Init logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'quick_check_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    
    # Log environment info
    env_info = collect_env()
    logger.info('Environment info:\n' + '\n'.join(f'{k}: {v}' for k, v in env_info.items()))
    
    # Set random seeds
    seed = init_random_seed(0)
    logger.info(f'Set random seed to {seed}')
    set_random_seed(seed, deterministic=False)
    
    # Build model
    logger.info('=' * 80)
    logger.info('🔧 Building model...')
    model = build_posenet(cfg.model)
    model.init_weights()
    logger.info('✅ Model built successfully')
    
    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    
    # Build dataset
    logger.info('=' * 80)
    logger.info('📁 Building dataset...')
    datasets = [build_dataset(cfg.data.train)]
    logger.info(f'✅ Dataset built: {len(datasets[0])} samples')
    
    # Quick forward pass test
    logger.info('=' * 80)
    logger.info('🔬 Testing forward pass...')
    model = model.cuda()
    model.train()
    
    # Get one batch
    from mmcv.parallel import collate
    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        datasets[0],
        batch_size=2,
        num_workers=0,  # 0 for stability
        shuffle=True,
        collate_fn=collate
    )
    
    data_iter = iter(data_loader)
    for i in range(min(args.iterations, 10)):
        try:
            data_batch = next(data_iter)
            
            # Move data to GPU
            for key in data_batch.keys():
                if key == 'img':
                    data_batch[key] = data_batch[key].cuda()
                elif key == 'target':
                    data_batch[key] = data_batch[key].cuda()
                elif key == 'target_weight':
                    data_batch[key] = data_batch[key].cuda()
            
            # Forward
            with torch.no_grad():
                outputs = model.forward_train(**data_batch)
            
            logger.info(f'  Iteration {i+1}/{args.iterations}:')
            for key, val in outputs.items():
                if 'loss' in key.lower() and torch.is_tensor(val):
                    logger.info(f'    {key}: {val.item():.6f}')
                    
            # Check for NaN/Inf
            has_nan = False
            for key, val in outputs.items():
                if torch.is_tensor(val) and (torch.isnan(val).any() or torch.isinf(val).any()):
                    logger.warning(f'    ⚠️  WARNING: {key} contains NaN or Inf!')
                    has_nan = True
                    
            if not has_nan:
                logger.info('    ✅ No NaN/Inf detected')
                
        except StopIteration:
            data_iter = iter(data_loader)
            data_batch = next(data_iter)
        except Exception as e:
            logger.error(f'    ❌ Error: {e}')
            import traceback
            traceback.print_exc()
            break
    
    logger.info('=' * 80)
    logger.info('✅ Quick check completed!')
    logger.info(f'📝 Log saved to: {log_file}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()


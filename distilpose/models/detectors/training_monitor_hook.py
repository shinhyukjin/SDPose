# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.runner import HOOKS, Hook
import warnings


@HOOKS.register_module()
class TrainingMonitorHook(Hook):
    """학습 중 이상 징후를 감지하는 Hook
    
    주요 기능:
    1. Loss 값의 급격한 변화 감지
    2. Heatmap 출력의 이상 감지 (NaN, Inf, 극단값)
    3. Token의 통계 추적
    4. 성능 하락 감지
    
    Args:
        check_interval (int): 체크 주기 (iteration). Default: 50
        loss_spike_threshold (float): Loss 급증 감지 임계값. Default: 2.0
        performance_drop_threshold (float): 성능 하락 감지 임계값. Default: 0.05
        save_stats (bool): 통계를 파일로 저장할지 여부. Default: True
    """
    
    def __init__(self, 
                 check_interval=50,
                 loss_spike_threshold=2.0,
                 performance_drop_threshold=0.05,
                 save_stats=True):
        self.check_interval = check_interval
        self.loss_spike_threshold = loss_spike_threshold
        self.performance_drop_threshold = performance_drop_threshold
        self.save_stats = save_stats
        
        # 통계 저장
        self.loss_history = []
        self.heatmap_stats_history = []
        self.token_stats_history = []
        self.best_ap = 0.0
        
    def before_run(self, runner):
        """학습 시작 전 초기화"""
        runner.logger.info("=" * 80)
        runner.logger.info("🔍 TrainingMonitorHook activated")
        runner.logger.info(f"  - Check interval: {self.check_interval} iterations")
        runner.logger.info(f"  - Loss spike threshold: {self.loss_spike_threshold}x")
        runner.logger.info(f"  - Performance drop threshold: {self.performance_drop_threshold}")
        runner.logger.info("=" * 80)
        
    def after_train_iter(self, runner):
        """각 iteration 후 체크"""
        if not self.every_n_iters(runner, self.check_interval):
            return
            
        # Loss 체크
        if 'loss' in runner.log_buffer.output:
            current_loss = runner.log_buffer.output['loss']
            self._check_loss(runner, current_loss)
            
    def after_train_epoch(self, runner):
        """Epoch 종료 후 체크"""
        epoch = runner.epoch + 1
        runner.logger.info("")
        runner.logger.info("=" * 80)
        runner.logger.info(f"📊 Epoch {epoch} Training Summary")
        runner.logger.info("=" * 80)
        
        # Loss summary
        if len(self.loss_history) > 0:
            recent_losses = self.loss_history[-10:]
            avg_loss = np.mean(recent_losses)
            std_loss = np.std(recent_losses)
            runner.logger.info(f"  Loss - Recent Avg: {avg_loss:.4f} ± {std_loss:.4f}")
            runner.logger.info(f"       - Min: {min(recent_losses):.4f}, Max: {max(recent_losses):.4f}")
            
        runner.logger.info("=" * 80)
        runner.logger.info("")
        
    def after_val_epoch(self, runner):
        """Validation 후 성능 체크"""
        if hasattr(runner, 'eval_res') and runner.eval_res is not None:
            if 'AP' in runner.eval_res:
                current_ap = runner.eval_res['AP']
                self._check_performance(runner, current_ap)
                
    def _check_loss(self, runner, current_loss):
        """Loss 이상 감지"""
        # NaN/Inf 체크
        if np.isnan(current_loss) or np.isinf(current_loss):
            runner.logger.warning("⚠️  WARNING: Loss is NaN or Inf!")
            runner.logger.warning(f"   Iteration: {runner.iter}")
            runner.logger.warning(f"   Loss value: {current_loss}")
            return
            
        # Loss 급증 체크
        if len(self.loss_history) > 5:
            recent_avg = np.mean(self.loss_history[-5:])
            if current_loss > recent_avg * self.loss_spike_threshold:
                runner.logger.warning("⚠️  WARNING: Loss spike detected!")
                runner.logger.warning(f"   Current: {current_loss:.4f}")
                runner.logger.warning(f"   Recent avg: {recent_avg:.4f}")
                runner.logger.warning(f"   Ratio: {current_loss/recent_avg:.2f}x")
                
        self.loss_history.append(current_loss)
        
        # 너무 많이 쌓이면 오래된 것 제거
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]
            
    def _check_performance(self, runner, current_ap):
        """성능 하락 체크"""
        if current_ap > self.best_ap:
            improvement = current_ap - self.best_ap
            runner.logger.info(f"🎉 New Best AP: {current_ap:.4f} (+{improvement:.4f})")
            self.best_ap = current_ap
        elif self.best_ap > 0 and (self.best_ap - current_ap) > self.performance_drop_threshold:
            drop = self.best_ap - current_ap
            runner.logger.warning("⚠️  WARNING: Performance drop detected!")
            runner.logger.warning(f"   Current AP: {current_ap:.4f}")
            runner.logger.warning(f"   Best AP: {self.best_ap:.4f}")
            runner.logger.warning(f"   Drop: {drop:.4f}")
            
    def after_run(self, runner):
        """학습 종료 후 최종 통계"""
        if not self.save_stats:
            return
            
        import os
        stats_file = os.path.join(runner.work_dir, 'training_stats.txt')
        
        with open(stats_file, 'w') as f:
            f.write("Training Statistics\n")
            f.write("=" * 80 + "\n\n")
            
            if len(self.loss_history) > 0:
                f.write("Loss Statistics:\n")
                f.write(f"  - Mean: {np.mean(self.loss_history):.4f}\n")
                f.write(f"  - Std: {np.std(self.loss_history):.4f}\n")
                f.write(f"  - Min: {min(self.loss_history):.4f}\n")
                f.write(f"  - Max: {max(self.loss_history):.4f}\n")
                f.write(f"  - Final: {self.loss_history[-1]:.4f}\n\n")
                
            f.write(f"Best AP: {self.best_ap:.4f}\n")
            
        runner.logger.info(f"📝 Training statistics saved to: {stats_file}")


@HOOKS.register_module()
class DetailedLossLogHook(Hook):
    """각 Loss 항목을 자세히 로깅하는 Hook
    
    Args:
        log_interval (int): 로깅 주기. Default: 10
    """
    
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        
    def after_train_iter(self, runner):
        """각 iteration 후 상세 loss 로깅"""
        if not self.every_n_iters(runner, self.log_interval):
            return
            
        # Log buffer에서 모든 loss 가져오기
        log_items = {}
        for key, val in runner.log_buffer.output.items():
            if 'loss' in key.lower():
                log_items[key] = val
                
        if len(log_items) > 0:
            log_str = f"Iter [{runner.iter}] "
            for key, val in log_items.items():
                if isinstance(val, (int, float)):
                    log_str += f"{key}: {val:.6f}, "
            runner.logger.info(log_str.rstrip(', '))


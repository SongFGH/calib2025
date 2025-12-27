#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
适配连续值的Platt校准（Platt Scaling for Continuous Targets）
目标值为0到1之间的连续值，而非离散的0/1标签
"""

import numpy as np
import logging
from scipy.optimize import minimize
from scipy.special import expit  # 数值稳定的sigmoid函数
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 自动查找支持中文的字体
def get_chinese_font():
    fonts = fm.fontManager.ttflist
    chinese_font_names = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'Source Han Sans', 'WenQuanYi']
    for font in fonts:
        for name in chinese_font_names:
            if name in font.name or name in font.fname:
                return font.name
    return fm.FontProperties().get_name()

# 设置全局字体
chinese_font = get_chinese_font()
plt.rcParams['font.sans-serif'] = [chinese_font]
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlattCalibrator:
    """
    适配连续值的Platt校准器（目标值为0-1之间的连续值）
    """
    
    def __init__(self, max_iter=100, tol=1e-9, loss_type='mse', verbose=True):
        """
        初始化校准器
        
        参数:
        max_iter: 最大迭代次数
        tol: 收敛阈值
        loss_type: 损失函数类型，可选'mse'（均方误差，推荐）或'log_loss'（连续对数损失）
        verbose: 是否输出详细日志
        """
        self.max_iter = max_iter
        self.tol = tol
        self.loss_type = loss_type  # 新增：选择损失函数类型
        self.verbose = verbose
        self.A = 0.0
        self.B = 0.0
        self.fitted = False
        
        # 校验损失函数类型
        if self.loss_type not in ['mse', 'log_loss']:
            raise ValueError("loss_type只能是'mse'或'log_loss'")
    
    def fit(self, scores, targets):
        """
        训练校准模型（适配0-1连续目标值）
        
        参数:
        scores: 模型原始输出分数（任意数值）
        targets: 连续目标值（0到1之间）
        """
        if self.verbose:
            # logger.info("开始拟合连续值Platt校准模型...")
            print("开始拟合连续值Platt校准模型...")
        
        # 输入验证与转换
        scores = np.array(scores, dtype=float)
        targets = np.array(targets, dtype=float)
        
        if len(scores) != len(targets):
            raise ValueError("scores和targets的长度必须相同")
        
        # 验证targets是否在0-1之间
        if np.min(targets) < 0 or np.max(targets) > 1:
            raise ValueError("targets必须是0到1之间的连续值")
        
        # 初始化参数（沿用Platt的先验初始化思路，适配连续值）
        avg_target = np.mean(targets)
        A = 0.0
        # 初始化B，使初始sigmoid输出接近目标均值
        B = np.log((1 - avg_target + 1e-10) / (avg_target + 1e-10))  # 防止除零
        
        # --------------------------
        # 核心修改：定义适配连续值的损失函数和梯度
        # --------------------------
        def objective(params):
            """目标函数（损失函数）"""
            a, b = params
            logits = a * scores + b
            probs = expit(np.clip(logits, -20, 20))  # 保持sigmoid映射
            
            if self.loss_type == 'mse':
                # 均方误差（适配连续值，最稳定）
                loss = np.mean((probs - targets) ** 2)
            else:  # log_loss（连续对数损失，需保证targets不接近0/1，否则需加平滑）
                # 连续对数损失：-E[ y*log(p) + (1-y)*log(1-p) ]
                loss = -np.mean(targets * np.log(probs + 1e-10) + (1 - targets) * np.log(1 - probs + 1e-10))
            return loss
        
        def gradient(params):
            """梯度函数（对应损失函数的一阶导数）"""
            a, b = params
            logits = a * scores + b
            probs = expit(np.clip(logits, -20, 20))
            n = len(scores)
            
            if self.loss_type == 'mse':
                # MSE的梯度：dL/da = 2/n * sum( (p-y)*p*(1-p)*scores )
                # dL/db = 2/n * sum( (p-y)*p*(1-p) )
                # 注：p*(1-p)是sigmoid的导数（dP/dlogits）
                d_logits = 2 * (probs - targets) * probs * (1 - probs) / n
            else:  # log_loss的梯度（与原始Platt一致，因为对数损失的梯度形式对连续y依然成立）
                d_logits = (probs - targets) / n  # 平均梯度（原始是总和，这里改为平均，优化更稳定）
            
            d_a = np.sum(d_logits * scores)
            d_b = np.sum(d_logits)
            return np.array([d_a, d_b])
        
        # 优化参数（使用L-BFGS-B算法，与原始Platt一致）
        initial_params = [A, B]
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            # jac=gradient,
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )
        
        # 更新参数
        self.A, self.B = result.x
        self.fitted = True
        
        # 输出拟合信息
        if self.verbose:
            # logger.info(f"校准模型拟合完成，优化状态: {result.success}")
            print(f"校准模型拟合完成，优化状态: {result.success}")
            # logger.info(f"迭代次数: {result.nit}，最终参数: A={self.A:.6f}, B={self.B:.6f}")
            print(f"迭代次数: {result.nit}，最终参数: A={self.A:.6f}, B={self.B:.6f}")

            before_calib_loss = mean_squared_error(targets, scores) * 1e+5
            print(f"校准前的MSE损失: {before_calib_loss:.6f}")

            # 计算校准后的损失
            calibrated_probs = self.predict(scores)

            if self.loss_type == 'mse':
                loss = mean_squared_error(targets, calibrated_probs) * 1e+5
                # logger.info(f"校准后的MSE损失: {loss:.6f}")
                print(f"校准后的MSE损失: {loss:.6f}")
            else:
                loss = -np.mean(targets * np.log(calibrated_probs + 1e-10) + (1 - targets) * np.log(1 - calibrated_probs + 1e-10))
                # logger.info(f"校准后的对数损失: {loss:.6f}")
                print(f"校准后的对数损失: {loss:.6f}")
        
        return self
    
    def predict(self, scores):
        """预测校准后的连续值（0-1之间）"""
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        scores = np.array(scores, dtype=float)
        logits = self.A * scores + self.B
        probs = expit(np.clip(logits, -20, 20))  # 保持0-1范围
        return probs
    
    def get_params(self):
        """获取模型参数"""
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        return {'A': self.A, 'B': self.B}
    
    def visualize_calibration(self, scores, targets, bins=10):
        """可视化连续值校准效果"""
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        scores = np.array(scores)
        targets = np.array(targets)
        calibrated_probs = self.predict(scores)
        
        # 归一化原始分数（便于可视化）
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
        # 分位数分桶（等频）
        bin_edges = np.quantile(scores_normalized, np.linspace(0, 1, bins + 1))
        bin_edges[-1] += 1e-10
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 计算每个桶的均值
        orig_means = []
        target_means = []
        cal_means = []
        bin_sizes = []
        
        for i in range(bins):
            mask = (scores_normalized >= bin_edges[i]) & (scores_normalized < bin_edges[i+1])
            if np.sum(mask) > 0:
                orig_means.append(np.mean(scores_normalized[mask]))
                target_means.append(np.mean(targets[mask]))
                cal_means.append(np.mean(calibrated_probs[mask]))
                bin_sizes.append(np.sum(mask))
            else:
                orig_means.append(np.nan)
                target_means.append(np.nan)
                cal_means.append(np.nan)
                bin_sizes.append(0)
        
        # 绘制可视化图
        plt.figure(figsize=(12, 6))
        
        # 校准曲线：预测值 vs 真实值
        plt.subplot(1, 2, 1)
        plt.scatter(orig_means, target_means, s=bin_sizes, alpha=0.5, c='blue', label='原始分数（归一化）')
        plt.scatter(cal_means, target_means, s=bin_sizes, alpha=0.5, c='red', label='校准后值')
        plt.plot([0, 1], [0, 1], 'k--', label='完美校准')
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Target Value')
        plt.title('Calibration Curve (Continuous Targets)')
        plt.grid(True)
        plt.legend()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # 分布对比
        plt.subplot(1, 2, 2)
        plt.hist(scores_normalized, bins=50, alpha=0.5, label='原始分数分布（归一化）')
        plt.hist(calibrated_probs, bins=50, alpha=0.5, label='校准后值分布')
        plt.hist(targets, bins=50, alpha=0.3, color='green', label='真实目标值分布')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution Comparison')
        plt.grid(True)
        plt.legend()
        plt.xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig('continuous_platt_calibration.png', dpi=300)
        plt.close()
        logger.info("校准可视化已保存为 continuous_platt_calibration.png")

# --------------------------
# 新增：连续值的评估指标函数
# --------------------------
def evaluate_continuous_calibration(preds, targets):
    """
    评估连续值校准效果的指标
    返回：MSE、MAE、R²
    """
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

# --------------------------
# 示例：使用连续值测试Platt校准
# --------------------------
def example_usage():
    logger.info("=== 连续值Platt校准示例 ===")
    np.random.seed(42)  # 固定随机种子，结果可复现
    
    # 生成模拟数据
    n_samples = 20
    # 原始分数（例如模型输出的未校准分数）
    raw_scores = np.random.beta(2, 5, n_samples)  # 右偏分布
    # 生成0-1之间的连续目标值（模拟真实的连续标签）
    # 目标值与原始分数呈非线性关系（模拟需要校准的场景）
    noise = np.random.normal(0, 0.05, n_samples)  # 加入少量噪声
    true_targets = 1 / (1 + np.exp(-8 * (raw_scores - 0.4))) + noise
    true_targets = np.clip(true_targets, 0, 1)  # 确保在0-1之间
    
    # 初始化校准器（使用MSE损失，推荐）
    calibrator = PlattCalibrator(loss_type='mse', verbose=True)
    calibrator.fit(raw_scores, true_targets)
    
    # 预测校准后的值
    calibrated_values = calibrator.predict(raw_scores)
    
    # 评估校准效果
    eval_before = evaluate_continuous_calibration(raw_scores, true_targets)
    eval_after = evaluate_continuous_calibration(calibrated_values, true_targets)
    
    logger.info("=== 校准前评估指标 ===")
    logger.info(f"MSE: {eval_before['mse']:.6f}, MAE: {eval_before['mae']:.6f}, R²: {eval_before['r2']:.6f}")
    logger.info("=== 校准后评估指标 ===")
    logger.info(f"MSE: {eval_after['mse']:.6f}, MAE: {eval_after['mae']:.6f}, R²: {eval_after['r2']:.6f}")
    
    # 可视化校准效果
    calibrator.visualize_calibration(raw_scores, true_targets)

if __name__ == "__main__":
    example_usage()
    logger.info("连续值Platt校准演示完成！")

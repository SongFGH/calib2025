#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import logging
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 复用原有的字体设置和日志配置（略，与原代码一致）
def get_chinese_font():
    fonts = fm.fontManager.ttflist
    chinese_font_names = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'Source Han Sans', 'WenQuanYi']
    for font in fonts:
        for name in chinese_font_names:
            if name in font.name or name in font.fname:
                return font.name
    return fm.FontProperties().get_name()

chinese_font = get_chinese_font()
plt.rcParams['font.sans-serif'] = [chinese_font]
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlattCalibrator:
    """
    Beta校准器（适配0-1连续目标值，替代Platt的sigmoid）
    核心：用Beta分布的均值作为校准值，参数由原始分数线性映射得到
    """
    def __init__(self, max_iter=200, tol=1e-9, verbose=True):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        # Beta分布的参数映射系数（4个参数：A,B对应alpha；C,D对应beta）
        self.A = 1.0
        self.B = 0.0
        self.C = 1.0
        self.D = 0.0
        self.fitted = False
        self.scores_min = None
        self.scores_max = None

    def fit(self, scores, targets):
        if self.verbose:
            logger.info("开始拟合Beta校准模型...")
        
        scores = np.array(scores, dtype=float)
        targets = np.array(targets, dtype=float)
        
        if len(scores) != len(targets):
            raise ValueError("scores和targets的长度必须相同")
        if np.min(targets) < 0 or np.max(targets) > 1:
            raise ValueError("targets必须是0到1之间的连续值")
        
        self.scores_min = scores.min()
        self.scores_max = scores.max()
        
        # 初始化参数：让Beta分布的均值接近目标均值
        avg_target = np.mean(targets)
        # 初始alpha=2, beta=2（均匀分布），映射到分数的线性关系
        initial_params = [1.0, np.log(2) - 1.0 * np.mean(scores), 1.0, np.log(2) - 1.0 * np.mean(scores)]

        def objective(params):
            """目标函数：MSE损失"""
            a, b, c, d = params
            # 映射为Beta分布的形状参数（必须为正，用exp保证）
            alpha = np.exp(a * scores + b)
            beta = np.exp(c * scores + d)
            # Beta分布的均值（校准值，落在0-1区间）
            probs = alpha / (alpha + beta)
            # MSE损失
            loss = np.mean((probs - targets) ** 2)
            return loss

        def gradient(params):
            """梯度函数（MSE损失的一阶导数）"""
            a, b, c, d = params
            alpha = np.exp(a * scores + b)
            beta = np.exp(c * scores + d)
            probs = alpha / (alpha + beta)
            n = len(scores)

            # 链式法则计算梯度
            d_loss = 2 * (probs - targets) / n
            # d_probs/d_alpha = beta/(alpha+beta)^2, d_probs/d_beta = -alpha/(alpha+beta)^2
            d_probs_d_alpha = beta / (alpha + beta) ** 2
            d_probs_d_beta = -alpha / (alpha + beta) ** 2
            # d_alpha/da = scores*alpha, d_alpha/db = alpha; d_beta/dc = scores*beta, d_beta/dd = beta
            d_a = np.sum(d_loss * d_probs_d_alpha * scores * alpha)
            d_b = np.sum(d_loss * d_probs_d_alpha * alpha)
            d_c = np.sum(d_loss * d_probs_d_beta * scores * beta)
            d_d = np.sum(d_loss * d_probs_d_beta * beta)

            return np.array([d_a, d_b, d_c, d_d])

        # 优化参数（L-BFGS-B）
        result = minimize(
            objective,
            initial_params,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )

        self.A, self.B, self.C, self.D = result.x
        self.fitted = True

        if self.verbose:
            logger.info(f"Beta校准模型拟合完成，优化状态: {result.success}")
            logger.info(f"迭代次数: {result.nit}，最终参数: A={self.A:.6f}, B={self.B:.6f}, C={self.C:.6f}, D={self.D:.6f}")
            # 计算校准前后的损失
            scores_normalized = (scores - self.scores_min) / (self.scores_max - self.scores_min + 1e-10)
            before_calib_mse = mean_squared_error(targets, scores_normalized)
            calibrated_probs = self.predict(scores)
            after_calib_mse = mean_squared_error(targets, calibrated_probs)
            logger.info(f"校准前的MSE损失（归一化后）: {before_calib_mse:.6f}")
            logger.info(f"校准后的MSE损失: {after_calib_mse:.6f}")

        return self

    def predict(self, scores):
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        scores = np.array(scores, dtype=float)
        alpha = np.exp(self.A * scores + self.B)
        beta = np.exp(self.C * scores + self.D)
        probs = alpha / (alpha + beta)
        return np.clip(probs, 0, 1)  # 防止数值误差导致超出0-1

    # 复用原有的可视化和评估方法（略，与原代码一致）
    def _normalize_scores(self, scores):
        return (scores - self.scores_min) / (self.scores_max - self.scores_min + 1e-10)

    def visualize_calibration(self, scores, targets, bins=10):
        # 与原代码的visualize_calibration方法完全一致，此处省略（可直接复制原代码）
        pass

# 评估函数（复用原代码）
def evaluate_continuous_calibration(preds, targets):
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    return {'mse': mse, 'mae': mae, 'r2': r2}

# 示例使用
def example_usage():
    logger.info("=== 0-1连续目标值的Beta校准示例 ===")
    np.random.seed(42)
    n_samples = 1000
    # 生成原始分数（右偏分布）
    raw_scores = np.random.beta(2, 5, n_samples) * 10
    # 生成0-1连续目标值（非线性+偏态）
    noise = np.random.normal(0, 0.05, n_samples)
    true_targets = 1 / (1 + np.exp(-0.8 * (raw_scores - 4))) + noise
    true_targets = np.clip(true_targets, 0, 1)

    # 初始化Beta校准器
    calibrator = PlattCalibrator(verbose=True)
    calibrator.fit(raw_scores, true_targets)
    calibrated_values = calibrator.predict(raw_scores)

    # 评估
    scores_normalized = calibrator._normalize_scores(raw_scores)
    eval_before = evaluate_continuous_calibration(scores_normalized, true_targets)
    eval_after = evaluate_continuous_calibration(calibrated_values, true_targets)

    logger.info("=== 校准前评估指标（归一化后） ===")
    logger.info(f"MSE: {eval_before['mse']:.6f}, MAE: {eval_before['mae']:.6f}, R²: {eval_before['r2']:.6f}")
    logger.info("=== 校准后评估指标 ===")
    logger.info(f"MSE: {eval_after['mse']:.6f}, MAE: {eval_after['mae']:.6f}, R²: {eval_after['r2']:.6f}")

if __name__ == "__main__":
    example_usage()

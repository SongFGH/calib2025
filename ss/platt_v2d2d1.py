#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门适配10^-7量级极小值的Beta校准器（0-1连续目标值）
核心优化：对数变换损失、针对性参数初始化、数值稳定性处理
"""

import numpy as np
import logging
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------
# 基础配置：字体和日志
# --------------------------
def get_chinese_font():
    """自动查找支持中文的字体，解决matplotlib中文显示问题"""
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

# --------------------------
# 核心类：极小值Beta校准器
# --------------------------
class MinValueBetaCalibrator:
    """
    专门适配10^-7量级极小值的Beta校准器（0-1连续目标值）
    特点：
    1. 对数变换损失，提升极小值拟合精度
    2. 针对性参数初始化，适配极小值场景
    3. 完善的数值稳定性处理
    """
    def __init__(self, max_iter=300, tol=1e-9, verbose=True):
        self.max_iter = max_iter  # 增加迭代次数，确保极小值场景收敛
        self.tol = tol
        self.verbose = verbose
        # Beta分布的参数映射系数（4个参数：A,B对应alpha；C,D对应beta）
        self.A = 1.0
        self.B = 0.0
        self.C = 1.0
        self.D = 0.0
        self.fitted = False
        # 保存预处理相关的参数
        self.scores_log_mean = None  # 原始分数对数变换后的均值
        self.targets_mean = None     # 目标值的均值（10^-7量级）
        self.eps = 1e-20             # 数值稳定性极小值，防止log(0)或除零

    def _preprocess_scores(self, scores):
        """
        预处理极小分数：对数变换将10^-7量级转换为易优化的负数区间
        输入：原始分数（10^-7量级）
        输出：对数变换后的分数
        """
        # 加eps防止log(0)，因为scores是10^-7量级，加eps不影响数值
        scores_log = np.log(scores + self.eps)
        # 保存对数分数的均值，用于后续逆变换或可视化
        if self.scores_log_mean is None:
            self.scores_log_mean = np.mean(scores_log)
        return scores_log

    def fit(self, scores, targets):
        """
        训练Beta校准模型（适配10^-7量级极小值）
        参数：
        scores: 模型原始输出分数（10^-7量级的极小值）
        targets: 连续目标值（10^-7量级的极小值，0-1之间）
        """
        if self.verbose:
            logger.info("开始拟合10^-7量级极小值的Beta校准模型...")
            logger.info(f"原始分数范围：[{np.min(scores):.2e}, {np.max(scores):.2e}]")
            logger.info(f"目标值范围：[{np.min(targets):.2e}, {np.max(targets):.2e}]")

        # 输入验证与转换
        scores = np.array(scores, dtype=np.float64)  # 使用float64提升精度
        targets = np.array(targets, dtype=np.float64)

        if len(scores) != len(targets):
            raise ValueError("scores和targets的长度必须相同")
        if np.min(targets) < 0 or np.max(targets) > 1:
            raise ValueError("targets必须是0到1之间的连续值")
        if np.min(scores) <= 0:
            raise ValueError("scores必须是正数（10^-7量级的极小值）")

        # 保存目标值均值（用于参数初始化）
        self.targets_mean = np.mean(targets)
        logger.info(f"目标值均值：{self.targets_mean:.2e}")

        # 预处理分数：对数变换（将10^-7量级转换为负数区间，避免数值下溢）
        scores_processed = self._preprocess_scores(scores)

        # --------------------------
        # 针对性参数初始化（适配极小值场景）
        # 思路：让初始Beta分布的均值等于目标值均值（10^-7量级）
        # alpha = target_mean, beta = 1 - target_mean ≈ 1（因为target_mean是10^-7）
        # 反推初始参数：alpha = exp(A*score + B), beta = exp(C*score + D)
        # 取score的均值作为基准，初始化A=0（先固定斜率），B=log(alpha), D=log(beta)
        # --------------------------
        score_mean = np.mean(scores_processed)
        alpha_init = self.targets_mean  # 初始alpha=目标值均值（10^-7）
        beta_init = 1.0 - self.targets_mean  # 初始beta≈1
        # 初始参数：A=0, B=log(alpha_init) - A*score_mean; C=0, D=log(beta_init) - C*score_mean
        initial_params = [
            0.0, np.log(alpha_init + self.eps) - 0.0 * score_mean,
            0.0, np.log(beta_init + self.eps) - 0.0 * score_mean
        ]
        logger.info(f"参数初始化完成，初始alpha={alpha_init:.2e}，初始beta={beta_init:.2e}")

        # --------------------------
        # 定义目标函数和梯度（对数变换MSE损失）
        # --------------------------
        def objective(params):
            """目标函数：对数变换后的MSE损失（提升极小值拟合精度）"""
            a, b, c, d = params
            # 映射为Beta分布的形状参数（exp保证参数为正）
            alpha = np.exp(a * scores_processed + b)
            beta = np.exp(c * scores_processed + d)
            # Beta分布的均值（校准值，落在0-1区间）
            probs = alpha / (alpha + beta + self.eps)  # 加eps防止除零

            # 对数变换损失：放大极小值的差异
            log_probs = np.log(probs + self.eps)
            log_targets = np.log(targets + self.eps)
            loss = np.mean((log_probs - log_targets) ** 2)
            return loss

        def gradient(params):
            """梯度函数（对数变换MSE损失的一阶导数，保证优化速度）"""
            a, b, c, d = params
            alpha = np.exp(a * scores_processed + b)
            beta = np.exp(c * scores_processed + d)
            probs = alpha / (alpha + beta + self.eps)
            n = len(scores_processed)

            # 对数变换损失的梯度（链式法则）
            log_probs = np.log(probs + self.eps)
            log_targets = np.log(targets + self.eps)
            d_loss = 2 * (log_probs - log_targets) / (probs + self.eps) / n  # 对probs的导数

            # 对alpha和beta的导数
            d_probs_d_alpha = beta / (alpha + beta + self.eps) ** 2
            d_probs_d_beta = -alpha / (alpha + beta + self.eps) ** 2

            # 对参数a,b,c,d的导数（链式法则：d_loss -> d_probs -> d_alpha/d_beta -> d_params）
            d_alpha_da = scores_processed * alpha
            d_alpha_db = alpha
            d_beta_dc = scores_processed * beta
            d_beta_dd = beta

            d_a = np.sum(d_loss * d_probs_d_alpha * d_alpha_da)
            d_b = np.sum(d_loss * d_probs_d_alpha * d_alpha_db)
            d_c = np.sum(d_loss * d_probs_d_beta * d_beta_dc)
            d_d = np.sum(d_loss * d_probs_d_beta * d_beta_dd)

            return np.array([d_a, d_b, d_c, d_d])

        # --------------------------
        # 优化参数（L-BFGS-B，设置参数边界防止发散）
        # --------------------------
        # 设置参数边界：避免参数过大导致数值爆炸（极小值场景参数不宜过大）
        bounds = [(-10.0, 10.0), (-20.0, 20.0), (-10.0, 10.0), (-20.0, 20.0)]
        result = minimize(
            objective,
            initial_params,
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.max_iter, 'gtol': self.tol, 'disp': False}
        )

        # 更新参数
        self.A, self.B, self.C, self.D = result.x
        self.fitted = True

        # 输出拟合信息
        if self.verbose:
            logger.info(f"Beta校准模型拟合完成，优化状态: {result.success}")
            logger.info(f"迭代次数: {result.nit}/{self.max_iter}")
            logger.info(f"最终参数：A={self.A:.6f}, B={self.B:.6f}, C={self.C:.6f}, D={self.D:.6f}")

            # 计算校准前后的损失（同时输出原始MSE和对数MSE）
            calibrated_probs = self.predict(scores)
            # 原始MSE（直观反映极小值的拟合误差）
            before_calib_mse = mean_squared_error(targets, scores)
            after_calib_mse = mean_squared_error(targets, calibrated_probs)
            # 对数MSE（反映极小值的相对误差）
            before_calib_log_mse = np.mean((np.log(scores + self.eps) - np.log(targets + self.eps)) ** 2)
            after_calib_log_mse = np.mean((np.log(calibrated_probs + self.eps) - np.log(targets + self.eps)) ** 2)

            logger.info(f"校准前原始MSE：{before_calib_mse:.2e}，对数MSE：{before_calib_log_mse:.6f}")
            logger.info(f"校准后原始MSE：{after_calib_mse:.2e}，对数MSE：{after_calib_log_mse:.6f}")

        return self

    def predict(self, scores):
        """
        预测校准后的极小值（10^-7量级）
        参数：
        scores: 模型原始输出分数（10^-7量级）
        返回：校准后的连续值（0-1之间，10^-7量级）
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")

        scores = np.array(scores, dtype=np.float64)
        if np.min(scores) <= 0:
            raise ValueError("scores必须是正数（10^-7量级的极小值）")

        # 预处理分数：对数变换
        scores_processed = self._preprocess_scores(scores)

        # 计算Beta分布参数和均值
        alpha = np.exp(self.A * scores_processed + self.B)
        beta = np.exp(self.C * scores_processed + self.D)
        probs = alpha / (alpha + beta + self.eps)

        # 确保输出在0-1之间（数值误差防护）
        return np.clip(probs, 0, 1)

    def get_params(self):
        """获取模型参数"""
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        return {'A': self.A, 'B': self.B, 'C': self.C, 'D': self.D}

    def visualize_calibration(self, scores, targets, save_path="min_value_beta_calibration.png"):
        """
        可视化极小值校准效果（针对10^-7量级做特殊展示）
        参数：
        scores: 原始分数
        targets: 目标值
        save_path: 图片保存路径
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")

        scores = np.array(scores, dtype=np.float64)
        targets = np.array(targets, dtype=np.float64)
        calibrated_probs = self.predict(scores)

        # 为了可视化清晰，对数值做对数变换展示（10^-7量级直接展示会压缩在0附近）
        scores_log = np.log10(scores + self.eps)
        targets_log = np.log10(targets + self.eps)
        cal_probs_log = np.log10(calibrated_probs + self.eps)

        # 绘制可视化图
        plt.figure(figsize=(12, 10))

        # 子图1：原始分数vs目标值（对数尺度）
        plt.subplot(2, 2, 1)
        plt.scatter(scores_log, targets_log, alpha=0.6, c='blue', label='原始分数（log10）')
        plt.xlabel('原始分数（log10尺度）')
        plt.ylabel('目标值（log10尺度）')
        plt.title('原始分数与目标值的关系（对数尺度）')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 子图2：校准后值vs目标值（对数尺度）
        plt.subplot(2, 2, 2)
        plt.scatter(cal_probs_log, targets_log, alpha=0.6, c='red', label='校准后值（log10）')
        # 绘制完美校准线（对数尺度下的y=x）
        min_log = min(np.min(cal_probs_log), np.min(targets_log))
        max_log = max(np.max(cal_probs_log), np.max(targets_log))
        plt.plot([min_log, max_log], [min_log, max_log], 'k--', label='完美校准')
        plt.xlabel('校准后值（log10尺度）')
        plt.ylabel('目标值（log10尺度）')
        plt.title('校准后值与目标值的关系（对数尺度）')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 子图3：原始分数与校准后值的分布（原始尺度，直方图）
        plt.subplot(2, 2, 3)
        bins = np.logspace(np.log10(np.min(scores + self.eps)), np.log10(np.max(scores + self.eps)), 50)
        plt.hist(scores, bins=bins, alpha=0.5, label='原始分数', color='blue')
        plt.hist(calibrated_probs, bins=bins, alpha=0.5, label='校准后值', color='red')
        plt.hist(targets, bins=bins, alpha=0.3, label='目标值', color='green')
        plt.xscale('log')  # 对数刻度展示极小值分布
        plt.xlabel('值（log尺度）')
        plt.ylabel('频率')
        plt.title('数值分布对比（对数尺度）')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 子图4：校准误差分布（原始尺度）
        plt.subplot(2, 2, 4)
        error_raw = scores - targets
        error_cal = calibrated_probs - targets
        plt.hist(error_raw, bins=50, alpha=0.5, label='原始分数误差', color='blue')
        plt.hist(error_cal, bins=50, alpha=0.5, label='校准后值误差', color='red')
        plt.xlabel('误差（原始尺度，10^-7量级）')
        plt.ylabel('频率')
        plt.title('校准误差分布')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 防止图片裁剪
        plt.close()
        logger.info(f"极小值校准可视化已保存为：{save_path}")

# --------------------------
# 辅助函数：评估校准效果
# --------------------------
def evaluate_min_value_calibration(preds, targets, eps=1e-20):
    """
    评估极小值校准效果的指标（包含原始和对数尺度的指标）
    返回：原始MSE、MAE、R²；对数MSE、MAE
    """
    # 原始尺度指标
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    # 对数尺度指标（反映相对误差）
    log_preds = np.log(preds + eps)
    log_targets = np.log(targets + eps)
    log_mse = mean_squared_error(log_targets, log_preds)
    log_mae = mean_absolute_error(log_targets, log_preds)

    return {
        'raw_mse': mse,
        'raw_mae': mae,
        'raw_r2': r2,
        'log_mse': log_mse,
        'log_mae': log_mae
    }

# --------------------------
# 示例：10^-7量级极小值的Beta校准
# --------------------------
def example_usage():
    """示例：生成10^-7量级的模拟数据，测试Beta校准器"""
    logger.info("=== 10^-7量级极小值的Beta校准示例 ===")
    np.random.seed(42)  # 固定随机种子，结果可复现

    # --------------------------
    # 生成10^-7量级的模拟数据
    # --------------------------
    n_samples = 1000
    # 原始分数：10^-7量级的极小值（右偏分布）
    # 生成方式：beta分布（0.5, 5）生成0-1的数，再乘以1e-7
    raw_scores = np.random.beta(0.5, 5, n_samples) * 1e-7
    # 目标值：与原始分数呈非线性关系的10^-7量级极小值（模拟需要校准的场景）
    # 加入少量噪声，确保在10^-7量级
    noise = np.random.normal(0, 0.1e-7, n_samples)
    true_targets = 1.2 * raw_scores ** 0.8 + noise
    # 确保目标值为正且在0-1之间（10^-7量级，不会超过1）
    true_targets = np.clip(true_targets, 1e-8, 1e-6)

    # --------------------------
    # 初始化并训练校准器
    # --------------------------
    calibrator = MinValueBetaCalibrator(verbose=True)
    calibrator.fit(raw_scores, true_targets)

    # --------------------------
    # 预测与评估
    # --------------------------
    calibrated_values = calibrator.predict(raw_scores)
    # 评估校准效果
    eval_before = evaluate_min_value_calibration(raw_scores, true_targets)
    eval_after = evaluate_min_value_calibration(calibrated_values, true_targets)

    logger.info("=== 校准前评估指标 ===")
    logger.info(f"原始MSE：{eval_before['raw_mse']:.2e}，原始R²：{eval_before['raw_r2']:.6f}")
    logger.info(f"对数MSE：{eval_before['log_mse']:.6f}，对数MAE：{eval_before['log_mae']:.6f}")

    logger.info("=== 校准后评估指标 ===")
    logger.info(f"原始MSE：{eval_after['raw_mse']:.2e}，原始R²：{eval_after['raw_r2']:.6f}")
    logger.info(f"对数MSE：{eval_after['log_mse']:.6f}，对数MAE：{eval_after['log_mae']:.6f}")

    # --------------------------
    # 可视化校准效果
    # --------------------------
    calibrator.visualize_calibration(raw_scores, true_targets)

    logger.info("=== 10^-7量级极小值的Beta校准演示完成 ===")

# --------------------------
# 主函数
# --------------------------
if __name__ == "__main__":
    example_usage()

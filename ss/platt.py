#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Platt校准策略实现

Platt校准（Platt Scaling）是一种常用的概率校准方法，主要用于将分类器的输出转换为更准确的概率估计。
它通过拟合一个S形函数（sigmoid）来映射模型输出到真实概率分布。

公式：P(y=1|f) = 1 / (1 + exp(A*f + B))
其中，f是模型的原始输出，A和B是通过最小二乘法优化得到的参数。
"""

import numpy as np
import logging
from sklearn.metrics import log_loss
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm
# 自动查找支持中文的字体
def get_chinese_font():
    fonts = fm.fontManager.ttflist
    # 常见中文字体名关键词
    chinese_font_names = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'Source Han Sans', 'WenQuanYi']
    for font in fonts:
        for name in chinese_font_names:
            if name in font.name or name in font.fname:
                return font.name
    # 如果没找到，返回第一个sans-serif字体
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
    Platt校准器类，实现Platt Scaling算法
    """
    
    def __init__(self, max_iter=100, tol=1e-5, verbose=False):
        """
        初始化Platt校准器
        
        参数:
        max_iter: 最大迭代次数
        tol: 收敛阈值
        verbose: 是否输出详细日志
        """
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.A = 0.0  # 初始化参数A
        self.B = 0.0  # 初始化参数B
        self.fitted = False
    
    def fit(self, scores, targets):
        """
        训练Platt校准模型
        
        参数:
        scores: 模型的原始输出分数（可以是概率、logits或其他分数）
        targets: 真实标签（0或1）
        
        返回:
        self: 拟合后的校准器实例
        """
        if self.verbose:
            logger.info("开始拟合Platt校准模型...")
        
        # 验证输入
        scores = np.array(scores, dtype=float)
        targets = np.array(targets, dtype=int)
        
        if len(scores) != len(targets):
            raise ValueError("scores和targets的长度必须相同")
        
        # 检查标签是否只有0和1
        unique_labels = np.unique(targets)
        if not np.array_equal(unique_labels, np.array([0, 1])) and not np.array_equal(unique_labels, np.array([1])) and not np.array_equal(unique_labels, np.array([0])):
            raise ValueError("targets必须只包含0和1")
        
        # 计算先验概率
        n_pos = np.sum(targets)
        n_neg = len(targets) - n_pos
        
        if self.verbose:
            logger.info(f"正样本数量: {n_pos}, 负样本数量: {n_neg}")
        
        # 初始化A和B
        # 根据Platt的论文，使用先验概率初始化
        prior1 = n_pos / len(targets)
        prior0 = n_neg / len(targets)
        
        # 初始参数设置
        A = 0.0
        B = np.log((prior0 + 1e-10) / (prior1 + 1e-10))  # 防止除零
        
        # 定义目标函数（负对数似然）
        def objective(params):
            a, b = params
            logits = a * scores + b
            # 使用稳定的sigmoid计算
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            # 计算负对数似然
            loss = -np.sum(targets * np.log(prob + 1e-10) + (1 - targets) * np.log(1 - prob + 1e-10))
            return loss
        
        # 定义梯度函数
        def gradient(params):
            a, b = params
            logits = a * scores + b
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            
            d_a = np.sum((prob - targets) * scores)
            d_b = np.sum(prob - targets)
            
            return np.array([d_a, d_b])
        
        # 优化参数
        initial_params = [A, B]
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            jac=gradient,
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )
        
        # 更新参数
        self.A, self.B = result.x
        self.fitted = True
        
        if self.verbose:
            logger.info(f"Platt校准模型拟合完成")
            logger.info(f"优化状态: {result.success}")
            logger.info(f"迭代次数: {result.nit}")
            logger.info(f"最终参数: A={self.A:.6f}, B={self.B:.6f}")
            
            # 计算校准后的损失
            calibrated_probs = self.predict(scores)
            loss = log_loss(targets, calibrated_probs)
            logger.info(f"校准后的对数损失: {loss:.6f}")
        
        return self
    
    def predict(self, scores):
        """
        使用拟合好的模型进行概率预测
        
        参数:
        scores: 模型的原始输出分数
        
        返回:
        probs: 校准后的概率值
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        scores = np.array(scores, dtype=float)
        logits = self.A * scores + self.B
        
        # 计算sigmoid函数值，使用数值稳定的方式
        # 对logits进行裁剪，避免溢出
        logits = np.clip(logits, -20, 20)
        probs = 1.0 / (1.0 + np.exp(-logits))
        
        return probs
    
    def get_params(self):
        """
        获取模型参数
        
        返回:
        params: 包含A和B参数的字典
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        return {'A': self.A, 'B': self.B}
    
    def set_params(self, A, B):
        """
        手动设置模型参数
        
        参数:
        A: 参数A
        B: 参数B
        """
        self.A = A
        self.B = B
        self.fitted = True
    
    def visualize_calibration(self, scores, targets, bins=10):
        """
        可视化校准效果
        
        参数:
        scores: 模型的原始输出分数
        targets: 真实标签
        bins: 分桶数量
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        scores = np.array(scores)
        targets = np.array(targets)
        
        # 计算原始分数的分位数
        bin_edges = np.linspace(min(scores), max(scores), bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 计算每个桶的平均原始概率和实际频率
        orig_probs = []
        actual_probs = []
        bin_sizes = []
        
        for i in range(bins):
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i+1])
            if np.sum(mask) > 0:
                orig_prob = np.mean(scores[mask])
                actual_prob = np.mean(targets[mask])
                orig_probs.append(orig_prob)
                actual_probs.append(actual_prob)
                bin_sizes.append(np.sum(mask))
            else:
                orig_probs.append(np.nan)
                actual_probs.append(np.nan)
                bin_sizes.append(0)
        
        # 计算校准后的概率
        calibrated_probs = self.predict(scores)
        
        # 计算校准后每个桶的平均概率
        cal_probs = []
        for i in range(bins):
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i+1])
            if np.sum(mask) > 0:
                cal_prob = np.mean(calibrated_probs[mask])
                cal_probs.append(cal_prob)
            else:
                cal_probs.append(np.nan)
        
        # 绘制校准曲线
        plt.figure(figsize=(12, 6))
        
        # 绘制原始概率vs实际概率
        plt.subplot(1, 2, 1)
        plt.scatter(orig_probs, actual_probs, s=bin_sizes, alpha=0.5, c='blue', label='原始概率')
        plt.scatter(cal_probs, actual_probs, s=bin_sizes, alpha=0.5, c='red', label='校准概率')
        plt.plot([0, 1], [0, 1], 'k--', label='完美校准')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Actual Probability')
        plt.title('Calibration Curve')
        plt.grid(True)
        plt.legend()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # 绘制校准前后的概率分布
        plt.subplot(1, 2, 2)
        plt.hist(scores, bins=50, alpha=0.5, label='原始概率分布')
        plt.hist(calibrated_probs, bins=50, alpha=0.5, label='校准概率分布')
        plt.xlabel('Probability Value')
        plt.ylabel('Frequency')
        plt.title('Probability Distribution Comparison')
        plt.grid(True)
        plt.legend()
        plt.xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig('platt_calibration_visualization.png', dpi=300)
        plt.close()
        
        logger.info("校准可视化已保存为 platt_calibration_visualization.png")

def calculate_ece(scores, targets, n_bins=10):
    """
    计算期望校准误差（Expected Calibration Error, ECE）
    
    参数:
    scores: 预测的概率值
    targets: 真实标签
    n_bins: 分桶数量
    
    返回:
    ece: 期望校准误差
    """
    scores = np.array(scores)
    targets = np.array(targets)
    
    # 计算分桶边界
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    total_samples = len(scores)
    
    for i in range(n_bins):
        # 找到当前桶中的样本
        in_bin = np.logical_and(scores >= bin_edges[i], scores < bin_edges[i+1])
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            # 计算桶内的平均预测概率和实际概率
            avg_pred = np.mean(scores[in_bin])
            avg_true = np.mean(targets[in_bin])
            
            # 计算该桶的加权误差
            bin_error = np.abs(avg_pred - avg_true) * (bin_size / total_samples)
            ece += bin_error
    
    return ece

def example_usage():
    """
    演示Platt校准的使用方法
    """
    logger.info("=== Platt校准示例 ===")
    
    # 生成模拟数据
    np.random.seed(42)
    
    # 生成原始分数（假设是SVM的输出或未校准的概率）
    n_samples = 1000
    # 生成有偏差的预测分数
    raw_scores = np.random.beta(2, 5, n_samples)  # 生成右偏的分数分布
    
    # 根据分数生成标签（分数越高，正例概率越大，但添加一些随机性）
    noise = np.random.normal(0, 0.1, n_samples)
    prob_true = 1 / (1 + np.exp(-10 * (raw_scores - 0.5))) + noise  # Sigmoid转换
    prob_true = np.clip(prob_true, 0, 1)  # 确保在[0,1]范围内
    targets = np.random.binomial(1, prob_true)
    
    # 初始化并拟合Platt校准器
    calibrator = PlattCalibrator(verbose=True)
    calibrator.fit(raw_scores, targets)
    
    # 进行预测
    calibrated_probs = calibrator.predict(raw_scores)
    
    # 计算校准前后的ECE
    ece_before = calculate_ece(raw_scores, targets)
    ece_after = calculate_ece(calibrated_probs, targets)
    
    logger.info(f"校准前ECE: {ece_before:.6f}")
    logger.info(f"校准后ECE: {ece_after:.6f}")
    logger.info(f"ECE改善比例: {(ece_before - ece_after) / ece_before * 100:.2f}%")
    
    # 可视化校准效果
    calibrator.visualize_calibration(raw_scores, targets)

def integrate_with_existing_calibration_system():
    """
    演示如何将Platt校准集成到现有的校准系统中
    """
    logger.info("\n=== 与现有校准系统集成示例 ===")
    logger.info("以下是如何将Platt校准器集成到现有校准框架中的示例代码：")
    
    integration_code = '''
# Integrate Platt calibration into existing calibration system
def calibrate_with_platt(df, dimension='adid', min_samples=100):
    """
    Calibrate pcvr in dataframe using Platt scaling.
    
    Args:
    df: DataFrame containing pcvr and conversion labels
    dimension: Grouping dimension
    min_samples: Minimum sample size threshold
    
    Returns:
    calibrated_df: DataFrame with calibrated cvr
    """
    from platt_calibration import PlattCalibrator
    import pandas as pd
    
    result_df = df.copy()
    result_df['platt_calibrated_cvr'] = result_df['pcvr']
    
    # Calibrate by dimension groups
    for key, group in df.groupby(dimension):
        if len(group) >= min_samples:
            # Fit Platt calibrator
            calibrator = PlattCalibrator()
            calibrator.fit(group['pcvr'].values, group['conversion_label'].values)
            
            # Perform calibration
            mask = result_df[dimension] == key
            result_df.loc[mask, 'platt_calibrated_cvr'] = calibrator.predict(result_df.loc[mask, 'pcvr'].values)
    
    return result_df
'''
    
    print(integration_code)

def compare_with_pava():
    """
    与PAVA算法进行比较
    """
    logger.info("\n=== 与PAVA算法比较 ===")
    logger.info("Platt校准 vs PAVA校准的主要区别：")
    
    comparison_points = [
        "1. 原理不同：Platt校准假设存在S形函数映射，而PAVA是基于Pool Adjacent Violators Algorithm的保序回归",
        "2. 适用场景：Platt校准适用于模型输出已经相对平滑的情况；PAVA适用于需要保证校准后概率单调递增的场景",
        "3. 计算复杂度：Platt校准通常更高效，特别是对于大规模数据；PAVA需要排序和迭代合并",
        "4. 单调性：PAVA保证校准后概率单调递增；Platt校准不保证单调性",
        "5. 灵活性：Platt校准参数更少，更容易解释；PAVA可以捕捉更复杂的校准关系",
        "6. 数据需求：Platt校准对小样本更鲁棒；PAVA通常需要足够的数据来形成有效的分桶"
    ]
    
    for point in comparison_points:
        print(point)

if __name__ == "__main__":
    # 运行示例
    example_usage()
    
    # 显示集成方法
    integrate_with_existing_calibration_system()
    
    # 与PAVA比较
    compare_with_pava()
    
    logger.info("\nPlatt校准实现完成！")

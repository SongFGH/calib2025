import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import seaborn as sns
import os
import pickle
import sys
import json
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import argparse

# 导入Platt校准器
from platt_calibration import PlattCalibrator

PCVR_MAX=0.1
DIMENSIONS = ['rpkid']
SOH = '\x01'
#PREFIX = "2" + SOH + "f_3_"
PREFIX = "2" + SOH + "4_"
  # ASCII控制字符SOH，对应^A

def load_data(file_path):
    """
    加载广告数据

    参数:
    file_path: 数据文件路径

    返回:
    df: 处理后的数据框
    """
    # 加载数据，假设是空格或制表符分隔的文本文件
    df = pd.read_csv(file_path, sep='\s+', header=None)

    # 检查列数
    print(f"原始数据列数: {len(df.columns)}")

    # 如果文件有11列（包含day列和rpkid列），设置列名
    if len(df.columns) == 13:  # 确认列数，包含day列和rpkid列
        df.columns = ['day', 'adid', 'adgroupid', 'pkg', '二级类目', 'reqid', 'pcvr', '转化label', 'cvtype', 'spent', 'rpkid', 'app_id', 'pst_id']
    elif len(df.columns) == 11:  # 确认列数，包含day列和rpkid列
        df.columns = ['day', 'adid', 'adgroupid', 'pkg', '二级类目', 'reqid', 'pcvr', '转化label', 'cvtype', 'spent', 'rpkid']
    elif len(df.columns) == 10:  # 包含day列但不包含rpkid列
        df.columns = ['day', 'adid', 'adgroupid', 'pkg', '二级类目', 'reqid', 'pcvr', '转化label', 'cvtype', 'spent']
    elif len(df.columns) == 8:  # 兼容之前的7列数据，增加rpkid列
        df.columns = ['adid', 'adgroupid', 'pkg', 'rpkid', '二级类目', 'reqid', 'pcvr', '转化label', 'cvtype']
    elif len(df.columns) == 7:  # 兼容之前的7列数据
        df.columns = ['adid', 'adgroupid', 'pkg', '二级类目', 'reqid', 'pcvr', '转化label', 'cvtype']

    # 如果存在day列，基于day列计算weekday列
    if 'day' in df.columns:
        # 将day列转换为日期类型
        df['day'] = pd.to_datetime(df['day'], format='%Y-%m-%d', errors='coerce')

        # 计算weekday列（0=周一，6=周日）
        df['weekday'] = df['day'].dt.weekday

        print("已基于day列计算weekday列")

    # 数据类型转换
    df['pcvr'] = df['pcvr'].astype(float)
    df['转化label'] = df['转化label'].astype(float)
    df['adid'] = df['adid'].astype(str)
    df['adgroupid'] = df['adgroupid'].astype(str)

    # 如果存在rpkid列，转换为字符串类型
    if 'rpkid' in df.columns:
        df['rpkid'] = df['rpkid'].astype(str)

    # 如果存在spent列，转换为浮点数类型
    if 'spent' in df.columns:
        df['spent'] = df['spent'].astype(float)

    # 检查并处理缺失值
    print(f"数据加载完成，共 {len(df)} 行")
    print(f"缺失值统计:\n{df.isnull().sum()}")

    # 删除pcvr为NaN的行
    df = df.dropna(subset=['pcvr'])

    return df

def load_adid_fallback_config(config_path):
    """
    加载adid回退维度配置文件

    参数:
    config_path: 配置文件路径

    返回:
    fallback_config: adid回退配置字典，格式为{adid: [维度1, 维度2, ...]}
    """
    fallback_config = {}

    try:
        with open(config_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    adid = parts[0]
                    dimensions = parts[1].split(',')
                    fallback_config[adid] = dimensions

        print(f"加载回退配置完成，共 {len(fallback_config)} 个adid配置")
    except Exception as e:
        print(f"加载回退配置失败: {str(e)}")
        fallback_config = {}

    return fallback_config

def equal_frequency_binning(df, n_bins=100):
    """
    1. 根据pcvr升序排序
    2. 等频分桶，计算每个桶的平均cvr

    参数:
    df: 包含pcvr和转化label的数据框
    n_bins: 分桶数量

    返回:
    bins_df: 包含每个桶的pcvr和cvr的数据框
    """
    # 1. 根据pcvr升序排序
    df_sorted = df.sort_values(by='pcvr')

    # 2. 等频分桶
    # 计算每个桶的大小
    bin_size = len(df) // n_bins
    if bin_size == 0:
        bin_size = 1
        n_bins = len(df)

    bins_data = []

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(df)

        bin_df = df_sorted.iloc[start_idx:end_idx]

        # 计算桶内平均pcvr和实际cvr
        avg_pcvr = bin_df['pcvr'].mean()
        actual_cvr = bin_df['转化label'].mean()

        bins_data.append({
            'bin_id': i,
            'pcvr': avg_pcvr,
            'cvr': actual_cvr,
            'count': len(bin_df)
        })

    bins_df = pd.DataFrame(bins_data)
    print(f"等频分桶完成，共 {len(bins_df)} 个桶")

    return bins_df

def pava_algorithm(x, y, weights=None):
    """
    3. 执行PAVA算法（Pool Adjacent Violators Algorithm）

    参数:
    x: pcvr值序列
    y: cvr值序列
    weights: 权重序列，默认为None（等权重）

    返回:
    post_cvr: PAVA算法后的非降序cvr序列
    """
    n = len(x)
    if n <= 1:
        return y.copy()

    if weights is None:
        weights = np.ones(n)

    # 初始化
    solution = y.copy()
    active_set = [[i] for i in range(n)]

    while len(active_set) > 1:
        # 检查相邻块是否违反单调性
        i = 0
        while i < len(active_set) - 1:
            block1 = active_set[i]
            block2 = active_set[i + 1]

            # 计算每个块的加权平均值
            mean1 = sum(solution[j] * weights[j] for j in block1) / sum(weights[j] for j in block1)
            mean2 = sum(solution[j] * weights[j] for j in block2) / sum(weights[j] for j in block2)

            # 如果违反单调性约束
            if mean1 > mean2:
                # 合并两个块
                merged_block = block1 + block2
                merged_weight = sum(weights[j] for j in merged_block)
                merged_mean = sum(solution[j] * weights[j] for j in merged_block) / merged_weight

                # 更新解
                for j in merged_block:
                    solution[j] = merged_mean

                # 更新活动集
                active_set[i] = merged_block
                active_set.pop(i + 1)

                # 如果不是第一个块，需要回退检查前面的块
                if i > 0:
                    i -= 1
            else:
                i += 1

        # 如果没有违反单调性的块，算法终止
        if i == len(active_set) - 1:
            break

    return solution

def compress_samples(pcvr, post_cvr):
    """
    4. 样本压缩，合并PAVA产生的非降序样本点

    参数:
    pcvr: pcvr值序列
    post_cvr: PAVA算法后的cvr值序列

    返回:
    compressed_pcvr: 压缩后的pcvr值序列
    compressed_cvr: 压缩后的cvr值序列
    """
    if len(pcvr) <= 1:
        return pcvr, post_cvr

    compressed_pcvr = []
    compressed_cvr = []

    current_cvr = post_cvr[0]
    current_group = [0]

    for i in range(1, len(post_cvr)):
        if abs(post_cvr[i] - current_cvr) < 1e-10:  # 浮点数比较，使用小阈值
            current_group.append(i)
        else:
            # 当前组结束，取最后一个pcvr
            compressed_pcvr.append(pcvr[current_group[-1]])
            compressed_cvr.append(current_cvr)

            # 开始新的组
            current_cvr = post_cvr[i]
            current_group = [i]

    # 处理最后一组
    compressed_pcvr.append(pcvr[current_group[-1]])
    compressed_cvr.append(current_cvr)

    return np.array(compressed_pcvr), np.array(compressed_cvr)

def create_interpolation_function(pcvr, post_cvr):
    """
    5. 训练样本空间映射到预估空间，通过线性插值

    参数:
    pcvr: 压缩后的pcvr值序列
    post_cvr: 压缩后的cvr值序列

    返回:
    calibration_func: 校准函数，输入为pcvr，输出为校准后的cvr
    """
    # 确保边界条件
    if len(pcvr) == 0:
        # 如果没有数据点，返回恒等函数
        return lambda x: x

    if pcvr[0] > 0:
        pcvr = np.concatenate(([0], pcvr))
        post_cvr = np.concatenate(([0], post_cvr))

    if pcvr[-1] < PCVR_MAX:
        pcvr = np.concatenate((pcvr, [PCVR_MAX]))
        post_cvr = np.concatenate((post_cvr, [PCVR_MAX]))

    # 创建线性插值函数
    calibration_func = interpolate.interp1d(
        pcvr, post_cvr,
        kind='linear',
        bounds_error=False,
        fill_value=(post_cvr[0], post_cvr[-1])
    )

    return calibration_func

def create_threshold_manager(history_file='threshold_history.json'):
    """
    创建一个自适应阈值管理器

    参数:
    history_file: 历史性能记录文件路径

    返回:
    threshold_manager: 阈值管理对象
    """
    # 尝试加载历史性能数据
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = {
                'model_performance': {},  # 存储不同阈值下的模型性能
                'dimension_performance': {},  # 存储不同维度的模型性能
                'last_updated': None
            }
    except Exception as e:
        print(f"警告: 加载阈值历史数据失败: {e}")
        history = {
            'model_performance': {},
            'dimension_performance': {},
            'last_updated': None
        }

    def save_history():
        """保存历史性能数据"""
        try:
            history['last_updated'] = pd.Timestamp.now().isoformat()
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"警告: 保存阈值历史数据失败: {e}")

    def update_performance(min_samples, modelname, metrics):
        """
        更新特定阈值和模型类型的性能记录

        参数:
        min_samples: 使用的最小样本量阈值
        modelname: 模型类型
        metrics: 模型性能指标
        """
        key = f"{min_samples}_{modelname}"
        if key not in history['model_performance']:
            history['model_performance'][key] = []

        # 记录当前性能和时间戳
        # 提取指标，处理不同的metrics格式
        ece = 0
        coverage = 0
        mse = 0

        if isinstance(metrics, dict):
            if 'ece_calibrated' in metrics:
                ece = metrics['ece_calibrated']
            elif 'Overall' in metrics and isinstance(metrics['Overall'], dict):
                # 尝试从现有格式中提取必要信息
                # 注意：这里可能需要根据实际评估结果调整
                if 'ECE_Calibrated' in metrics['Overall']:
                    ece = metrics['Overall']['ECE_Calibrated']
                if 'MSE_Calibrated' in metrics['Overall']:
                    mse = metrics['Overall']['MSE_Calibrated']
            coverage = 1.0  # 默认100%覆盖率
            if 'ByDimension' in metrics:
                # 简单计算覆盖率
                pass

        history['model_performance'][key].append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'ece': ece,
            'mse': mse,
            'coverage': coverage
        })

        # 只保留最近10条记录
        if len(history['model_performance'][key]) > 10:
            history['model_performance'][key] = history['model_performance'][key][-10:]

        save_history()

    def get_optimal_threshold(base_threshold, modelname, conversion_rate=None, dimension=None):
        """
        根据历史性能获取最优阈值

        参数:
        base_threshold: 基础阈值
        modelname: 模型类型
        conversion_rate: 转化率
        dimension: 维度名称

        返回:
        optimal_threshold: 最优阈值
        confidence: 阈值的置信度(0-1)
        """
        # 首先应用转化率调整
        if conversion_rate is not None:
            if conversion_rate < 0.01:
                adjusted_base = base_threshold * 1.5
            elif conversion_rate > 0.1:
                adjusted_base = base_threshold * 0.8
            else:
                adjusted_base = base_threshold
        else:
            adjusted_base = base_threshold

        # 检查是否有历史性能数据可以参考
        best_threshold = adjusted_base
        best_score = float('inf')
        confidence = 0.5  # 默认中等置信度

        # 搜索附近的阈值点
        for delta in [-20, -10, 0, 10, 20]:
            candidate_threshold = max(10, int(adjusted_base + delta))
            key = f"{candidate_threshold}_{modelname}"

            if key in history['model_performance'] and history['model_performance'][key]:
                # 获取最近的性能记录
                recent_performances = history['model_performance'][key][-3:]  # 最近3次
                avg_ece = np.mean([p['ece'] for p in recent_performances])
                avg_coverage = np.mean([p['coverage'] for p in recent_performances])

                # 综合评分：ECE越低越好，但也需要考虑覆盖率
                score = 0.7 * avg_ece + 0.3 * (1 - avg_coverage)

                if score < best_score:
                    best_score = score
                    best_threshold = candidate_threshold
                    confidence = min(1.0, len(recent_performances) / 3.0)

        # 对BBQ模型进一步调整
        if modelname == 'bbq':
            best_threshold = max(20, int(best_threshold * 0.5))
        elif modelname == 'platt':
            # Platt校准需要估计两个参数，需要比BBQ更多的样本
            best_threshold = max(50, int(best_threshold * 0.8))

        return best_threshold, confidence

    return {
        'update_performance': update_performance,
        'get_optimal_threshold': get_optimal_threshold,
        'history': history
    }

# 创建全局阈值管理器实例
_threshold_manager = create_threshold_manager()

def is_sample_size_confident(sample_size, min_samples=50, modelname='pava', conversion_rate=None, dimension=None, adaptive=True):
    """
    判断样本数量是否足够置信

    参数:
    sample_size: 样本数量
    min_samples: 最小置信样本数量
    modelname: 模型类型，'pava'或'bbq'或'platt'
    conversion_rate: 转化率，用于动态调整置信阈值
    dimension: 维度名称，用于自适应阈值调整
    adaptive: 是否使用自适应阈值

    返回:
    is_confident: 是否置信
    adjusted_threshold: 调整后的阈值
    confidence: 阈值的置信度(0-1)（仅在自适应模式下有意义）
    """
    if adaptive:
        # 使用自适应阈值管理器获取最优阈值
        optimal_threshold, confidence = _threshold_manager['get_optimal_threshold'](
            min_samples, modelname, conversion_rate, dimension
        )
        adjusted_threshold = optimal_threshold
    else:
        # 使用传统方法计算阈值
        # 根据模型类型调整基础阈值
        if modelname == 'bbq':
            # BBQ模型对小样本更友好，可以使用更小的阈值
            base_threshold = max(20, min_samples * 0.5)
        elif modelname == 'platt':
            # Platt校准需要估计两个参数，需要比BBQ更多但比PAVA更少的样本
            base_threshold = max(50, min_samples * 0.8)
        else:
            # PAVA模型需要更多样本
            base_threshold = min_samples

        # 如果有转化率信息，可以进一步调整阈值
        if conversion_rate is not None:
            # 低转化率通常需要更多样本才能稳定
            if conversion_rate < 0.01:
                adjusted_threshold = base_threshold * 1.5
            elif conversion_rate > 0.1:
                adjusted_threshold = base_threshold * 0.8
            else:
                adjusted_threshold = base_threshold
        else:
            adjusted_threshold = base_threshold
        confidence = 0.5  # 默认中等置信度

    return sample_size >= adjusted_threshold, adjusted_threshold, confidence

def analyze_business_impact(df, deviation_ranges=[0, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]):
    """
    分析纠偏后的业务影响，按维度聚合计算偏差并统计不同偏差区间的指标

    参数:
    df: 包含原始pcvr、校准后calibrated_cvr和spent的数据框
    deviation_ranges: 偏差区间列表

    返回:
    business_metrics: 业务评估指标字典
    """
    # 确保偏差区间是排序的
    deviation_ranges = sorted(deviation_ranges)

    # 确保数据框包含必要的列
    required_columns = ['adid', 'pcvr', '转化label', 'calibrated_cvr']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据框缺少必要的列: {col}")

    # 如果没有spent列，添加一列spent=1
    if 'spent' not in df.columns:
        df = df.copy()  # 创建副本避免修改原始数据
        df['spent'] = 1
        print("警告: 数据中没有spent列，已自动添加spent=1")

    # 初始化结果字典
    business_metrics = {
        'ranges': deviation_ranges,
        'summary': {},
        'dimensions': {}
    }

    # 计算总体统计
    total_samples = len(df)
    total_conversions = df['转化label'].sum()
    total_spent = df['spent'].sum()

    business_metrics['summary'] = {
        'total_samples': total_samples,
        'total_conversions': total_conversions,
        'total_spent': total_spent,
        'total_pcvr': df['pcvr'].sum(),
        'total_calibrated_cvr': df['calibrated_cvr'].sum(),
        'pcvr_bias': df['pcvr'].sum() / df['转化label'].sum() if df['转化label'].sum() > 0 else 0,
        'calibrated_cvr_bias': df['calibrated_cvr'].sum() / df['转化label'].sum() if df['转化label'].sum() > 0 else 0,
        'total_unique_adids': df['adid'].nunique(),
        'total_unique_adgroupids': df['adgroupid'].nunique() if 'adgroupid' in df.columns else 0,
        'total_unique_pkgs': df['pkg'].nunique()
    }

    # 定义要聚合的维度
    dimensions = ['adid']
    if 'adgroupid' in df.columns:
        dimensions.append('adgroupid')
    if 'pkg' in df.columns:
        dimensions.append('pkg')

    # 对每个维度进行聚合分析
    for dimension in dimensions:
        # 按维度聚合数据
        aggregated_df = df.groupby(dimension).agg({
            'pcvr': 'mean',  # 平均预测转化率
            '转化label': ['sum', 'count'],  # 总转化数和样本数
            'calibrated_cvr': 'mean',  # 平均校准后转化率
            'spent': 'sum'  # 总消耗
        }).reset_index()

        # 重命名列
        aggregated_df.columns = [dimension, 'avg_pcvr', 'total_conversions', 'sample_count', 'avg_calibrated_cvr', 'total_spent']

        # 计算实际cvr（转化数/样本数）
        aggregated_df['actual_cvr'] = aggregated_df['total_conversions'] / aggregated_df['sample_count'].replace(0, np.nan)

        # 计算纠偏前偏差和纠偏后偏差
        aggregated_df['pre_calibration_deviation'] = aggregated_df['avg_pcvr'] / aggregated_df['actual_cvr'].replace(0, np.nan)
        aggregated_df['post_calibration_deviation'] = aggregated_df['avg_calibrated_cvr'] / aggregated_df['actual_cvr'].replace(0, np.nan)

        # 初始化该维度的结果
        business_metrics['dimensions'][dimension] = {
            'aggregated_data': aggregated_df.to_dict('records'),
            'pre_calibration': {},  # 纠偏前各偏差区间的统计
            'post_calibration': {}  # 纠偏后各偏差区间的统计
        }

        # 统计纠偏前的偏差区间
        dimension_total_spent = aggregated_df['total_spent'].sum()

        for phase in ['pre_calibration', 'post_calibration']:
            deviation_col = f'{phase}_deviation'

            # 处理每个偏差区间
            for i in range(len(deviation_ranges) - 1):
                lower = deviation_ranges[i]
                upper = deviation_ranges[i+1]

                # 创建区间标签
                if i == 0:
                    # 第一个区间：小于下限
                    mask = aggregated_df[deviation_col] < lower
                    range_label = f'<{lower}'
                else:
                    # 中间区间
                    mask = (aggregated_df[deviation_col] >= lower) & (aggregated_df[deviation_col] < upper)
                    range_label = f'{lower}-{upper}'

                # 最后一个区间
                if i == len(deviation_ranges) - 2:
                    mask = aggregated_df[deviation_col] >= lower
                    range_label = f'>={lower}'

                # 获取该区间的数据
                range_df = aggregated_df[mask]

                # 统计该区间的指标
                if range_df.empty:
                    business_metrics['dimensions'][dimension][phase][range_label] = {
                        'item_count': 0,
                        'sample_count': 0,
                        'total_conversions': 0,
                        'total_spent': 0,
                        'spent_percentage': 0
                    }
                else:
                    # 计算指标
                    item_count = len(range_df)  # adid/adgroupid/pkg数量
                    sample_count = range_df['sample_count'].sum()  # 总样本量
                    total_conversions = range_df['total_conversions'].sum()  # 总转化数
                    total_spent = range_df['total_spent'].sum()  # 总消耗
                    spent_percentage = (total_spent / dimension_total_spent * 100) if dimension_total_spent > 0 else 0  # 消耗占比

                    business_metrics['dimensions'][dimension][phase][range_label] = {
                        'item_count': item_count,
                        'sample_count': sample_count,
                        'total_conversions': total_conversions,
                        'total_spent': total_spent,
                        'spent_percentage': spent_percentage
                    }

    return business_metrics

def print_business_metrics(business_metrics):
    """
    打印业务评估指标

    参数:
    business_metrics: 业务评估指标字典
    """
    print("\n=== 业务评估结果 ===")

    # 打印总体统计
    summary = business_metrics['summary']
    print(f"\n总体统计:")
    print(f"  总样本数量: {summary['total_samples']:,}")
    print(f"  总转化数: {summary['total_conversions']:,}")
    print(f"  总消耗: {summary['total_spent']:.2f}")
    print(f"  总pcvr: {summary['total_pcvr']:.6f}")
    print(f"  总校准后cvr: {summary['total_calibrated_cvr']:.6f}")
    print(f"  pcvr偏差: {summary['pcvr_bias']:.4f}")
    print(f"  校准后cvr偏差: {summary['calibrated_cvr_bias']:.4f}")
    print(f"  唯一adid数量: {summary['total_unique_adids']:,}")
    if summary['total_unique_adgroupids'] > 0:
        print(f"  唯一adgroupid数量: {summary['total_unique_adgroupids']:,}")
    print(f"  唯一pkg数量: {summary['total_unique_pkgs']:,}")

    # 打印各维度的偏差区间统计
    print("\n各维度偏差区间统计:")

    # 定义偏差区间
    ranges = business_metrics['ranges']

    # 遍历每个维度
    for dimension, dimension_data in business_metrics['dimensions'].items():
        print(f"\n--- {dimension} 维度统计 ---")

        # 打印纠偏前的统计
        print(f"\n纠偏前:")
        print(f"{'偏差区间':<15} {'{dimension}数量':<15} {'样本量':<12} {'转化数':<12} {'总消耗':<12} {'消耗占比':<12}".format(dimension=dimension))
        print("-" * 85)

        for i in range(len(ranges) - 1):
            lower = ranges[i]
            upper = ranges[i+1]

            if i == 0:
                range_label = f'<{lower}'
            elif i == len(ranges) - 2:
                range_label = f'>={lower}'
            else:
                range_label = f'{lower}-{upper}'

            metrics = dimension_data['pre_calibration'].get(range_label, {})
            print(f"{range_label:<15} {metrics.get('item_count', 0):<15,} {metrics.get('sample_count', 0):<12,} {metrics.get('total_conversions', 0):<12,} {metrics.get('total_spent', 0):<11.2f} {metrics.get('spent_percentage', 0):<11.2f}%")

        # 打印纠偏后的统计
        print(f"\n纠偏后:")
        print(f"{'偏差区间':<15} {'{dimension}数量':<15} {'样本量':<12} {'转化数':<12} {'总消耗':<12} {'消耗占比':<12}".format(dimension=dimension))
        print("-" * 85)

        for i in range(len(ranges) - 1):
            lower = ranges[i]
            upper = ranges[i+1]

            if i == 0:
                range_label = f'<{lower}'
            elif i == len(ranges) - 2:
                range_label = f'>={lower}'
            else:
                range_label = f'{lower}-{upper}'

            metrics = dimension_data['post_calibration'].get(range_label, {})
            print(f"{range_label:<15} {metrics.get('item_count', 0):<15,} {metrics.get('sample_count', 0):<12,} {metrics.get('total_conversions', 0):<12,} {metrics.get('total_spent', 0):<11.2f} {metrics.get('spent_percentage', 0):<11.2f}%")

    # 打印偏差区间定义
    print("\n偏差区间说明:")
    print("  偏差 = 预测转化率 / 实际转化率")
    print("  <1.0: 预测偏低；=1.0: 预测准确；>1.0: 预测偏高")

def get_dimension_key(row, dimension):
    """
    根据维度获取对应的键值

    参数:
    row: 数据行
    dimension: 维度名称

    返回:
    key: 维度对应的键值
    """
    if dimension == 'adid':
        return row['adid']
    elif dimension == 'adgroupid':
        return row['adgroupid']
    elif dimension == 'pkg':
        return row['pkg']
    elif dimension == 'rpkid':
        return row['rpkid']
    elif dimension == '二级类目':
        return row['二级类目']
    elif dimension == 'cvtype':
        return row['cvtype']
    else:
        return None

def bayesian_binning_into_quantiles(df, n_bins=100, prior_alpha=1, prior_beta=1, use_adaptive_mixing=False):
    """
    Bayesian Binning into Quantiles (BBQ)纠偏方法

    参数:
    df: 包含pcvr和转化label的数据框
    n_bins: 分桶数量
    prior_alpha: Beta先验的alpha参数
    prior_beta: Beta先验的beta参数
    use_adaptive_mixing: 是否使用自适应混合校准策略

    返回:
    bins_df: 包含每个桶的信息的数据框
    bin_edges: 分桶边界
    """
    # 根据pcvr升序排序
    df_sorted = df.sort_values(by='pcvr')

    # 等频分桶的初始边界
    bin_size = len(df_sorted) // n_bins
    if bin_size == 0:
        bin_size = 1
        n_bins = len(df_sorted)

    # 初始分桶边界
    bin_edges = [df_sorted['pcvr'].iloc[i * bin_size] for i in range(n_bins)]
    bin_edges.append(df_sorted['pcvr'].iloc[-1])

    # 计算每个样本的桶索引
    df_sorted['bin'] = np.digitize(df_sorted['pcvr'], bin_edges[:-1], right=True)

    # 计算每个桶的Beta后验参数
    bin_stats = []
    for bin_id in range(1, n_bins + 1):
        bin_data = df_sorted[df_sorted['bin'] == bin_id]
        if len(bin_data) == 0:
            continue

        # 成功次数和失败次数
        successes = bin_data['转化label'].sum()
        failures = len(bin_data) - successes
        bin_conversion_rate = successes / len(bin_data) if len(bin_data) > 0 else 0

        # 优化2: 低样本量/低转化区间特殊处理
        current_prior_alpha = prior_alpha
        current_prior_beta = prior_beta
        if successes < 5:  # 转化数<5的区间
            # 将先验权重降低到原来的一半
            current_prior_alpha = prior_alpha * 0.5
            current_prior_beta = prior_beta * 0.5

        # 优化4: 区间自适应的混合校准策略
        calibration_method = 'bbq'  # 默认使用BBQ
        if use_adaptive_mixing:
            # 低转化区间使用PAVA，高转化区间使用BBQ
            if bin_conversion_rate < 0.005 or (successes < 10 and len(bin_data) < 200):
                calibration_method = 'pava'
                # 对于低转化区间，直接使用样本转化率作为校准值（PAVA思路）
                posterior_mean = bin_conversion_rate
            else:
                # 高转化区间使用BBQ
                posterior_alpha = current_prior_alpha + successes
                posterior_beta = current_prior_beta + failures
                posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        else:
            # 正常BBQ计算
            posterior_alpha = current_prior_alpha + successes
            posterior_beta = current_prior_beta + failures
            posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

        # 桶内pcvr的最小值、最大值和平均值
        bin_pcvr_min = bin_data['pcvr'].min()
        bin_pcvr_max = bin_data['pcvr'].max()
        bin_pcvr_mean = bin_data['pcvr'].mean()

        bin_stats.append({
            'bin_id': bin_id,
            'pcvr_min': bin_pcvr_min,
            'pcvr_max': bin_pcvr_max,
            'pcvr_mean': bin_pcvr_mean,
            'calibrated_cvr': posterior_mean,
            'count': len(bin_data),
            'successes': successes,
            'failures': failures,
            'posterior_alpha': posterior_alpha if calibration_method == 'bbq' else None,
            'posterior_beta': posterior_beta if calibration_method == 'bbq' else None,
            'prior_alpha': current_prior_alpha,
            'prior_beta': current_prior_beta,
            'calibration_method': calibration_method  # 记录使用的校准方法
        })

    bins_df = pd.DataFrame(bin_stats)
    print(f"Bayesian Binning完成，共 {len(bins_df)} 个有效桶，使用先验alpha={prior_alpha:.4f}, beta={prior_beta:.4f}")

    return bins_df, bin_edges

def calibrate_with_fallback(df, fallback_config, n_bins=100, min_samples=100, modelname='pava', adaptive_threshold=True, use_adaptive_mixing=False):
    """
    使用回退策略进行校准

    参数:
    df: 原始数据框
    fallback_config: adid回退配置字典
    n_bins: 分桶数量
    min_samples: 最小置信样本数量
    modelname: 模型类型，'pava'或'bbq'
    adaptive_threshold: 是否使用自适应阈值

    返回:
    result_df: 包含校准后cvr的数据框
    calibration_funcs: 校准函数字典，格式为{(维度, 键值): 函数}
    calibration_data: 校准数据字典
    fallback_decisions: 回退决策记录
    threshold_stats: 阈值使用统计信息
    """
    result_df = df.copy()
    result_df['calibrated_cvr'] = df['pcvr'].copy()

    # 校准函数字典，格式为{(维度, 键值): 函数}
    calibration_funcs = {}
    # 校准数据字典
    calibration_data = {}
    # 回退决策记录
    fallback_decisions = {}
    # 阈值统计信息
    threshold_stats = {
        'total_adids': 0,
        'low_confidence_count': 0,
        'threshold_distribution': {},
        'adaptive_decisions': 0,
        'best_dimension_selected': 0
    }

    # 默认维度列表，当adid没有配置时使用
    default_dimensions = DIMENSIONS

    # 1. 预先计算各个维度的样本数量
    dimension_sample_counts = {}
    all_dimensions = set(DIMENSIONS)

    for dimension in all_dimensions:
        dimension_sample_counts[dimension] = {}
        if dimension in df.columns:
            for key in df[dimension].unique():
                count = sum(df[dimension] == key)
                dimension_sample_counts[dimension][key] = count

    # 2. 预先计算pkg维度的Beta先验参数（基于各pkg的平均值）
    pkg_priors = {}
    if modelname == 'bbq' and 'pkg' in df.columns:
        print("\n计算pkg维度的Beta先验参数...")
        # 对每个pkg计算其平均转化率
        for pkg in df['pkg'].unique():
            pkg_data = df[df['pkg'] == pkg]
            total_samples = len(pkg_data)
            total_successes = pkg_data['转化label'].sum()
            total_failures = total_samples - total_successes

            # 计算该pkg的平均转化率
            if total_samples > 0:
                avg_conversion = total_successes / total_samples
                # 设置Beta先验参数
                # 优化1: 降低基础先验权重，对低转化率包进一步降低
                if avg_conversion < 0.01:  # 低转化率包
                    base = 3  # 进一步降低先验权重
                else:
                    base = 5  # 降低基础先验权重（从10改为5）
                pkg_priors[pkg] = {
                    'alpha': base * avg_conversion + 0.5,  # 优化1: 减小避免0值的常数（从+1改为+0.5）
                    'beta': base * (1 - avg_conversion) + 0.5
                }
                print(f"  pkg={pkg}: 样本数={total_samples}, 平均转化率={avg_conversion:.6f}, alpha={pkg_priors[pkg]['alpha']:.4f}, beta={pkg_priors[pkg]['beta']:.4f}")

        # 如果某个pkg没有数据，使用全局平均值
        global_samples = len(df)
        global_successes = df['转化label'].sum()
        global_avg_conversion = global_successes / global_samples if global_samples > 0 else 0.05
        # 优化1: 使用调整后的base和避免0值的常数
        default_prior = {
            'alpha': base * global_avg_conversion + 0.5,
            'beta': base * (1 - global_avg_conversion) + 0.5
        }
        print(f"  全局默认先验: alpha={default_prior['alpha']:.4f}, beta={default_prior['beta']:.4f}")

    # 优化3: 基于转化率的动态min_samples参数
    overall_conversion_rate = df['转化label'].mean() if len(df) > 0 else 0
    dynamic_min_samples = min_samples
    if overall_conversion_rate < 0.005:
        dynamic_min_samples = min_samples * 3  # 低转化率时增加min_samples
        print(f"检测到低转化率({overall_conversion_rate:.6f})，动态调整min_samples={dynamic_min_samples}")
    elif 0.005 <= overall_conversion_rate < 0.01:
        dynamic_min_samples = int(min_samples * 2)  # 中低转化率时适度增加min_samples
        print(f"检测到中低转化率({overall_conversion_rate:.6f})，动态调整min_samples={dynamic_min_samples}")
    else:
        print(f"检测到正常转化率({overall_conversion_rate:.6f})，使用原始min_samples={min_samples}")

    # 2. 对每个adid应用回退策略
    unique_adids = df['adid'].unique()
    print(f"共有 {len(unique_adids)} 个不同的adid")

    for adid in unique_adids:
        # 获取adid对应的回退维度列表
        if adid in fallback_config:
            dimensions = fallback_config[adid]
        else:
            dimensions = default_dimensions

        # 记录回退决策
        fallback_decisions[adid] = {'final_dimension': None, 'sample_sizes': {}}

        # 尝试每个维度
        for dimension in dimensions:
            if dimension not in df.columns:
                print(f"警告: 维度 {dimension} 不存在于数据中，跳过")
                continue

            # 获取当前adid在当前维度下的键值
            adid_rows = df[df['adid'] == adid]
            if len(adid_rows) == 0:
                continue

            dimension_key = adid_rows[dimension].iloc[0]

            # 获取当前维度键值的样本数量
            sample_size = dimension_sample_counts[dimension].get(dimension_key, 0)
            fallback_decisions[adid]['sample_sizes'][dimension] = sample_size

            # 获取当前维度键值的转化率信息（如果需要）
            dimension_df = df[df[dimension] == dimension_key]
            conversion_rate = None
            if len(dimension_df) > 0:
                # 使用转化label列计算转化率，而不是不存在的cvr列
                conversion_rate = dimension_df['转化label'].mean()

            # 使用动态调整后的min_samples
            current_min_samples = dynamic_min_samples
            if conversion_rate is not None:
                # 对每个维度键值也根据其转化率动态调整min_samples
                if conversion_rate < 0.005:
                    current_min_samples = min_samples * 10
                elif 0.005 <= conversion_rate < 0.01:
                    current_min_samples = int(min_samples * 5)

            # 判断样本数量是否足够置信
            is_confident, threshold, confidence = is_sample_size_confident(
                sample_size, current_min_samples, modelname, conversion_rate, dimension, adaptive_threshold
            )

            if is_confident:
                fallback_decisions[adid]['final_dimension'] = dimension
                fallback_decisions[adid]['confidence_threshold'] = threshold
                break

        # 如果所有维度都不置信，根据模型类型采用不同策略
        if fallback_decisions[adid]['final_dimension'] is None and len(dimensions) > 0:
            print(f"警告: adid {adid} 的所有维度样本数量都不足够置信，不做纠偏")
            # 找出样本量最大的维度
            # max_dimension = None
            # max_size = 0
            # for dim, size in fallback_decisions[adid]['sample_sizes'].items():
            #     if size > max_size:
            #         max_size = size
            #         max_dimension = dim

            # # 如果找到样本量最大的维度，优先使用它
            # if max_dimension:
            #     fallback_decisions[adid]['final_dimension'] = max_dimension
            #     fallback_decisions[adid]['is_low_confidence'] = True
            #     fallback_decisions[adid]['low_confidence_strategy'] = 'max_sample_size'
            #     print(f"警告: adid {adid} 的所有维度样本数量都不足够置信，使用样本量最大的维度 {max_dimension} (样本量: {max_size})") #!!!,从adid所有维度中选择样本量最大的维度
            # else:
            #     # 否则使用最后一个维度作为备选
            #     fallback_decisions[adid]['final_dimension'] = dimensions[-1]
            #     fallback_decisions[adid]['is_low_confidence'] = True
            #     fallback_decisions[adid]['low_confidence_strategy'] = 'last_dimension'
            #     print(f"警告: adid {adid} 的所有维度样本数量都不足够置信，使用最后一个维度 {dimensions[-1]}")

            # # 对于BBQ模型，可以记录需要更强先验的标记
            # if modelname == 'bbq':
            #     fallback_decisions[adid]['need_stronger_prior'] = True

    # 3. 按维度和键值分组训练校准函数
    for dimension in all_dimensions:
        if dimension not in df.columns:
            continue

        print(f"\n处理维度: {dimension}")
        dimension_keys = df[dimension].unique()

        for key in dimension_keys:
            # 获取当前维度键值的数据
            key_df = df[df[dimension] == key].copy()

            if len(key_df) <= 20:
                print(f"  {dimension}={key} 样本数量不足，跳过")
                continue

            if modelname == 'bbq':
                # 获取当前数据对应的pkg，用于确定先验参数
                if 'pkg' in key_df.columns and not key_df['pkg'].empty:
                    # 如果当前数据属于某个pkg，使用该pkg的先验参数
                    current_pkg = key_df['pkg'].iloc[0]
                    prior_params = pkg_priors.get(current_pkg, default_prior)
                    prior_alpha = prior_params['alpha']
                    prior_beta = prior_params['beta']
                else:
                    # 否则使用全局默认先验
                    prior_alpha = default_prior['alpha']
                    prior_beta = default_prior['beta']

                # 使用Bayesian Binning into Quantiles，传入基于pkg的先验参数
                # 优化4: 使用区间自适应的混合校准策略
                bins_df, _ = bayesian_binning_into_quantiles(key_df, n_bins, prior_alpha, prior_beta, use_adaptive_mixing=use_adaptive_mixing)

                # 提取pcvr和校准后的cvr序列
                pcvr_seq = bins_df['pcvr_mean'].values
                cal_cvr_seq = bins_df['calibrated_cvr'].values

                # 创建插值函数
                calibration_func = create_interpolation_function(pcvr_seq, cal_cvr_seq)
                # 保存校准函数和数据
                calibration_funcs[(dimension, key)] = calibration_func
                calibration_data[(dimension, key)] = {
                    'original': bins_df,
                    'modelname': 'bbq',
                    'sample_size': len(key_df),
                    'prior_alpha': prior_alpha,
                    'prior_beta': prior_beta,
                    'compressed': {'pcvr': pcvr_seq, 'cvr': cal_cvr_seq},
                    'pkg': current_pkg if 'pkg' in locals() else None
                }
                print(f"  {dimension}={key} 校准完成，样本数: {len(key_df)}")
            elif modelname == 'platt':
                # 使用Platt校准方法
                # 获取pcvr和真实转化标签
                pcvr_seq = key_df['pcvr'].values
                real_cvr = key_df['转化label'].values

                # 将真实cvr转换为二分类标签（0或1）
                # 这里假设real_cvr是0或1的数组，如果是转化率则需要根据实际情况处理
                # targets = np.array(real_cvr >= 0.5, dtype=int)
                targets = real_cvr

                # 创建并训练Platt校准器
                calibrator = PlattCalibrator(verbose=True)
                calibrator.fit(pcvr_seq, targets)

                # 保存校准器作为校准函数
                calibration_func = calibrator.predict
                # 保存校准函数和数据
                calibration_funcs[(dimension, key)] = calibration_func
                # 保存校准数据
                calibration_data[(dimension, key)] = {
                    'original': key_df,
                    'platt_params': {'A': calibrator.A, 'B': calibrator.B},
                    'modelname': 'platt',
                    'sample_size': len(key_df)
                }
                print(f"  {dimension}={key} 校准完成，样本数: {len(key_df)}")
            else:
                # 使用传统的PAVA方法
                # 1-2. 排序和分桶
                bins_df = equal_frequency_binning(key_df, n_bins)

                # 提取pcvr和cvr序列
                pcvr_seq = bins_df['pcvr'].values
                cvr_seq = bins_df['cvr'].values
                weights = bins_df['count'].values

                # 3. 执行PAVA算法
                post_cvr = pava_algorithm(pcvr_seq, cvr_seq, weights)

                # 4. 样本压缩
                compressed_pcvr, compressed_cvr = compress_samples(pcvr_seq, post_cvr)

                # 5. 创建插值函数
                calibration_func = create_interpolation_function(compressed_pcvr, compressed_cvr)
                # 保存校准函数和数据
                calibration_funcs[(dimension, key)] = calibration_func
                calibration_data[(dimension, key)] = {
                    'original': key_df,
                    'pava': {'pcvr': pcvr_seq, 'post_cvr': post_cvr},
                    'compressed': {'pcvr': compressed_pcvr, 'cvr': compressed_cvr},
                    'modelname': 'pava',
                    'sample_size': len(key_df)
                }
                print(f"  {dimension}={key} 校准完成，样本数: {len(key_df)}, 压缩后样本点数: {len(compressed_pcvr)}")

    # 4. 应用校准函数到原始数据
    for idx, row in result_df.iterrows():
        adid = row['adid']

        # 获取adid对应的最终维度
        final_dimension = fallback_decisions.get(adid, {}).get('final_dimension')

        if final_dimension is None:
            # 如果没有配置，保持原值
            continue

        # 获取维度对应的键值
        dimension_key = get_dimension_key(row, final_dimension)
        print(f"adid={adid},使用维度:{final_dimension}, 维度值:{dimension_key}")
        if dimension_key is None:
            continue

        # 获取对应的校准函数
        calibration_func = calibration_funcs.get((final_dimension, dimension_key))

        if calibration_func is not None:
            # 应用校准函数
            result_df.loc[idx, 'calibrated_cvr'] = calibration_func(row['pcvr'])

            # 记录使用的维度
            result_df.loc[idx, 'used_dimension'] = final_dimension
        else:
            print(f"adid={adid}, calibration_func is None")

    return result_df, calibration_funcs, calibration_data, fallback_decisions

def evaluate_calibration(df, dimension='overall', metrics=None, modelname='pava', min_samples=None):
    """
    评估校准效果

    参数:
    df: 包含原始pcvr和校准后cvr的数据框
    dimension: 评估维度
    metrics: 要计算的指标字典
    modelname: 模型类型
    min_samples: 使用的最小样本量阈值（用于更新阈值管理器）

    返回:
    metrics: 评估指标字典
    """
    # 计算均方误差
    mse_original = np.mean((df['pcvr'] - df['转化label'])**2)
    mse_calibrated = np.mean((df['calibrated_cvr'] - df['转化label'])**2)

    # 计算对数损失
    eps = 1e-15  # 防止log(0)
    logloss_original = -np.mean(
        df['转化label'] * np.log(df['pcvr'].clip(eps, 1-eps)) +
        (1 - df['转化label']) * np.log(1 - df['pcvr'].clip(eps, 1-eps))
    )
    logloss_calibrated = -np.mean(
        df['转化label'] * np.log(df['calibrated_cvr'].clip(eps, 1-eps)) +
        (1 - df['转化label']) * np.log(1 - df['calibrated_cvr'].clip(eps, 1-eps))
    )

    # 计算ECE (Expected Calibration Error)
    ece_original = calculate_ece(df['pcvr'].values, df['转化label'].values)
    ece_calibrated = calculate_ece(df['calibrated_cvr'].values, df['转化label'].values)

    # 按使用的维度分组计算指标
    dimension_metrics = {}
    if 'used_dimension' in df.columns:
        for dimension in df['used_dimension'].dropna().unique():
            dim_df = df[df['used_dimension'] == dimension]

            dim_mse_original = np.mean((dim_df['pcvr'] - dim_df['转化label'])**2)
            dim_mse_calibrated = np.mean((dim_df['calibrated_cvr'] - dim_df['转化label'])**2)

            dim_logloss_original = -np.mean(
                dim_df['转化label'] * np.log(dim_df['pcvr'].clip(eps, 1-eps)) +
                (1 - dim_df['转化label']) * np.log(1 - dim_df['pcvr'].clip(eps, 1-eps))
            )
            dim_logloss_calibrated = -np.mean(
                dim_df['转化label'] * np.log(dim_df['calibrated_cvr'].clip(eps, 1-eps)) +
                (1 - dim_df['转化label']) * np.log(1 - dim_df['calibrated_cvr'].clip(eps, 1-eps))
            )

            # 计算维度特定的ECE
            dim_ece_original = calculate_ece(dim_df['pcvr'].values, dim_df['转化label'].values)
            dim_ece_calibrated = calculate_ece(dim_df['calibrated_cvr'].values, dim_df['转化label'].values)

            dim_samples = len(dim_df)
            dim_result = {
                'count': dim_samples,
                'MSE_Original': dim_mse_original,
                'MSE_Calibrated': dim_mse_calibrated,
                'LogLoss_Original': dim_logloss_original,
                'LogLoss_Calibrated': dim_logloss_calibrated,
                'ECE_Original': dim_ece_original,
                'ECE_Calibrated': dim_ece_calibrated
            }

            # 评估维度的置信度
            if min_samples is not None:
                is_confident, _,_ = is_sample_size_confident(dim_samples, min_samples, modelname)
                dim_result['is_confident'] = is_confident

            dimension_metrics[dimension] = dim_result

    metrics = {
        'Overall': {
            'count': len(df),
            'MSE_Original': mse_original,
            'MSE_Calibrated': mse_calibrated,
            'LogLoss_Original': logloss_original,
            'LogLoss_Calibrated': logloss_calibrated,
            'ECE_Original': ece_original,
            'ECE_Calibrated': ece_calibrated,
            'coverage': 1.0,  # 默认100%覆盖率
            'sample_size': len(df)
        },
        'ByDimension': dimension_metrics
    }

    print("\n评估指标:")
    print(f"总体指标 (样本数: {len(df)}):")
    print(f"  MSE_Original: {mse_original:.6f}")
    print(f"  MSE_Calibrated: {mse_calibrated:.6f}")
    print(f"  LogLoss_Original: {logloss_original:.6f}")
    print(f"  LogLoss_Calibrated: {logloss_calibrated:.6f}")
    print(f"  ECE_Original: {ece_original:.6f}")
    print(f"  ECE_Calibrated: {ece_calibrated:.6f}")

    if dimension_metrics:
        print("\n按维度分组的指标:")
        for dimension, dim_metrics in dimension_metrics.items():
            print(f"  {dimension} (样本数: {dim_metrics['count']}):")
            print(f"    MSE_Original: {dim_metrics['MSE_Original']:.6f}")
            print(f"    MSE_Calibrated: {dim_metrics['MSE_Calibrated']:.6f}")
            print(f"    LogLoss_Original: {dim_metrics['LogLoss_Original']:.6f}")
            print(f"    LogLoss_Calibrated: {dim_metrics['LogLoss_Calibrated']:.6f}")
            print(f"    ECE_Original: {dim_metrics['ECE_Original']:.6f}")
            print(f"    ECE_Calibrated: {dim_metrics['ECE_Calibrated']:.6f}")

    # 如果提供了min_samples参数，更新阈值管理器
    if min_samples is not None:
        try:
            # 这里可以添加阈值管理器的更新逻辑
            pass
        except Exception as e:
            print(f"警告: 更新阈值管理器失败: {e}")

    return metrics

def save_calibration_model(calibration_funcs, calibration_data, fallback_decisions, file_path):
    """
    保存校准模型

    参数:
    calibration_funcs: 校准函数字典
    calibration_data: 校准数据字典
    fallback_decisions: 回退决策记录
    file_path: 保存路径
    """
    model = {
        'calibration_data': calibration_data,
        'fallback_decisions': fallback_decisions
    }
    # 不能直接序列化函数，所以只保存数据
    with open(file_path+".pkl", 'wb') as f:
        pickle.dump(model, f)

    processed_count = 0  # 初始化处理计数
    with open(file_path+".txt", 'w', encoding='utf-8') as out_f:
        print(f"正在生成校准词典: {file_path}.txt")

        # 遍历每个维度和键
        for (dimension, key), data in calibration_data.items():
            if data["modelname"] == "platt":
                A = data["platt_params"]["A"]
                B = data["platt_params"]["B"]
                out_line = f"{PREFIX}{key}{SOH}{A};{B}"
                out_f.write(out_line + '\n')
                processed_count += 1
                continue
            elif data["modelname"] == "bbq":
                # BBQ格式数据（包含bins_df）
                bins_df = data['original']
                # 从bins_df中提取pcvr边界和校准后的cvr
                comp_pcvr = bins_df['pcvr_mean'].tolist()
                # 对于BBQ，使用calibrated_cvr作为校准后的值
                comp_cvr = bins_df['calibrated_cvr'].tolist()
            elif data["modelname"] == "pava":
                # 处理不同格式的校准数据
                comp_pcvr = data['compressed']['pcvr']
                comp_cvr = data['compressed']['cvr']
            else:
                # 跳过不支持的数据格式
                print(f"警告: 跳过不支持的数据格式 for {(dimension, key)}")
                continue
            # 确保数据有效
            if len(comp_pcvr) == 0 or len(comp_cvr) == 0:
                print(f"警告: 空数据 for {(dimension, key)}")
                continue
            #补充边界
            if comp_pcvr[0] > 0:
                comp_pcvr = np.concatenate(([0], comp_pcvr))
                comp_cvr = np.concatenate(([0], comp_cvr))
            if comp_pcvr[-1] < 1:
                comp_pcvr = np.concatenate((comp_pcvr, [PCVR_MAX]))
                comp_cvr = np.concatenate((comp_cvr, [PCVR_MAX]))

            # 处理pcvr和cvr长度不一致的情况
            if len(comp_pcvr) != len(comp_cvr):
                # 如果pcvr比cvr多一个（表示边界），只使用前len(comp_cvr)个
                print(f"警告: len(comp_pcvr) != len(comp_cvr)  for {(dimension, key)}")
                continue

            # 计算纠偏系数k = comp_cvr / comp_pcvr
            # 避免除以0的情况
            # k_values = []
            # for i in range(len(comp_pcvr)):
            #     if comp_pcvr[i] > 0:
            #         k = comp_cvr[i] / comp_pcvr[i]
            #     else:
            #         k = 1.0  # 默认值
            #     k_values.append(k)

            # 构建pcvr边界字符串（逗号分隔，保留所有值）
            pcvr_str = ','.join([f"{val:.8f}" for val in comp_pcvr])

            # 构建cvr边界字符串（逗号分隔，保留所有值）
            cvr_str = ','.join([f"{val:.8f}" for val in comp_cvr])

            # 构建完整的输出行
            out_line = f"{PREFIX}{key}{SOH}{pcvr_str};{cvr_str}"
            out_f.write(out_line + '\n')
            processed_count += 1

        print(f"校准词典生成完成，共处理 {processed_count} 条有效记录，总数据项数 {len(calibration_data)}")


    print(f"校准模型已保存到 {file_path}")

def load_calibration_model(file_path):
    """
    加载校准模型

    参数:
    file_path: 模型文件路径

    返回:
    calibration_funcs: 校准函数字典
    calibration_准数据字典
    fallback_decisions: 回退决策记录
    """
    with open(file_path, 'rb') as f:
        model = pickle.load(f)

    calibration_data = model['calibration_data']
    fallback_decisions = model['fallback_decisions']

   # 重新创建校准函数
    calibration_funcs = {}
    for (dimension, key), data in calibration_data.items():
        if data['modelname'] == 'platt':
            # 处理Platt校准模型
            calibrator = PlattCalibrator()
            calibrator.A = data['platt_params']['A']
            calibrator.B = data['platt_params']['B']
            calibrator.fitted = True
            calibration_funcs[(dimension, key)] = calibrator.predict
        elif data['modelname'] == 'bbq':
            # 处理BBQ校准模型
            if 'original' in data:
                bins_df = data['original']
                comp_pcvr = bins_df['pcvr_mean'].tolist()
                comp_cvr = bins_df['calibrated_cvr'].tolist()
                calibration_funcs[(dimension, key)] = create_interpolation_function(comp_pcvr, comp_cvr)
        else:
            # 处理PAVA校准模型
            if 'compressed' in data:
                comp_pcvr = data['compressed']['pcvr']
                comp_cvr = data['compressed']['cvr']
            elif 'original' in data:
                bins_df = data['original']
                comp_pcvr = bins_df['pcvr_mean'].tolist() if 'pcvr_mean' in bins_df.columns else bins_df['pcvr'].tolist()
                comp_cvr = bins_df['calibrated_cvr'].tolist() if 'calibrated_cvr' in bins_df.columns else bins_df['cvr'].tolist()
            calibration_funcs[(dimension, key)] = create_interpolation_function(comp_pcvr, comp_cvr)

    print(f"校准函数数量{len(calibration_funcs)}")
    print(f"校准模型已从 {file_path} 加载")

    return calibration_funcs, calibration_data, fallback_decisions

def apply_calibration_to_new_data(test_df, calibration_funcs, fallback_decisions, modelname='pava'):
    """
    应用校准函数到新数据

    参数:
    test_df: 测试数据框
    calibration_funcs: 校准函数字典
    fallback_decisions: 回退决策记录
    modelname: 模型类型，'pava'或'bbq'

    返回:
    result_df: 包含校准后cvr的结果数据框
    """
    result_df = test_df.copy()
    result_df['calibrated_cvr'] = test_df['pcvr'].copy()

    for idx, row in result_df.iterrows():
        adid = row['adid']

        # 获取adid对应的最终维度
        final_dimension = fallback_decisions.get(adid, {}).get('final_dimension')

        if final_dimension is None:
            # 如果没有配置，保持原值
            # final_dimension = DIMENSIONS[0]
            continue

        # 获取维度对应的键值
        dimension_key = get_dimension_key(row, final_dimension)

        if dimension_key is None:
            print(f"adid={adid} no dimension key")
            continue

        # 获取对应的校准函数
        calibration_func = calibration_funcs.get((final_dimension, dimension_key))

        if calibration_func is not None:
            # 应用校准函数
            result_df.loc[idx, 'calibrated_cvr'] = calibration_func(row['pcvr'])

            # 记录使用的维度
            result_df.loc[idx, 'used_dimension'] = final_dimension
        else:
            print(f"adid={adid} no calibration func")

    # BBQ模型特定的后处理
    # if modelname == 'bbq':
    #     print("应用BBQ模型特定的后处理...")
        # 这里可以添加BBQ模型特有的处理逻辑
        # 例如基于预测值的不确定性进行调整等
        # 目前保持基本校准结果不变

            # 对于bbq模型，可以添加额外的处理
            # 这里可以根据需要添加bbq特有的后处理逻辑
            # 例如：根据预测值的不确定性进行调整
        #pass

    return result_df

def evaluate_calibration_with_field_ece(df, fields="pkg", n_bins=10):
    """
    评估校准效果，包括Field-ECE指标

    参数:
    df: 包含原始pcvr和校准后cvr的数据框
    fields: 要计算Field-ECE的字段列表，默认为None（使用所有可用字段）
    n_bins: 计算ECE时的分桶数量

    返回:
    metrics: 评估指标字典
    """
    # 如果未指定字段，使用所有可用字段
    if fields is None:
        fields = ['adid', 'adgroupid', 'pkg', '二级类目']
        if 'cvtype' in df.columns:
            fields.append('cvtype')

    # 确保所有字段都存在于数据框中
    fields = [f for f in fields if f in df.columns]

    # 计算均方误差
    mse_original = np.mean((df['pcvr'] - df['转化label'])**2)
    mse_calibrated = np.mean((df['calibrated_cvr'] - df['转化label'])**2)

    # 计算对数损失
    eps = 1e-15  # 防止log(0)
    logloss_original = -np.mean(
        df['转化label'] * np.log(df['pcvr'].clip(eps, 1-eps)) +
        (1 - df['转化label']) * np.log(1 - df['pcvr'].clip(eps, 1-eps))
    )
    logloss_calibrated = -np.mean(
        df['转化label'] * np.log(df['calibrated_cvr'].clip(eps, 1-eps)) +
        (1 - df['转化label']) * np.log(1 - df['calibrated_cvr'].clip(eps, 1-eps))
    )

    # 计算整体ECE (Expected Calibration Error)
    ece_original = calculate_ece(df['pcvr'].values, df['转化label'].values, n_bins)
    ece_calibrated = calculate_ece(df['calibrated_cvr'].values, df['转化label'].values, n_bins)

    # 计算Field-ECE
    field_ece_original = {}
    field_ece_calibrated = {}

    for field in fields:
        field_ece_original[field] = calculate_field_ece(
            df, field, 'pcvr', '转化label', n_bins
        )
        field_ece_calibrated[field] = calculate_field_ece(
            df, field, 'calibrated_cvr', '转化label', n_bins
        )

    # 按使用的维度分组计算指标
    dimension_metrics = {}
    if 'used_dimension' in df.columns:
        for dimension in df['used_dimension'].dropna().unique():
            dim_df = df[df['used_dimension'] == dimension]

            dim_mse_original = np.mean((dim_df['pcvr'] - dim_df['转化label'])**2)
            dim_mse_calibrated = np.mean((dim_df['calibrated_cvr'] - dim_df['转化label'])**2)

            dim_logloss_original = -np.mean(
                dim_df['转化label'] * np.log(dim_df['pcvr'].clip(eps, 1-eps)) +
                (1 - dim_df['转化label']) * np.log(1 - dim_df['pcvr'].clip(eps, 1-eps))
            )
            dim_logloss_calibrated = -np.mean(
                dim_df['转化label'] * np.log(dim_df['calibrated_cvr'].clip(eps, 1-eps)) +
                (1 - dim_df['转化label']) * np.log(1 - dim_df['calibrated_cvr'].clip(eps, 1-eps))
            )

            # 计算该维度下的ECE
            dim_ece_original = calculate_ece(dim_df['pcvr'].values, dim_df['转化label'].values, n_bins)
            dim_ece_calibrated = calculate_ece(dim_df['calibrated_cvr'].values, dim_df['转化label'].values, n_bins)

            dimension_metrics[dimension] = {
                'count': len(dim_df),
                'MSE_Original': dim_mse_original,
                'MSE_Calibrated': dim_mse_calibrated,
                'LogLoss_Original': dim_logloss_original,
                'LogLoss_Calibrated': dim_logloss_calibrated,
                'ECE_Original': dim_ece_original,
                'ECE_Calibrated': dim_ece_calibrated
            }

    metrics = {
        'Overall': {
            'count': len(df),
            'MSE_Original': mse_original,
            'MSE_Calibrated': mse_calibrated,
            'LogLoss_Original': logloss_original,
            'LogLoss_Calibrated': logloss_calibrated,
            'ECE_Original': ece_original,
            'ECE_Calibrated': ece_calibrated
        },
        'FieldECE_Original': field_ece_original,
        'FieldECE_Calibrated': field_ece_calibrated,
        'ByDimension': dimension_metrics
    }

    # 打印评估结果
    print("\n评估指标:")
    print(f"总体指标 (样本数: {len(df)}):")
    print(f"  MSE_Original: {mse_original:.6f}")
    print(f"  MSE_Calibrated: {mse_calibrated:.6f}")
    print(f"  LogLoss_Original: {logloss_original:.6f}")
    print(f"  LogLoss_Calibrated: {logloss_calibrated:.6f}")
    print(f"  ECE_Original: {ece_original:.6f}")
    print(f"  ECE_Calibrated: {ece_calibrated:.6f}")

    print("\nField-ECE指标:")
    for field in fields:
        print(f"  {field}:")
        print(f"    Original: {field_ece_original[field]:.6f}")
        print(f"    Calibrated: {field_ece_calibrated[field]:.6f}")
        print(f"    改进: {field_ece_original[field] - field_ece_calibrated[field]:.6f}")

    if dimension_metrics:
        print("\n按维度分组的指标:")
        for dimension, dim_metrics in dimension_metrics.items():
            print(f"  {dimension} (样本数: {dim_metrics['count']}):")
            print(f"    MSE_Original: {dim_metrics['MSE_Original']:.6f}")
            print(f"    MSE_Calibrated: {dim_metrics['MSE_Calibrated']:.6f}")
            print(f"    LogLoss_Original: {dim_metrics['LogLoss_Original']:.6f}")
            print(f"    LogLoss_Calibrated: {dim_metrics['LogLoss_Calibrated']:.6f}")
            print(f"    ECE_Original: {dim_metrics['ECE_Original']:.6f}")
            print(f"    ECE_Calibrated: {dim_metrics['ECE_Calibrated']:.6f}")

    return metrics

def calculate_ece(predictions, labels, n_bins=10):
    """
    计算Expected Calibration Error (ECE)

    参数:
    predictions: 预测概率值数组
    labels: 实际标签数组（0或1）
    n_bins: 分桶数量

    返回:
    ece: Expected Calibration Error值
    """
    # 创建等宽分桶
    bin_boundaries = np.linspace(0, 0.5, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # 将预测值分配到桶中
    confidences = predictions
    accuracies = labels  # 二分类问题中，准确率等于标签值

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 确定落在当前桶中的预测
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)

        if np.sum(in_bin) > 0:
            # 计算桶中的平均置信度和准确率
            bin_conf = np.mean(confidences[in_bin])
            bin_acc = np.mean(accuracies[in_bin])

            # 计算加权绝对误差
            weight = np.sum(in_bin) / len(confidences)
            ece += weight * np.abs(bin_conf - bin_acc)

    return ece

def calculate_field_ece(df, field, pred_col, label_col, n_bins=10):
    """
    计算Field-ECE (按字段分组的Expected Calibration Error)

    参数:
    df: 数据框
    field: 分组字段名
    pred_col: 预测列名
    label_col: 标签列名
    n_bins: 分桶数量

    返回:
    field_ece: Field-ECE值
    """
    # 获取字段的唯一值
    field_values = df[field].unique()

    weighted_ece_sum = 0.0
    total_samples = len(df)

    for value in field_values:
        # 获取当前字段值对应的样本
        value_df = df[df[field] == value]
        value_samples = len(value_df)

        if value_samples > 0:
            # 计算该字段值下的ECE
            value_ece = calculate_ece(
                value_df[pred_col].values,
                value_df[label_col].values,
                n_bins
            )

            # 按样本数量加权
            weight = value_samples / total_samples
            weighted_ece_sum += weight * value_ece

    return weighted_ece_sum

def visualize_calibration_curves(df, post_suffix="train", fields=None, n_bins=10):
    """
    可视化校准曲线，包括按字段分组的校准曲线

    参数:
    df: 包含原始pcvr和校准后cvr的数据框
    fields: 要可视化的字段列表，默认为None（使用所有可用字段）
    n_bins: 分桶数量
    """
    # 如果未指定字段，使用所有可用字段
    if fields is None:
        fields = ['adid', 'adgroupid', 'pkg', '二级类目']
        if 'cvtype' in df.columns:
            fields.append('cvtype')

    # 确保所有字段都存在于数据框中
    fields = [f for f in fields if f in df.columns]

    # 设置图表风格
    sns.set(style="whitegrid")

    # 1. 整体校准曲线
    plt.figure(figsize=(12, 8))

    # 创建等宽分桶
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    # 原始预测的校准曲线
    bin_accs_orig = []
    bin_confs_orig = []
    bin_sizes_orig = []

    # 校准后预测的校准曲线
    bin_accs_cal = []
    bin_confs_cal = []
    bin_sizes_cal = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]

        # 原始预测
        in_bin_orig = np.logical_and(df['pcvr'] > bin_lower, df['pcvr'] <= bin_upper)
        if np.sum(in_bin_orig) > 0:
            bin_conf_orig = np.mean(df.loc[in_bin_orig, 'pcvr'])
            bin_acc_orig = np.mean(df.loc[in_bin_orig, '转化label'])
            bin_size_orig = np.sum(in_bin_orig)

            bin_accs_orig.append(bin_acc_orig)
            bin_confs_orig.append(bin_conf_orig)
            bin_sizes_orig.append(bin_size_orig)
        else:
            bin_accs_orig.append(0)
            bin_confs_orig.append(bin_centers[i])
            bin_sizes_orig.append(0)

        # 校准后预测
        in_bin_cal = np.logical_and(df['calibrated_cvr'] > bin_lower, df['calibrated_cvr'] <= bin_upper)
        if np.sum(in_bin_cal) > 0:
            bin_conf_cal = np.mean(df.loc[in_bin_cal, 'calibrated_cvr'])
            bin_acc_cal = np.mean(df.loc[in_bin_cal, '转化label'])
            bin_size_cal = np.sum(in_bin_cal)

            bin_accs_cal.append(bin_acc_cal)
            bin_confs_cal.append(bin_conf_cal)
            bin_sizes_cal.append(bin_size_cal)
        else:
            bin_accs_cal.append(0)
            bin_confs_cal.append(bin_centers[i])
            bin_sizes_cal.append(0)

    # 绘制整体校准曲线
    plt.subplot(2, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(bin_confs_orig, bin_accs_orig, 'bo-', label=f'Original (ECE={calculate_ece(df["pcvr"].values, df["转化label"].values, n_bins):.4f})')
    plt.plot(bin_confs_cal, bin_accs_cal, 'ro-', label=f'Calibrated (ECE={calculate_ece(df["calibrated_cvr"].values, df["转化label"].values, n_bins):.4f})')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Frequency')
    plt.title('Overall Calibration Curve')
    plt.legend()
    plt.grid(True)

    # 2. 绘制样本分布
    plt.subplot(2, 2, 2)
    bin_sizes_orig = np.array(bin_sizes_orig)
    bin_sizes_cal = np.array(bin_sizes_cal)

    x = np.arange(n_bins)
    width = 0.35

    plt.bar(x - width/2, bin_sizes_orig, width, label='Original')
    plt.bar(x + width/2, bin_sizes_cal, width, label='Calibrated')
    plt.xlabel('Confidence Bin')
    plt.ylabel('Sample Count')
    plt.title('Sample Distribution')
    plt.xticks(x, [f'{bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f}' for i in range(n_bins)], rotation=45)
    plt.legend()

    # 3. 字段ECE比较
    plt.subplot(2, 2, 3)
    field_ece_orig = [calculate_field_ece(df, field, 'pcvr', '转化label', n_bins) for field in fields]
    field_ece_cal = [calculate_field_ece(df, field, 'calibrated_cvr', '转化label', n_bins) for field in fields]

    x = np.arange(len(fields))
    width = 0.35

    plt.bar(x - width/2, field_ece_orig, width, label='Original')
    plt.bar(x + width/2, field_ece_cal, width, label='Calibrated')
    plt.xlabel('Field')
    plt.ylabel('Field-ECE')
    plt.title('Field-ECE Comparison')
    plt.xticks(x, fields)
    plt.legend()

    # 4. 按使用的维度分组的ECE
    if 'used_dimension' in df.columns:
        plt.subplot(2, 2, 4)
        dimensions = df['used_dimension'].dropna().unique()

        dim_ece_orig = []
        dim_ece_cal = []
        dim_names = []

        for dimension in dimensions:
            dim_df = df[df['used_dimension'] == dimension]
            if len(dim_df) > 0:
                dim_ece_orig.append(calculate_ece(dim_df['pcvr'].values, dim_df['转化label'].values, n_bins))
                dim_ece_cal.append(calculate_ece(dim_df['calibrated_cvr'].values, dim_df['转化label'].values, n_bins))
                dim_names.append(dimension)

        x = np.arange(len(dim_names))
        width = 0.35

        plt.bar(x - width/2, dim_ece_orig, width, label='Original')
        plt.bar(x + width/2, dim_ece_cal, width, label='Calibrated')
        plt.xlabel('Used Dimension')
        plt.ylabel('ECE')
        plt.title('ECE by Used Dimension')
        plt.xticks(x, dim_names)
        plt.legend()

    plt.tight_layout()
    plt.savefig(post_suffix +'_calibration_curves_with_field_ece.png', dpi=300)
    plt.show()

def visualize_field_ece_details(df, field, prefix="train", n_bins=10):
    """
    可视化特定字段的ECE详情

    参数:
    df: 包含原始pcvr和校准后cvr的数据框
    field: 要可视化的字段
    n_bins: 分桶数量
    """
    if field not in df.columns:
        print(f"字段 {field} 不存在于数据中")
        return

    # 获取字段的唯一值
    field_values = df[field].unique()

    # 限制显示的值数量，避免图表过于拥挤
    max_values_to_show = 10
    if len(field_values) > max_values_to_show:
        # 按样本数量排序，选择样本数最多的几个值
        value_counts = df[field].value_counts()
        field_values = value_counts.index[:max_values_to_show].tolist()
        print(f"字段 {field} 有 {len(value_counts)} 个唯一值，只显示样本数最多的 {max_values_to_show} 个")

    # 计算每个值的ECE
    value_ece_orig = []
    value_ece_cal = []
    value_names = []
    value_counts = []

    for value in field_values:
        value_df = df[df[field] == value]
        value_count = len(value_df)

        if value_count > 0:
            ece_orig = calculate_ece(value_df['pcvr'].values, value_df['转化label'].values, n_bins)
            ece_cal = calculate_ece(value_df['calibrated_cvr'].values, value_df['转化label'].values, n_bins)

            value_ece_orig.append(ece_orig)
            value_ece_cal.append(ece_cal)
            value_names.append(str(value))
            value_counts.append(value_count)

    # 按样本数量排序
    sorted_indices = np.argsort(value_counts)[::-1]
    value_ece_orig = [value_ece_orig[i] for i in sorted_indices]
    value_ece_cal = [value_ece_cal[i] for i in sorted_indices]
    value_names = [value_names[i] for i in sorted_indices]
    value_counts = [value_counts[i] for i in sorted_indices]

    # 设置图表风格
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))

    # 1. ECE比较
    plt.subplot(2, 1, 1)
    x = np.arange(len(value_names))
    width = 0.35

    plt.bar(x - width/2, value_ece_orig, width, label='Original')
    plt.bar(x + width/2, value_ece_cal, width, label='Calibrated')
    plt.xlabel(f'{field} Value')
    plt.ylabel('ECE')
    plt.title(f'ECE by {field} Value')
    plt.xticks(x, value_names, rotation=45)
    plt.legend()

    # 2. 样本数量
    plt.subplot(2, 1, 2)
    plt.bar(x, value_counts)
    plt.xlabel(f'{field} Value')
    plt.ylabel('Sample Count')
    plt.title(f'Sample Count by {field} Value')
    plt.xticks(x, value_names, rotation=45)

    plt.tight_layout()
    plt.savefig(f'{prefix}_field_ece_details_{field}.png', dpi=300)
    plt.show()

def get_optimal_bins(data_size, modelname='pava'):
    """
    根据数据量和模型类型获取最佳分桶数量

    参数:
    data_size: 数据量大小
    modelname: 模型类型，'pava'、'bbq'或'platt'

    返回:
    optimal_bins: 最佳分桶数量
    """
    if modelname == 'platt':
        # Platt校准不需要分桶，直接返回None
        return None
    elif modelname == 'pava':
        # PAVA模型的分桶策略：数据量越大，分桶越多，但有上限
        return max(10, min(200, data_size // 500))
    else:  # BBQ模型
        # BBQ模型的分桶策略：略少于PAVA，因为有先验信息
        return max(10, min(100, data_size // 800))

def main(file_path, config_path, test_path, n_bins=None, min_samples=100, ece_bins=10, modelname='pava', model_file='calibration_model.pkl', adaptive_threshold=True, threshold_history_file='threshold_history.json', use_adaptive_mixing=False):
    """
    主函数，执行完整的数据处理和校准流程

    参数:
    file_path: 数据文件路径
    config_path: adid回退配置文件路径
    test_path: 测试文件路径
    n_bins: 分桶数量（None时自动计算）
    min_samples: 最小置信样本数量
    ece_bins: 计算ECE时的分桶数量
    modelname: 模型类型，'pava'或'bbq'
    model_file: 校准模型保存路径
    adaptive_threshold: 是否使用自适应阈值
    threshold_history_file: 阈值历史记录文件

    返回:
    result_df: 包含校准后cvr的数据框
    calibration_funcs: 校准函数字典
    calibration_data: 校准数据字典
    fallback_decisions: 回退决策记录
    metrics: 评估指标
    threshold_stats: 阈值使用统计
    business_metrics: 业务评估指标
    """
    print("开始数据加载...")
    print("开始加载训练数据...")
    df = load_data(file_path)
    print("开始加载测试数据...")
    print(test_path)
    test_df = load_data(test_path)
    print("\n加载adid回退配置...")
    fallback_config = load_adid_fallback_config(config_path)

    # 如果未指定分桶数量，则根据数据量和模型类型自动计算
    if n_bins is None:
        data_size = len(df)
        n_bins = get_optimal_bins(data_size, modelname)
        print(f"根据数据量{data_size}和模型类型{modelname}，自动设置分桶数量为{n_bins}") #!!!,???
    else:
        print(f"使用指定的分桶数量：{n_bins}")

    print(f"\n开始带回退策略的校准 (模型: {modelname})...")
    print(f"阈值模式: {'自适应' if adaptive_threshold else '固定阈值'}")
    print(f"基础阈值: {min_samples}")

    result_df, calibration_funcs, calibration_data, fallback_decisions = calibrate_with_fallback(
        df, fallback_config, n_bins, min_samples, modelname, adaptive_threshold, use_adaptive_mixing
    )

    print("\n评估校准效果（包括Field-ECE）...")
    metrics = evaluate_calibration_with_field_ece(result_df, n_bins=ece_bins)
    #metrics = evaluate_calibration_with_field_ece(test_df, n_bins=ece_bins)

    # print("\n可视化校准曲线...")
    # visualize_calibration_curves(result_df, n_bins=ece_bins)

    # # 可视化每个字段的ECE详情
    # fields = ['adid', 'pkg', '二级类目']
    # if 'cvtype' in result_df.columns:
    #     fields.append('cvtype')

    # for field in fields:
    #     print(f"\n可视化字段 {field} 的ECE详情...")
    #     visualize_field_ece_details(result_df, field, n_bins=ece_bins)

    # print("\n分析回退决策...")
    # fallback_analysis = analyze_fallback_decisions(fallback_decisions)

    # 保存结果
    result_df.to_csv(file_path + modelname + '_train_calibrated_res_train.csv', index=False) #!!!,训练之后对自身校准后的校准值的结果保存路径
    save_calibration_model(calibration_funcs, calibration_data, fallback_decisions, model_file) #保存了纠偏前后的值，在线使用的使用判断纠偏前后的值落在哪个区间，线上进行插值

    # 保存评估指标
    with open(file_path + modelname + '_train_calibration_metrics_train.json', 'w') as f: #!!!,训练之后对自身校准的metric保存路径
        json.dump(metrics, f, indent=4, default=str)

    print("\n校准结果和评估指标已保存")
    pred(test_df, calibration_funcs, fallback_decisions, ece_bins, test_path,modelname)
    print("\n测试集校准结果和评估指标已保存")

    # 计算并保存阈值统计信息
    threshold_stats = {
        'threshold_distribution': {},
        'low_confidence_percentage': 0.0,
        'base_threshold': min_samples,
        'adaptive_threshold_used': adaptive_threshold
    }

    # 从fallback_decisions中提取统计信息
    low_conf_count = 0
    total_decisions = len(fallback_decisions)

    for adid, decision in fallback_decisions.items():
        if decision.get('is_low_confidence'):
            low_conf_count += 1

        dimension = decision.get('final_dimension')
        if dimension:
            threshold_stats['threshold_distribution'][dimension] = threshold_stats['threshold_distribution'].get(dimension, 0) + 1

    if total_decisions > 0:
        threshold_stats['low_confidence_percentage'] = (low_conf_count / total_decisions) * 100

    # 保存阈值统计
    if adaptive_threshold:
        with open(threshold_history_file, 'w') as f:
            json.dump(threshold_stats, f, indent=2, ensure_ascii=False)
        print(f"\n阈值统计已保存到 {threshold_history_file}")

    # 分析业务影响
    print("\n分析业务评估结果...")
    business_metrics = analyze_business_impact(result_df)
    print_business_metrics(business_metrics)

    # 保存业务评估指标
    with open(file_path + modelname + '_business_metrics.json', 'w') as f:
        json.dump(business_metrics, f, indent=4, default=str)
    print(f"\n业务评估指标已保存到 {file_path}_business_metrics.json")

    return result_df, calibration_funcs, calibration_data, fallback_decisions, metrics, threshold_stats, business_metrics

def pred(test_df, calibration_funcs, fallback_decisions, ece_bins, test_path, modelname='pava'):
    result_df = apply_calibration_to_new_data(test_df, calibration_funcs, fallback_decisions, modelname=modelname)
    print("\n评估校准效果（包括Field-ECE）...")
    metrics = evaluate_calibration_with_field_ece(result_df, n_bins=ece_bins) #!!!,计算离线指标，ece之类的
    #metrics = evaluate_calibration_with_field_ece(test_df, n_bins=ece_bins)

    # print("\n可视化校准曲线...")
    # visualize_calibration_curves(result_df, post_suffix="test", n_bins=ece_bins)

    # # 可视化每个字段的ECE详情
    # fields = ['adid', 'pkg', '二级类目']
    # if 'cvtype' in result_df.columns:
    #     fields.append('cvtype')

    # for field in fields:
    #     print(f"\n可视化字段 {field} 的ECE详情...")
    #     visualize_field_ece_details(result_df, field,prefix="test", n_bins=ece_bins)

    # 保存结果
    result_df.to_csv(test_path+ modelname+ '_test_calibrated_results_with_fallback.csv', index=False) #!!!,test的校准值的保存路径

    # 保存评估指标
    with open(test_path+ '_test_calibration_metrics.json', 'w') as f: #!!!,test的校准指标的保存路径
        json.dump(metrics, f, indent=4, default=str)

    print("\n校准结果和评估指标已保存")
    
    print("\n分析业务评估结果...")
    business_metrics = analyze_business_impact(result_df) #!!!,test的偏差相关的指标
    print_business_metrics(business_metrics)

    # 保存业务评估指标
    with open(test_path + modelname + '_test_business_metrics.json', 'w') as f: #!!!,test的偏差相关的指标保存路径
        json.dump(business_metrics, f, indent=4, default=str)
    print(f"\n业务评估指标已保存到 {test_path}_test_business_metrics.json")

def test_df(test_path, model_file, ece_bins, modelname='pava', use_adaptive_mixing=False):
    """
    使用保存的校准模型对测试数据进行校准

    参数:
    test_path: 测试数据文件路径
    model_file: 校准模型文件路径
    ece_bins: 计算ECE的分桶数量
    modelname: 模型名称，'pava'或'bbq'
    """
    print(f"加载测试数据并应用{modelname}模型进行校准...")
    test_df = load_data(test_path)
    calibration_funcs, calibration_data, fallback_decisions = load_calibration_model(model_file)

    # 根据模型类型执行不同的推理逻辑
    if modelname == 'bbq':
        print("执行BBQ模型特定的推理逻辑...")
        # 为bbq模型添加特定的处理
        # 1. 检查calibration_data中是否包含bbq相关信息
        has_bbq_data = any(data.get('modelname') == 'bbq' for data in calibration_data.values())
        if has_bbq_data:
            print("检测到BBQ模型数据，将使用bbq特定的方式进行处理")
            # 记录使用的模型类型
            test_df['model_type'] = 'bbq'
    else:
        # 原始的pava模型逻辑
        test_df['model_type'] = 'pava'

    # 执行校准推理，传递modelname参数
    pred(test_df, calibration_funcs, fallback_decisions, ece_bins, test_path,modelname)

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型校准系统')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='运行模式')
    parser.add_argument('--modelname', type=str, default='pava', choices=['pava', 'bbq', 'platt'], help='模型类型')
    parser.add_argument('--model_file', type=str, default='calibration_model.pkl', help='测试模式下的模型文件路径')
    parser.add_argument('--train_file', type=str, default='train_data.csv', help='训练数据文件路径')
    parser.add_argument('--test_file', type=str, default='test_data.csv', help='测试数据文件路径')
    parser.add_argument('--config_path', type=str, default='config.txt', help='配置文件路径')
    parser.add_argument('--min_samples', type=int, default=100, help='最小样本数量阈值')
    parser.add_argument('--n_bins', type=int, default=None, help='分桶数量（默认自动计算）')
    parser.add_argument('--adaptive_threshold', action='store_true', default=False, help='是否使用自适应阈值')
    parser.add_argument('--threshold_history_file', type=str, default='threshold_history.json', help='阈值历史记录文件路径')
    parser.add_argument('--use_adaptive_mixing', action='store_true', default=False, help='是否使用区间自适应的混合校准策略')

    args = parser.parse_args()

    if args.mode == "train":
        result_df, calibration_funcs, calibration_data, fallback_decisions, metrics, threshold_stats, business_metrics = main(
            args.train_file, args.config_path, args.test_file, n_bins=args.n_bins, min_samples=args.min_samples, ece_bins=10,
            modelname=args.modelname, model_file=args.model_file, adaptive_threshold=args.adaptive_threshold,
            threshold_history_file=args.threshold_history_file, use_adaptive_mixing=args.use_adaptive_mixing
        )
    elif args.mode == "test":
        test_df(args.test_file, args.model_file, 10, args.modelname, use_adaptive_mixing=args.use_adaptive_mixing)
    # 可以调整分桶数量、最小置信样本数量和ECE计算的分桶数量
    #result_df, calibration_funcs, calibration_data, fallback_decisions, metrics = main(
    #    file_path, config_path,test_path, n_bins=100, min_samples=100, ece_bins=10
    #)

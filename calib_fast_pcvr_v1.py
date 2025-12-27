import pandas as pd
import numpy as np
import time
import re
from collections import defaultdict

class PCvrCalibrator:
    """PCvr校准器：支持多维度训练和评估，集成多种校准算法"""

    def __init__(self, calibration_method="histogram_binning", base_bin_info_file="bin_information",
                 increasing=True):
        self.calibration_method = calibration_method  # "histogram_binning" 或 "isotonic_regression"
        self.bin_info = {}  # 统一存储每个维度的分箱/区间信息（适用于所有算法）
        self.dim_stats = {}  # 存储训练过程中每个维度的基本统计信息
        self.overall_real_ltv_train = 0  # 训练数据中所有样本的real_ltv总和
        self.base_bin_info_file = base_bin_info_file  # 分箱/模型信息文件的基础名称
        self.train_dims = []  # 训练维度列表
        self.dim_min_samples = {}  # 每个维度的每个分组的最小样本数

        # 保序回归相关参数
        self.increasing = increasing  # 保序回归是否为递增

    def _parse_dimension(self, dim):
        """解析维度字符串，返回构成该维度的字段列表"""
        if '*' in dim:
            return dim.split('*')
        return [dim]

    def _get_group_key(self, row, fields):
        """根据字段列表从行数据中生成分组键"""
        if len(fields) == 1:
            return row[fields[0]]
        return tuple(row[field] for field in fields)

    def _histogram_binning(self, group_df, min_samples):
        """
        直方图分箱算法实现
        返回：分箱信息字典 {bin_id: (bin_left, bin_right, calibration_value, sample_count)}
        """
        group_sample_count = len(group_df)

        # 动态计算分箱数量：确保每个分箱至少有min_samples个样本
        max_possible_bins = group_sample_count // min_samples
        n_bins = max(1, max_possible_bins)  # 至少保留1个分箱

        try:
            # 等频分箱（默认）
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges = group_df["pltv"].quantile(quantiles).unique()

            # 处理分位数去重后分箱数不足的特殊情况
            actual_bins = len(bin_edges) - 1
            if actual_bins < 1:
                # 所有pltv值相同：手动创建1个分箱
                min_pltv = group_df["pltv"].min()
                max_pltv = group_df["pltv"].max()
                bin_edges = [min_pltv - 1e-9, max_pltv + 1e-9]
                group_df["pltv_bin"] = 0
                actual_bins = 1
            else:
                group_df["pltv_bin"] = pd.cut(
                    group_df["pltv"],
                    bins=bin_edges,
                    labels=False,
                    include_lowest=True
                )

            # 计算每个分箱的校准值和样本数
            bin_calibrated = group_df.groupby("pltv_bin").agg(
                real_ltv_mean=("real_ltv", "mean"),
                bin_sample_count=("real_ltv", "count")
            ).reset_index()

            # 保存有效的分箱信息
            group_bin_info = {}
            for bin_id in range(len(bin_edges) - 1):
                bin_left = bin_edges[bin_id]
                bin_right = bin_edges[bin_id + 1]
                bin_data = bin_calibrated[bin_calibrated["pltv_bin"] == bin_id]

                if len(bin_data) > 0 and bin_data["bin_sample_count"].values[0] >= min_samples:
                    group_bin_info[bin_id] = (
                        bin_left,
                        bin_right,
                        bin_data["real_ltv_mean"].values[0],
                        bin_data["bin_sample_count"].values[0]
                    )

            return group_bin_info

        except Exception as e:
            print(f"直方图分箱失败: {str(e)}")
            return {}

    def _isotonic_regression(self, group_df, min_samples):
        """
        自定义实现Isotonic Regression算法（PAVA）
        输入：group_df - 包含pltv和real_ltv的分组数据框
             min_samples - 该分组的最小样本数要求
        输出：分箱信息字典 {bin_id: (bin_left, bin_right, calibration_value, sample_count)}
        """
        group_sample_count = len(group_df)

        # 样本数不足，不生成模型
        if group_sample_count < min_samples:
            return {}

        try:
            # 提取并排序数据（PAVA算法要求输入已排序）
            sorted_df = group_df.sort_values('pltv').copy()
            x = sorted_df['pltv'].values
            y = sorted_df['real_ltv'].values
            n = len(x)

            if n == 0:
                return {}

            # 初始化区间信息
            intervals = np.arange(n, dtype=int)  # 区间终点索引
            values = y.copy()  # 区间平均值
            sizes = np.ones(n, dtype=int)  # 区间样本数量

            i = 0
            while i < len(intervals) - 1:
                # 检查单调性
                monotonic_violation = (self.increasing and values[i] > values[i+1]) or \
                                     (not self.increasing and values[i] < values[i+1])

                # 检查当前区间样本数是否满足最小要求（使用dim_min_samples）
                size_violation = sizes[i] < min_samples

                if monotonic_violation or size_violation:
                    # 合并区间
                    merged_value = (values[i] * sizes[i] + values[i+1] * sizes[i+1]) / (sizes[i] + sizes[i+1])
                    values[i] = merged_value
                    intervals[i] = intervals[i+1]
                    sizes[i] += sizes[i+1]

                    # 移除被合并区间
                    intervals = np.delete(intervals, i+1)
                    values = np.delete(values, i+1)
                    sizes = np.delete(sizes, i+1)

                    # 回溯检查
                    if i > 0:
                        i -= 1
                else:
                    i += 1

            # 处理可能剩余的样本数不足的区间（从后往前检查）
            i = len(intervals) - 1
            while i > 0:
                if sizes[i] < min_samples:
                    # 合并当前区间与前一个区间
                    values[i-1] = (values[i-1] * sizes[i-1] + values[i] * sizes[i]) / (sizes[i-1] + sizes[i])
                    intervals[i-1] = intervals[i]
                    sizes[i-1] += sizes[i]

                    # 移除当前区间
                    intervals = np.delete(intervals, i)
                    values = np.delete(values, i)
                    sizes = np.delete(sizes, i)

                i -= 1

            # 生成校准结果和区间信息（确保相邻区间连续）
            idx = 0
            group_bin_info = {}  # 存储分箱信息
            for bin_id in range(len(intervals)):
                end = intervals[bin_id] + 1

                # 确定区间边界（确保相邻区间连续）
                current_left = x[idx] if bin_id == 0 else group_bin_info[bin_id-1][1]
                current_right = x[end-1]

                # 记录区间的左右边界、校准值和样本数
                group_bin_info[bin_id] = (
                    current_left,
                    current_right,
                    values[bin_id],
                    sizes[bin_id]
                )
                idx = end

            return group_bin_info

        except Exception as e:
            print(f"保序回归训练失败: {str(e)}")
            return {}

    def train(self, df, train_dims=None, dim_min_samples=None, data_type="train"):
        """
        使用训练数据训练校准模型：支持多维度和多种校准算法
        """
        print(f"开始{data_type}数据训练（算法: {self.calibration_method}）：")
        # 确定训练维度
        self.train_dims = train_dims if train_dims and isinstance(train_dims, list) and len(train_dims) > 0 else ["sample_level"]
        print(f"训练维度: {self.train_dims}")

        # 为每个维度设置最小样本数
        self.dim_min_samples = dim_min_samples if dim_min_samples and isinstance(dim_min_samples, dict) else {}
        for dim in self.train_dims:
            if dim not in self.dim_min_samples:
                self.dim_min_samples[dim] = 5  # 未指定时的默认最小值

        # 计算训练数据的总real_ltv
        self.overall_real_ltv_train = df["real_ltv"].sum()

        # 为每个维度初始化信息存储和文件
        for dim in self.train_dims:
            self.bin_info[dim] = {}  # 统一使用bin_info存储分箱信息
            self.dim_stats[dim] = {}

            # 初始化信息文件并写入表头（所有算法共用相同表头）
            fields = self._parse_dimension(dim)
            dim_sanitized = dim.replace('*', '_')
            info_file = f"{self.base_bin_info_file}_{dim_sanitized}.txt"

            with open(info_file, 'w', encoding='utf-8') as f:
                header = "|".join(fields) + "|bin_id,bin_left,bin_right,calibration_value,sample_count\n"
                f.write(header)

        # 对每个维度进行训练
        for dim in self.train_dims:
            fields = self._parse_dimension(dim)
            dim_sanitized = dim.replace('*', '_')
            info_file = f"{self.base_bin_info_file}_{dim_sanitized}.txt"
            min_samples = self.dim_min_samples[dim]

            # 检查该维度所需的字段是否存在
            missing_fields = [f for f in fields if f not in df.columns]
            if missing_fields:
                print(f"警告：维度 {dim} 所需字段 {missing_fields} 不存在，跳过该维度训练")
                continue

            # 按维度分组
            groups = df.groupby(fields, observed=True)
            # 按样本数量降序排序分组
            sorted_groups = sorted(groups, key=lambda x: len(x[1]), reverse=True)

            # 处理每个分组以生成模型和统计信息
            for group_key, group_df in sorted_groups:
                group_sample_count = len(group_df)  # 该分组中的训练样本数

                # 计算该分组训练集的核心统计量
                group_pltv_sum = group_df["pltv"].sum()
                group_real_ltv_sum = group_df["real_ltv"].sum()
                group_real_ltv_percent = (group_real_ltv_sum / self.overall_real_ltv_train) * 100 if self.overall_real_ltv_train > 0 else 0

                # 保存训练集统计信息
                self.dim_stats[dim][group_key] = {
                    "train_sample_count": group_sample_count,
                    "train_pltv_sum": group_pltv_sum,
                    "train_real_ltv_sum": group_real_ltv_sum,
                    "train_real_ltv_percent": group_real_ltv_percent
                }

                # 如果样本数不足，则不生成模型
                if group_sample_count < min_samples:
                    print(f"{data_type} 维度 {dim} 分组 {group_key} 训练样本不足({group_sample_count} < {min_samples})，不生成校准模型")
                    self.bin_info[dim][group_key] = {}
                    continue

                try:
                    # 根据选择的算法训练模型，统一使用group_bin_info变量
                    if self.calibration_method == "histogram_binning":
                        # 使用直方图分箱算法
                        group_bin_info = self._histogram_binning(group_df, min_samples)
                    else:
                        # 使用保序回归算法，使用dim_min_samples作为最小样本数
                        group_bin_info = self._isotonic_regression(group_df, min_samples)

                    self.bin_info[dim][group_key] = group_bin_info
                    bin_count = len(group_bin_info)

                    # 共用同一套代码将分箱信息写入文件（适用于所有算法）
                    if bin_count > 0:
                        with open(info_file, 'a', encoding='utf-8') as f:
                            for bin_id, (left, right, calib_val, count) in group_bin_info.items():
                                if isinstance(group_key, tuple):
                                    key_str = "|".join(map(str, group_key))
                                else:
                                    key_str = str(group_key)
                                f.write(f"{key_str}|{bin_id},{left:.6f},{right:.6f},{calib_val:.6f},{count}\n")

                    print(f"{data_type} 维度 {dim} 分组 {group_key} 校准模型训练完成：有效箱子数目={bin_count}, 训练样本数={group_sample_count}")

                except Exception as e:
                    print(f"{data_type} 维度 {dim} 分组 {group_key} 训练失败：{str(e)}, 不生成校准模型")
                    self.bin_info[dim][group_key] = {}

        return self

    def calibrate(self, df, dim_priority=None, data_type="test"):
        """
        使用训练好的模型校准测试数据，生成pltv_calibrated
        """
        print(f"开始{data_type}数据校准（算法: {self.calibration_method}）：")
        df_calibrated = df.copy()
        # 初始化校准列和跟踪校准来源的列
        df_calibrated["pltv_calibrated"] = np.nan
        df_calibrated["calibration_dim"] = "null"  # 记录使用的校准维度
        df_calibrated["calibration_bin_id"] = -1   # 记录使用的箱子ID

        # 确定校准维度优先级，默认为训练维度的逆序
        calibration_dim_priority = dim_priority if dim_priority and isinstance(dim_priority, list) else sorted(self.train_dims, reverse=True)
        print(f"校准维度优先级: {calibration_dim_priority}")

        # 用于统计每个维度校准的记录数
        calibration_counts = {dim: 0 for dim in calibration_dim_priority}
        total_samples = len(df_calibrated)
        uncalibrated_count = total_samples

        # 按优先级遍历每个校准维度
        for dim in calibration_dim_priority:
            # 检查分箱信息是否存在
            if dim not in self.bin_info:
                print(f"维度 {dim} 没有训练数据，跳过校准")
                continue

            fields = self._parse_dimension(dim)

            # 检查该维度所需的字段是否存在
            missing_fields = [f for f in fields if f not in df_calibrated.columns]
            if missing_fields:
                print(f"维度 {dim} 所需字段 {missing_fields} 不存在，跳过该维度校准")
                continue

            # 找到未校准的样本
            uncalibrated_mask = df_calibrated["pltv_calibrated"].isna()
            current_uncalibrated = uncalibrated_mask.sum()
            if current_uncalibrated == 0:
                print("所有样本已完成校准，停止校准过程")
                break

            print(f"维度 {dim} 开始校准，当前未校准样本数: {current_uncalibrated}")
            uncalibrated_df = df_calibrated[uncalibrated_mask].copy()

            # 按维度分组处理未校准样本
            groups = uncalibrated_df.groupby(fields, observed=True)
            dim_calibrated_count = 0

            for group_key, group_df in groups:
                # 检查是否有有效的分箱信息
                if group_key not in self.bin_info[dim] or len(self.bin_info[dim][group_key]) == 0:
                    # 不打印常规跳过信息，避免输出过多
                    continue

                # 获取分箱信息并进行校准
                bins = self.bin_info[dim][group_key]
                group_calibrated = 0

                for idx, row in group_df.iterrows():
                    current_pltv = row["pltv"]
                    # 遍历分箱以找到包含当前pltv的分箱
                    for bin_id, (bin_left, bin_right, calib_value, _) in bins.items():
                        if bin_left <= current_pltv < bin_right:
                            # 记录校准值
                            df_calibrated.at[idx, "pltv_calibrated"] = calib_value
                            # 记录校准维度和箱子ID
                            df_calibrated.at[idx, "calibration_dim"] = dim
                            df_calibrated.at[idx, "calibration_bin_id"] = bin_id
                            group_calibrated += 1
                            dim_calibrated_count += 1
                            break

                if group_calibrated > 0:
                    # 仅打印有校准结果的分组信息
                    print(f"维度 {dim} 分组 {group_key} 校准完成 {group_calibrated} 条记录")

            # 更新校准计数
            calibration_counts[dim] = dim_calibrated_count
            uncalibrated_count -= dim_calibrated_count
            print(f"维度 {dim} 校准完成，本维度共校准 {dim_calibrated_count} 条记录，剩余未校准: {uncalibrated_count}")

        # 打印各维度校准统计
        print("\n各维度校准记录统计:")
        for dim, count in calibration_counts.items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"  维度 {dim}: {count} 条 ({percentage:.2f}%)")

        # 填充未校准的NaN值（用原始pltv替代）
        na_mask = df_calibrated["pltv_calibrated"].isna()
        na_count = na_mask.sum()
        if na_count > 0:
            df_calibrated.loc[na_mask, "pltv_calibrated"] = df_calibrated.loc[na_mask, "pltv"]
            percentage = (na_count / total_samples) * 100 if total_samples > 0 else 0
            print(f"\n{data_type} 数据中有 {na_count} 条记录未匹配到任何校准模型 ({percentage:.2f}%)，用原始pltv填充")
        else:
            print("\n所有记录均已通过校准模型校准，无使用原始pltv的记录")

        return df_calibrated

    def evaluate(self, df, evaluate_dim=None, is_test_data=False):
        """评估校准效果：支持按不同维度进行评估"""
        # 确定评估维度
        eval_dims = evaluate_dim if evaluate_dim and isinstance(evaluate_dim, list) and len(evaluate_dim) > 0 else []
        results = {}

        # 计算整体评估
        print(f"开始{'测试数据' if is_test_data else '训练数据'}整体评估：")
        overall_eval, overall_metrics = self._evaluate_single_dim(df, "overall", is_test_data)
        results["overall"] = (overall_eval, overall_metrics)

        # 按指定维度进行评估
        for dim in eval_dims:
            print(f"开始{'测试数据' if is_test_data else '训练数据'}按{dim}维度评估：")
            # 检查维度是否有效
            fields = self._parse_dimension(dim)
            missing_fields = [f for f in fields if f not in df.columns]
            if missing_fields:
                print(f"警告：评估维度 {dim} 所需字段 {missing_fields} 不存在，跳过该维度评估")
                continue

            eval_df, eval_metrics = self._evaluate_single_dim(df, dim, is_test_data)
            results[dim] = (eval_df, eval_metrics)

        return results

    def _evaluate_single_dim(self, df, dim, is_test_data):
        """评估单个维度的校准效果"""
        data_type = "测试数据" if is_test_data else "训练数据"
        group_dim = dim
        eval_results = []

        # 计算当前数据集的总real_ltv
        overall_real_ltv = df["real_ltv"].sum() if is_test_data else self.overall_real_ltv_train

        if dim == "overall":
            # 整体评估，只有一个分组
            groups = [("overall", df)]
        else:
            # 按维度分组
            fields = self._parse_dimension(dim)
            groups = df.groupby(fields, observed=True)
            # 转换为列表以便处理
            groups = [(k, v) for k, v in groups]

        # 遍历每个分组计算评估指标
        for group_key, group_data in groups:
            # 没有real_ltv列则无法评估，跳过
            if "real_ltv" not in group_data.columns:
                print(f"{data_type} 警告：{group_dim} {group_key} 没有real_ltv列，跳过评估")
                continue

            try:
                # 1. 计算分组核心统计量
                group_sample_count = len(group_data)
                group_pltv_sum = group_data["pltv"].sum()
                group_pltv_calibrated_sum = group_data["pltv_calibrated"].sum()
                group_real_ltv_sum = group_data["real_ltv"].sum()
                group_real_ltv_percent = (group_real_ltv_sum / overall_real_ltv) * 100 if overall_real_ltv > 0 else 0

                # 2. 计算校准前后的偏差
                before_bias = (group_pltv_sum / group_real_ltv_sum) if group_real_ltv_sum != 0 else 0.0
                after_bias = (group_pltv_calibrated_sum / group_real_ltv_sum) if group_real_ltv_sum != 0 else 0.0

                # 3. 计算偏差减少量
                before_abs = abs(before_bias - 1)
                after_abs = abs(after_bias - 1)
                bias_reduction = before_abs - after_abs

                # 整理该分组的评估结果
                if dim == "overall":
                    result = {
                        "group": "overall",
                        "sample_count": group_sample_count,
                        "pltv_sum": round(group_pltv_sum, 6),
                        "pltv_calibrated_sum": round(group_pltv_calibrated_sum, 6),
                        "real_ltv_sum": round(group_real_ltv_sum, 6),
                        "real_ltv_percent(%)": 100.0,  # 整体占比为100%
                        "before_calib_bias": round(before_bias, 6),
                        "after_calib_bias": round(after_bias, 6),
                        "bias_reduction": round(bias_reduction, 6)
                    }
                else:
                    result = {
                        group_dim: group_key if not isinstance(group_key, tuple) else "*".join(map(str, group_key)),
                        "sample_count": group_sample_count,
                        "pltv_sum": round(group_pltv_sum, 6),
                        "pltv_calibrated_sum": round(group_pltv_calibrated_sum, 6),
                        "real_ltv_sum": round(group_real_ltv_sum, 6),
                        "real_ltv_percent(%)": round(group_real_ltv_percent, 4),
                        "before_calib_bias": round(before_bias, 6),
                        "after_calib_bias": round(after_bias, 6),
                        "bias_reduction": round(bias_reduction, 6)
                    }

                # 为测试数据的评估指标添加前缀
                if is_test_data:
                    result = {
                        f"test_data_{k}" if k in ["before_calib_bias", "after_calib_bias", "bias_reduction"]
                        else k: v for k, v in result.items()
                    }

                eval_results.append(result)

            except Exception as e:
                print(f"{data_type} {group_dim} {group_key} 评估失败：{str(e)}, 跳过该分组")
                continue

        # 转换为DataFrame并按real_ltv_percent降序排序
        eval_df = pd.DataFrame(eval_results)
        if not eval_df.empty and dim != "overall":
            eval_df = eval_df.sort_values(by="real_ltv_percent(%)", ascending=False).reset_index(drop=True)

        # 提取整体评估指标
        overall_metrics = eval_results[0] if eval_results else {}
        overall_metrics[f"{group_dim}_evaluation"] = True

        return eval_df, overall_metrics

    def analyze_bias_distribution(self, eval_result, overall_real_ltv, dim="overall"):
        """分析指定维度的偏差分布情况"""
        # 从评估结果中提取数据框
        eval_df, _ = eval_result

        # 定义偏差区间
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, float('inf')]
        labels = [
            "[0,0.1)", "[0.1,0.2)", "[0.2,0.3)", "[0.3,0.4)", "[0.4,0.5)",
            "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0)",
            "[1.0,1.1)", "[1.1,1.2)", "[1.2,∞)"
        ]

        # 确定使用哪个偏差列
        bias_column = "test_data_before_calib_bias" if "test_data_before_calib_bias" in eval_df.columns else "before_calib_bias"
        value_column = "real_ltv_sum"

        # 分析校准前的偏差分布
        before_distribution = self._calculate_bias_distribution(
            eval_df, bias_column, value_column, bins, labels, overall_real_ltv
        )

        # 分析校准后的偏差分布
        bias_column_after = "test_data_after_calib_bias" if "test_data_after_calib_bias" in eval_df.columns else "after_calib_bias"
        after_distribution = self._calculate_bias_distribution(
            eval_df, bias_column_after, value_column, bins, labels, overall_real_ltv
        )

        return before_distribution, after_distribution

    def _calculate_bias_distribution(self, df, bias_column, value_column, bins, labels, total_value):
        """计算特定偏差列的分布情况"""
        distribution = defaultdict(lambda: {"count": 0, "value_sum": 0.0, "value_percent": 0.0})

        for _, row in df.iterrows():
            if pd.isna(row[bias_column]):
                continue

            bias = row[bias_column]
            value = row[value_column]

            # 找到对应的区间
            for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
                if lower <= bias < upper:
                    distribution[labels[i]]["count"] += 1
                    distribution[labels[i]]["value_sum"] += value
                    break

        # 计算每个区间的百分比
        for label in distribution:
            if total_value > 0:
                distribution[label]["value_percent"] = (distribution[label]["value_sum"] / total_value) * 100
            else:
                distribution[label]["value_percent"] = 0.0

        # 转换为DataFrame并按区间升序排序
        dist_df = pd.DataFrame.from_dict(distribution, orient="index").reset_index()
        dist_df.columns = ["bias_interval", "count", f"{value_column}_sum", f"{value_column}_percentage(%)"]
        dist_df = dist_df.sort_values(by="bias_interval", key=lambda x: [labels.index(label) for label in x])

        return dist_df

def read_data(txt_path, data_type="data", train_dims=None, calibrate_dims=None, evaluate_dims=None):
    """读取TSV格式的LTV数据，从第一行获取列名，并展示关键列信息"""
    try:
        # 尝试读取第一行获取列名
        with open(txt_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                raise ValueError("文件为空，无法读取列名")

            columns = first_line.split('\t')
            if len(columns) == 0:
                raise ValueError("文件第一行不包含有效的列名信息")

        # 读取数据
        df = pd.read_csv(
            txt_path,
            sep="\t",
            header=0,  # 第一行作为列名
            dtype={
                "pltv": float,
                "real_ltv": float,
                "app_pkg": str,
                "media_type": str,
                "day": str,
                "imei": str
            }
        )

        # 验证所需列是否存在
        required_columns = ["pltv", "real_ltv"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据文件缺少必要的列: {', '.join(missing_columns)}")

        # 收集所有维度相关的字段
        all_dim_fields = set()

        # 处理训练维度
        if train_dims and isinstance(train_dims, list):
            for dim in train_dims:
                fields = dim.split('*') if '*' in dim else [dim]
                all_dim_fields.update(fields)

        # 处理校准维度
        if calibrate_dims and isinstance(calibrate_dims, list):
            for dim in calibrate_dims:
                fields = dim.split('*') if '*' in dim else [dim]
                all_dim_fields.update(fields)

        # 处理评估维度
        if evaluate_dims and isinstance(evaluate_dims, list):
            for dim in evaluate_dims:
                fields = dim.split('*') if '*' in dim else [dim]
                all_dim_fields.update(fields)

        # 过滤出实际存在的维度字段
        existing_dim_fields = [field for field in all_dim_fields if field in df.columns]
        missing_dim_fields = [field for field in all_dim_fields if field not in df.columns]

        # 打印基本数据信息
        print(f"{data_type}读取完成：")
        print(f"  共 {len(df)} 行，{len(df.columns)} 列")
        print(f"  必要列：{', '.join(required_columns)} (均存在)")

        # 打印维度相关字段信息
        if existing_dim_fields:
            print(f"  维度相关字段 ({len(existing_dim_fields)} 个):")
            for field in existing_dim_fields:
                unique_count = df[field].nunique()
                print(f"    {field}: {unique_count} 个不同值")

        # 打印缺失的维度字段警告
        if missing_dim_fields:
            print(f"  警告：以下维度字段不存在于数据中: {', '.join(missing_dim_fields)}")

        # 打印组合维度信息
        all_dims = set()
        if train_dims:
            all_dims.update(train_dims)
        if calibrate_dims:
            all_dims.update(calibrate_dims)
        if evaluate_dims:
            all_dims.update(evaluate_dims)

        combo_dims = [dim for dim in all_dims if '*' in dim]
        if combo_dims:
            print(f"  组合维度信息:")
            for dim in combo_dims:
                fields = dim.split('*')
                if all(field in df.columns for field in fields):
                    groups_count = df.groupby(fields, observed=True).ngroups
                    print(f"    {dim}: {groups_count} 个不同组合")
                else:
                    missing = [f for f in fields if f not in df.columns]
                    print(f"    {dim}: 无法计算组合数 (缺少字段: {', '.join(missing)})")

        return df

    except Exception as e:
        print(f"读取{data_type}失败：{str(e)}")
        print("程序将退出")
        exit(1)


# 主流程：训练→校准→评估→结果保存
if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()

    # 1. 配置参数（可根据实际需求调整）
    TRAIN_DATA_PATH = "/Users/11179767/Code/test/py/Calib/tmp_fast_pcvr_offline_Calibration_input_v1_data.txt"
    TEST_DATA_PATH = "/Users/11179767/Code/test/py/Calib/tmp_fast_pcvr_offline_Calibration_test_v1_data.txt"
    # 校准算法选择："histogram_binning"（直方图分箱）或 "isotonic_regression"（保序回归）
    CALIBRATION_METHOD = "histogram_binning"  # 在这里配置选择的校准算法
    OUTPUT_TRAIN = f"./{CALIBRATION_METHOD}_train_calibrated.txt"
    OUTPUT_TEST = f"./{CALIBRATION_METHOD}_test_calibrated.txt"
    BASE_BIN_INFO_FILE = f"./{CALIBRATION_METHOD}_bin_details"  # 分箱/模型信息文件的基础名称

    # 保序回归参数配置
    ISOTONIC_INCREASING = True  # 保序回归是否为递增

    # 新增配置：评估结果显示的最大行数，-1表示显示所有
    MAX_EVAL_DISPLAY_ROWS = 25

    # 训练维度配置 - 支持单个维度和组合维度（用*连接）
    TRAIN_DIMS = ["ad_id", "app_pkg"]  # "ad_id", "app_pkg", "industry_name"
    # 每个训练维度的最小样本数（同时用于保序回归的区间最小样本数）
    DIM_MIN_SAMPLES = {
        "ad_id": 100,
        "app_pkg": 100,
        "industry_name": 100 
    }
    # 校准维度优先级配置 - 组合维度通常应优先于单一维度
    CALIBRATE_DIM_PRIORITY = ["ad_id", "app_pkg"]  # "ad_id", "app_pkg", "industry_name"
    # 评估维度配置
    EVALUATE_DIMS = ["ad_id"]  # "app_pkg", "media_type", "app_pkg*media_type"

    # 2. 读取训练数据并训练模型
    train_df = read_data(
        TRAIN_DATA_PATH,
        data_type="训练数据",
        train_dims=TRAIN_DIMS,
        calibrate_dims=CALIBRATE_DIM_PRIORITY,
        evaluate_dims=EVALUATE_DIMS
    )
    calibrator = PCvrCalibrator(
        calibration_method=CALIBRATION_METHOD,
        base_bin_info_file=BASE_BIN_INFO_FILE,
        increasing=ISOTONIC_INCREASING
    )
    calibrator.train(
        train_df,
        train_dims=TRAIN_DIMS,
        dim_min_samples=DIM_MIN_SAMPLES,
        data_type="train"
    )

    # 3. 校准训练数据并评估
    train_calibrated = calibrator.calibrate(
        train_df,
        dim_priority=CALIBRATE_DIM_PRIORITY,
        data_type="train"
    )
    train_eval_results = calibrator.evaluate(
        train_calibrated,
        evaluate_dim=EVALUATE_DIMS,
        is_test_data=False
    )

    # 输出训练数据评估结果
    print("\n=== 训练数据整体评估结果 ===")
    overall_eval, overall_metrics = train_eval_results["overall"]
    for key, value in overall_metrics.items():
        if key != "overall_evaluation":
            print(f"{key}: {value}")

    for dim in EVALUATE_DIMS:
        if dim in train_eval_results:
            print(f"\n=== 训练数据按{dim}维度评估结果 ===")
            eval_df, eval_metrics = train_eval_results[dim]
            if not eval_df.empty:
                # 根据配置决定显示的行数
                if MAX_EVAL_DISPLAY_ROWS > 0 and len(eval_df) > MAX_EVAL_DISPLAY_ROWS:
                    print(eval_df.head(MAX_EVAL_DISPLAY_ROWS).to_string(index=False))
                    print(f"... 仅显示前{MAX_EVAL_DISPLAY_ROWS}行，共{len(eval_df)}行 ...")
                else:
                    print(eval_df.to_string(index=False))

            # 分析并打印训练数据的偏差分布
            is_test_data = False
            overall_real_ltv = overall_metrics["real_ltv_sum"] if not is_test_data else overall_metrics_test["real_ltv_sum"]
            before_dist, after_dist = calibrator.analyze_bias_distribution(train_eval_results[dim], overall_real_ltv, dim)

            print(f"\n=== 训练数据按{dim}维度校准前偏差分布 ===")
            print(before_dist.to_string(index=False))

            print(f"\n=== 训练数据按{dim}维度校准后偏差分布 ===")
            print(after_dist.to_string(index=False))

    # 4. 处理测试数据
    test_df = read_data(
        TEST_DATA_PATH,
        data_type="测试数据",
        train_dims=TRAIN_DIMS,
        calibrate_dims=CALIBRATE_DIM_PRIORITY,
        evaluate_dims=EVALUATE_DIMS
    )
    test_calibrated = calibrator.calibrate(
        test_df,
        dim_priority=CALIBRATE_DIM_PRIORITY,
        data_type="test"
    )
    test_eval_results = calibrator.evaluate(
        test_calibrated,
        evaluate_dim=EVALUATE_DIMS,
        is_test_data=True
    )

    # 输出测试数据评估结果
    print("\n=== 测试数据整体评估结果 ===")
    overall_eval_test, overall_metrics_test = test_eval_results["overall"]
    for key, value in overall_metrics_test.items():
        if key != "overall_evaluation":
            print(f"{key}: {value}")

    for dim in EVALUATE_DIMS:
        if dim in test_eval_results:
            print(f"\n=== 测试数据按{dim}维度评估结果 ===")
            eval_df, eval_metrics = test_eval_results[dim]
            if not eval_df.empty:
                # 根据配置决定显示的行数
                if MAX_EVAL_DISPLAY_ROWS > 0 and len(eval_df) > MAX_EVAL_DISPLAY_ROWS:
                    print(eval_df.head(MAX_EVAL_DISPLAY_ROWS).to_string(index=False))
                    print(f"... 仅显示前{MAX_EVAL_DISPLAY_ROWS}行，共{len(eval_df)}行 ...")
                else:
                    print(eval_df.to_string(index=False))

            # 分析该维度的偏差分布
            is_test_data = True
            overall_real_ltv = overall_metrics_test["real_ltv_sum"] if is_test_data else overall_metrics["real_ltv_sum"]
            before_dist, after_dist = calibrator.analyze_bias_distribution(test_eval_results[dim], overall_real_ltv, dim)

            print(f"\n=== 测试数据按{dim}维度校准前偏差分布 ===")
            print(before_dist.to_string(index=False))

            print(f"\n=== 测试数据按{dim}维度校准后偏差分布 ===")
            print(after_dist.to_string(index=False))

    # 5. 保存结果
    train_calibrated.to_csv(OUTPUT_TRAIN, sep="\t", index=False, header=True)
    test_calibrated.to_csv(OUTPUT_TEST, sep="\t", index=False, header=True)
    print(f"\n校准结果已保存至：\n{OUTPUT_TRAIN}\n{OUTPUT_TEST}")

    # 计算并打印总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n程序总运行时间：{total_time:.2f}秒")

    # 再次打印整体评估结果（按要求移至运行时间之后）
    print("\n===== 最终整体评估结果汇总 =====")
    print(CALIBRATION_METHOD)
    print("\n--- 训练数据整体评估结果 ---")
    for key, value in overall_metrics.items():
        if key != "overall_evaluation":
            print(f"{key}: {value}")

    print("\n--- 测试数据整体评估结果 ---")
    for key, value in overall_metrics_test.items():
        if key != "overall_evaluation":
            print(f"{key}: {value}")

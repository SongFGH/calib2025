import pandas as pd
import numpy as np
import time
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split

#版本v2d2
#针对v2d1，优化相关细节,保存箱子中实际用到的信息
#运行代码：python calib_fast_pcvr_v2d2.py > ./result/hb_calibrationV2d2_result_fast_pcvr_flow1_1015.txt

class BinNode:
    """分箱节点类，用于链表存储分箱信息"""
    def __init__(self, start_idx, end_idx, value, size):
        self.start_idx = start_idx  # 分箱在x中的起始索引
        self.end_idx = end_idx      # 分箱在x中的结束索引
        self.value = value          # 分箱正例率
        self.size = size            # 分箱样本数
        self.prev = None            # 前驱节点
        self.next = None            # 后继节点

class ClassifierCalibrator:
    """分类模型校准器：支持多维度训练和评估，集成多种校准算法，新增单adid调优功能"""

    def __init__(self, calibration_method="histogram_binning", base_bin_info_file="bin_information",
                 increasing=True, positive_class=1):
        self.calibration_method = calibration_method  # "histogram_binning" 或 "isotonic_regression"
        self.bin_info = {}  # 统一存储每个维度的分箱/区间信息
        self.dim_stats = {}  # 存储训练过程中每个维度的基本统计信息
        self.overall_positive_train = 0  # 训练数据中所有正样本总数
        self.base_bin_info_file = base_bin_info_file  # 分箱/模型信息文件的基础名称
        self.train_dims = []  # 训练维度列表
        self.dim_min_samples = {}  # 每个维度的每个分组的最小样本数
        self.positive_class = positive_class  # 正样本标签
        self.adid_best_dim_mapping = {}  # 存储每个adid选择的最佳校准维度
        
        # 评估指标阈值参数
        self.ctr_diff_threshold = 0.001  # CTR差异提升阈值
        self.ece_improvement_threshold = 0.001  # ECE提升阈值
        self.min_positive_samples = 0  # 最小正样本数阈值
        self.min_negative_samples = 50  # 最小负样本数阈值

        # 保序回归相关参数
        self.increasing = increasing  # 保序回归是否为递增（预测概率与正例率正相关）

        # 新增：维度选择统计信息，包含更多原因分类
        self.dim_selection_stats = {
            "total_adids": 0,                  # 总adid数量
            "dimension_counts": defaultdict(int),  # 各维度被选择的数量
            "no_dimension_reason": defaultdict(int)  # 未选择维度的原因统计
        }

    def set_evaluation_thresholds(self, ctr_diff=0.01, ece_improve=0.01, min_pos=0, min_neg=50):
        """设置评估指标的阈值参数"""
        self.ctr_diff_threshold = ctr_diff
        self.ece_improvement_threshold = ece_improve
        self.min_positive_samples = min_pos
        self.min_negative_samples = min_neg

    def set_evaluation_attribute(self, fast_or_single_game="f", flow_type_id=0):
        """设置评估属性的参数"""
        self.fast_or_single_game = fast_or_single_game
        self.flow_type_id = flow_type_id
        
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

    def _calculate_ece(self, df, prob_col, label_col, n_bins=10):
        """计算预期校准误差(Expected Calibration Error)"""
        df = df.copy()
        df['bin'] = pd.cut(df[prob_col], bins=n_bins, labels=False)
        
        bin_stats = df.groupby('bin').agg({
            prob_col: ['mean'],
            label_col: ['mean', 'count']
        }).dropna()
        
        if bin_stats.empty:
            return -1.0
            
        bin_stats.columns = ['pred_mean', 'true_mean', 'count']
        bin_stats['weight'] = bin_stats['count'] / len(df)
        bin_stats['abs_diff'] = np.abs(bin_stats['pred_mean'] - bin_stats['true_mean'])
        
        return (bin_stats['weight'] * bin_stats['abs_diff']).sum()

    def _calculate_ctr_diff(self, df, pred_col, calib_col, label_col, group_key = "", data_type = ""):
        """计算校准前后的CTR差异提升"""
        # 校准前的CTR预测
        pred_ctr = df[pred_col].mean()
        # 校准后的CTR预测
        calib_ctr = df[calib_col].mean()
        # 实际CTR
        actual_ctr = (df[label_col] == self.positive_class).mean() + 1e-6
        
        # 计算差异
        pre_diff = abs(pred_ctr/actual_ctr - 1.0)
        post_diff = abs(calib_ctr/actual_ctr - 1.0)

        if (group_key == "239662521") and (data_type == "测试数据"): #debug
            print("test中239662521数据信息打印进行debug：")
            # print(df)
            print(df[['pred_prob', 'calibrated_prob']])
            # 替换为实际的列名（例如 'ctr_before' 和 'ctr_after'）
            col1 = "pred_prob"
            col2 = "calibrated_prob"

            # 比较两列元素是否相等（返回布尔值 Series）
            equal_mask = df[col1] == df[col2]

            # 统计相同元素和不同元素的个数
            same_count = equal_mask.sum()  # True 的数量（相同）
            diff_count = len(equal_mask) - same_count  # 总数量 - 相同数量 = 不同数量

            # 打印结果
            print(f"两列相同元素的个数：{same_count}")
            print(f"两列不同元素的个数：{diff_count}")
            print(pred_col, calib_col, label_col)
            print(pred_ctr, calib_ctr, actual_ctr)
            print(pre_diff - post_diff, pre_diff, post_diff)
        
        # 返回差异减少量（提升）
        return pre_diff - post_diff, pre_diff, post_diff

    def _histogram_binning(self, group_df, min_samples):
        """
        直方图分箱算法实现（先排序再分箱）
        返回：分箱信息字典 {bin_id: (bin_left, bin_right, calibration_value, sample_count)}
        """
        group_sample_count = len(group_df)
        # 样本数不足，不生成模型
        if group_sample_count < min_samples:
            return {}

        # 动态计算分箱数量：确保每个分箱至少有min_samples个样本
        max_possible_bins = group_sample_count // min_samples
        n_bins = max(1, max_possible_bins)  # 至少保留1个分箱

        try:
            # 先对pred_prob进行排序
            sorted_df = group_df.sort_values('pred_prob').copy()
            
            # 等频分箱（基于排序后的数据）
            # 计算分位数位置索引（基于排序后的数据位置）
            quantile_indices = np.linspace(0, group_sample_count - 1, n_bins + 1, dtype=int)
            # 获取分箱边界（使用排序后的数据）
            bin_edges = sorted_df["pred_prob"].iloc[quantile_indices].unique()

            # 处理分位数去重后分箱数不足的特殊情况
            actual_bins = len(bin_edges) - 1
            if actual_bins < 1:
                # 所有pred_prob值相同：手动创建1个分箱
                min_prob = sorted_df["pred_prob"].min()
                max_prob = sorted_df["pred_prob"].max()
                bin_edges = [min_prob - 1e-9, max_prob + 1e-9]
                sorted_df["prob_bin"] = 0
                actual_bins = 1
            else:
                # 使用pd.cut进行分箱，基于排序后的数据
                sorted_df["prob_bin"] = pd.cut(
                    sorted_df["pred_prob"],
                    bins=bin_edges,
                    labels=False,  # 分箱结果用整数索引（0, 1, 2...）表示
                    include_lowest=True  # 第一个区间左闭右闭,其他区间左开右闭
                )

            # 对分箱后的结果进行分组统计，计算每个概率区间的正例率和样本数量
            bin_calibrated = sorted_df.groupby("prob_bin").agg(
                positive_rate=("label", lambda x: (x == self.positive_class).mean()),
                bin_sample_count=("label", "count")
            ).reset_index()

            # 保存有效的分箱信息
            group_bin_info = {}
            for bin_id in range(len(bin_edges) - 1):
                bin_left = bin_edges[bin_id]
                bin_right = bin_edges[bin_id + 1]
                bin_data = bin_calibrated[bin_calibrated["prob_bin"] == bin_id]

                if len(bin_data) > 0 and bin_data["bin_sample_count"].values[0] >= min_samples:
                    group_bin_info[bin_id] = (
                        bin_left,
                        bin_right,
                        bin_data["positive_rate"].values[0],  # 校准值为正例率
                        bin_data["bin_sample_count"].values[0]
                    )

            return group_bin_info

        except Exception as e:
            print(f"直方图分箱失败: {str(e)}")
            return {}

    def _isotonic_regression_v1(self, group_df, min_samples):
        """
        自定义实现Isotonic Regression算法（PAVA）
        输入：group_df - 包含pred_prob和label的分组数据框
             min_samples - 该分组的最小样本数要求
        输出：分箱信息字典 {bin_id: (bin_left, bin_right, calibration_value, sample_count)}
        """
        group_sample_count = len(group_df)

        # 样本数不足，不生成模型
        if group_sample_count < min_samples:
            return {}

        try:
            # 提取并排序数据（PAVA算法要求输入已排序）
            sorted_df = group_df.sort_values('pred_prob').copy()
            x = sorted_df['pred_prob'].values
            y = (sorted_df['label'] == self.positive_class).astype(int).values  # 转换为0/1
            n = len(x)

            if n == 0:
                return {}

            # 初始化区间信息
            intervals = np.arange(n, dtype=int)  # 区间终点索引
            values = y.copy().astype(float)  # 区间平均值（正例率）
            sizes = np.ones(n, dtype=int)  # 区间样本数量

            i = 0
            while i < len(intervals) - 1:
                # 计算当前区间和下一个区间的正例率
                current_rate = values[i] if sizes[i] > 0 else 0
                next_rate = values[i+1] if sizes[i+1] > 0 else 0

                # 检查单调性
                monotonic_violation = (self.increasing and current_rate > next_rate) or \
                                     (not self.increasing and current_rate < next_rate)

                # 检查当前区间样本数是否满足最小要求
                size_violation = sizes[i] < min_samples

                if monotonic_violation or size_violation:
                    # 合并区间：使用加权平均计算合并后的正例率
                    total_size = sizes[i] + sizes[i+1]
                    merged_rate = (values[i] * sizes[i] + values[i+1] * sizes[i+1]) / total_size

                    # 更新当前分箱信息
                    values[i] = merged_rate  # 存储加权均值
                    intervals[i] = intervals[i+1]  # 更新终点索引
                    sizes[i] = total_size  # 更新样本数

                    # 移除被合并区间
                    intervals = np.delete(intervals, i+1)
                    values = np.delete(values, i+1)
                    sizes = np.delete(sizes, i+1)

                    # 回溯检查，合并后可能与前一个区间产生新的违规
                    if i > 0:
                        i -= 1 
                else:
                    i += 1

            # 处理可能剩余的样本数不足的区间（从后往前检查）
            i = len(intervals) - 1
            while i > 0:
                if sizes[i] < min_samples:
                    # 合并当前区间与前一个区间：使用加权平均
                    total_size = sizes[i-1] + sizes[i]
                    merged_rate = (values[i-1] * sizes[i-1] + values[i] * sizes[i]) / total_size

                    # 更新前一个区间信息
                    values[i-1] = merged_rate
                    intervals[i-1] = intervals[i]
                    sizes[i-1] = total_size

                    # 移除当前区间
                    intervals = np.delete(intervals, i)
                    values = np.delete(values, i)
                    sizes = np.delete(sizes, i)

                i -= 1

            # 生成校准结果和区间信息（确保相邻区间连续）
            group_bin_info = {}  # 存储分箱信息
            for bin_id in range(len(intervals)):

                # 确定区间边界（确保相邻区间连续）
                current_left = x[0] if bin_id == 0 else group_bin_info[bin_id-1][1]
                current_right = x[intervals[bin_id]]

                # 记录区间的左右边界、校准值（正例率）和样本数
                group_bin_info[bin_id] = (
                    current_left,
                    current_right,
                    values[bin_id],
                    sizes[bin_id]
                )

            return group_bin_info

        except Exception as e:
            print(f"保序回归训练失败: {str(e)}")
            return {}

    def _isotonic_regression_v2(self, group_df, min_samples):
        """
        优化版保序回归（PAVA）实现，使用链表降低时间复杂度
        输入：group_df - 包含pred_prob和label的分组数据框
            min_samples - 分箱最小样本数要求
        输出：分箱信息字典
        """
        group_sample_count = len(group_df)
        if group_sample_count < min_samples:
            return {}

        try:
            # 数据预处理（同原逻辑）
            sorted_df = group_df.sort_values('pred_prob').copy()
            x = sorted_df['pred_prob'].values
            y = (sorted_df['label'] == self.positive_class).astype(int).values
            n = len(x)
            if n == 0:
                return {}

            # 1. 初始化链表（替代数组存储）
            head = None
            prev_node = None
            for i in range(n):
                # 每个样本初始为一个分箱
                node = BinNode(
                    start_idx=i,
                    end_idx=i,
                    value=float(y[i]),
                    size=1
                )
                if prev_node:
                    prev_node.next = node
                    node.prev = prev_node
                else:
                    head = node  # 头节点
                prev_node = node

            # 2. 遍历链表合并分箱（主循环）
            current = head
            while current and current.next:
                next_node = current.next

                # 检查单调性
                current_rate = current.value
                next_rate = next_node.value
                monotonic_violation = (self.increasing and current_rate > next_rate) or \
                                    (not self.increasing and current_rate < next_rate)

                # 检查样本量
                size_violation = current.size < min_samples

                if monotonic_violation or size_violation:
                    # 合并当前节点与下一节点（O(1)操作）
                    merged_size = current.size + next_node.size
                    merged_value = (current.value * current.size + next_node.value * next_node.size) / merged_size

                    # 更新当前节点信息
                    current.end_idx = next_node.end_idx  # 合并后的结束索引
                    current.value = merged_value
                    current.size = merged_size

                    # 删除下一节点（调整指针，O(1)）
                    current.next = next_node.next
                    if next_node.next:
                        next_node.next.prev = current

                    # 回溯检查（回到前一个节点，避免重复遍历）
                    if current.prev:
                        current = current.prev
                    else:
                        current = head  # 若已在头节点，无需回溯
                else:
                    # 无违规，继续遍历下一个节点
                    current = current.next

            # 3. 生成分箱信息（遍历链表）
            group_bin_info = {}
            bin_id = 0
            current = head
            prev_right = None  # 记录前一分箱的右边界，确保连续性
            while current:
                if current.size < min_samples:
                    bin_id += 1
                    current = current.next
                    continue

                # 确定分箱边界
                current_left = x[current.start_idx] if bin_id == 0 else prev_right
                current_right = x[current.end_idx]
                prev_right = current_right  # 更新前序右边界

                group_bin_info[bin_id] = (
                    current_left,
                    current_right,
                    current.value,
                    current.size
                )
                bin_id += 1
                current = current.next

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
        self.train_dims = train_dims if train_dims and isinstance(train_dims, list) and len(train_dims) > 0 else ["req_level"]
        print(f"训练维度: {self.train_dims}")

        # 为每个维度设置最小样本数
        self.dim_min_samples = dim_min_samples if dim_min_samples and isinstance(dim_min_samples, dict) else {}
        for dim in self.train_dims:
            if dim not in self.dim_min_samples:
                self.dim_min_samples[dim] = 100  # 未指定时的默认最小值

        # 计算训练数据的总正样本数
        self.overall_positive_train = (df["label"] == self.positive_class).sum()

        # 为每个维度初始化信息存储和文件
        for dim in self.train_dims:
            self.bin_info[dim] = {}  # 统一使用bin_info存储分箱信息
            self.dim_stats[dim] = {}

            # 初始化信息文件并写入表头
            fields = self._parse_dimension(dim)
            dim_sanitized = dim.replace('*', '_')
            info_file = f"{self.base_bin_info_file}_{dim_sanitized}.txt"

            with open(info_file, 'w', encoding='utf-8') as f:
                header = "|".join(fields) + "|bin_id,bin_left,bin_right,calibration_value,positive_rate,sample_count,used\n"
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
            groups = df.groupby(fields[0], observed=True) if len(fields) == 1 else df.groupby(fields, observed=True)
            # 按样本数量降序排序分组
            sorted_groups = sorted(groups, key=lambda x: len(x[1]), reverse=True)

            # 处理每个分组以生成模型和统计信息
            for group_key, group_df in sorted_groups:
                group_sample_count = len(group_df)  # 该分组中的训练样本数
                group_positive_count = (group_df["label"] == self.positive_class).sum()
                group_negative_count = group_sample_count - group_positive_count
                group_positive_rate = group_positive_count / group_sample_count if group_sample_count > 0 else 0

                # 计算该分组训练集的核心统计量
                group_positive_rate_over_all = (group_positive_count / self.overall_positive_train) * 100 if self.overall_positive_train > 0 else 0

                # 保存训练集统计信息
                self.dim_stats[dim][group_key] = {
                    "train_sample_count": group_sample_count,
                    "train_positive_count": group_positive_count,
                    "train_negative_count": group_negative_count,
                    "train_positive_rate": group_positive_rate,
                    "train_positive_rate_over_all": group_positive_rate_over_all
                }

                # 如果样本数不足，则不生成模型
                if group_sample_count < min_samples:
                    print(f"{data_type}，维度 = {dim}，分组 = {group_key} 训练样本不足({group_sample_count} < {min_samples})，不生成校准模型")
                    self.bin_info[dim][group_key] = {}
                    continue

                try:
                    # 根据选择的算法训练模型
                    if self.calibration_method == "histogram_binning":
                        group_bin_info = self._histogram_binning(group_df, min_samples)
                    else:
                        group_bin_info = self._isotonic_regression_v2(group_df, min_samples)

                    self.bin_info[dim][group_key] = group_bin_info
                    bin_count = len(group_bin_info)

                    # 直接将分箱信息写入文件，used先设为0
                    if bin_count > 0:
                        with open(info_file, 'a', encoding='utf-8') as f:
                            for bin_id, (left, right, calib_val, count) in group_bin_info.items():
                                if isinstance(group_key, tuple):
                                    key_str = "|".join(map(str, group_key))
                                else:
                                    key_str = str(group_key)
                                f.write(f"{key_str}|{bin_id},{left:.6f},{right:.6f},{calib_val:.6f},{count},0\n")

                    print(f"{data_type}，维度 = {dim}，分组 = {group_key} 校准模型训练完成：有效箱子数目={bin_count}, 训练样本数={group_sample_count}, 正例率={group_positive_rate:.6f}")

                except Exception as e:
                    print(f"{data_type}，维度 = {dim}，分组 = {group_key} 训练失败：{str(e)}, 不生成校准模型")
                    self.bin_info[dim][group_key] = {}

        return self

    def generate_online_dict(self,  train_df, file_path=None):
        """为每个adid生成在线使用的词典"""
        print("在线使用的离线词典开始保存")
        if not self.adid_best_dim_mapping:
            print("没有离线词典可保存")
            return
        
        # 如果未指定路径，使用基础文件名
        if file_path is None:
            print("在线使用的离线词典路径不存在")
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # 写入每条映射
                for adid, best_dim in self.adid_best_dim_mapping.items():
                    dim_str = self._generate_adId_online_dict(train_df, adid, best_dim)
                    if dim_str != "":
                        f.write(f"{dim_str}\n")
            
            print(f"在线使用的离线词典已保存至: {file_path}")
            return
            
        except Exception as e:
            print(f"保存在线使用的离线词典失败: {str(e)}")
            return None

    def _generate_adId_online_dict(self, train_df, adid, best_dim):
        """
        为每个adid生成在线使用的词典
        在线词典格式：0/1/2^Af/s_cvType_1/2/3/4_adId/groupId/appId^A箱子边界1，箱子边界2，箱子边界3，箱子边界4；纠偏值1，纠偏值2，纠偏值3
        第一列0/1/2表示cpd、cpc、adnet场景，^A是分隔符，
        第二列f/s表示是单机游戏或者快游戏的词典，cvType表示adId的转化类型，1/2/3表示纠偏维度是adId/groupId/appId的哪一类，^A是分隔符，
        第三列箱子边界1，箱子边界2，箱子边界3，箱子边界4是箱子的分隔符号，中间是分号；纠偏值1，纠偏值2，纠偏值3是每个箱子的纠偏值
        """
        if best_dim is None:
            return ""

        m_xuanyuanDelim = chr(1)  # 请求维度和广告维度的分隔符
        first_column = str(self.flow_type_id) #0-cpd，1-cpc，2-广告联盟
        second_column = self.fast_or_single_game #"f","s"
        third_column = ""


        # 筛选当前adid的数据
        adid_train_df = train_df[train_df['ad_id'] == adid].copy()
        if len(adid_train_df) == 0:
            return ""


        cvType = "0"
        if "cvType" not in adid_train_df.columns:
            print("生成在线使用的离线词典的时候：adid_train_df中不存在'cvType'列")
        else:
            # 提取第一行的cvType值
            cvType = adid_train_df.iloc[0]["cvType"]

        #1/2/3表示纠偏维度是adId/groupId/appId的哪一类
        dim_class = "0"
        if best_dim == "ad_id":
            dim_class = "1"
        elif best_dim == "ad_group_id":
            dim_calss = "2"
        elif best_dim == "app_id":
            dim_class = "3"


        #获取adId/groupId/appId
        dim_value = "0"
        # 提取该adid在当前维度的分组键
        fields = self._parse_dimension(best_dim)
        dim_value = self._get_group_key(adid_train_df.iloc[0], fields)
        # 检查该分组是否有校准信息
        if dim_value not in self.bin_info[best_dim] or len(self.bin_info[best_dim][dim_value]) == 0:
            print("生成在线使用的离线词典的时候：不存在该维度的校准信息")
            return ""
        second_column = second_column + "_" + cvType + "_" + dim_class + "_" + dim_value
        # print("second_column")
        # print(second_column)


        bin_edge_list = []
        calib_value_list = []
        for bin_id, (bin_left, bin_right, calib_value, _) in self.bin_info[best_dim][dim_value].items():
            if len(bin_edge_list) == 0:
                bin_edge_list.append(bin_left)
                bin_edge_list.append(bin_right)
            else:
                bin_edge_list.append(bin_right)
            calib_value_list.append(calib_value)
        if len(bin_edge_list) != (len(calib_value_list) + 1):
            print("生成在线使用的离线词典的时候：箱子边界和校准值对不上")
            return ""

        bin_edge_result = ','.join(f"{num:.6f}" for num in bin_edge_list)
        calib_value_result = ','.join(f"{num:.6f}" for num in calib_value_list)
        third_column = bin_edge_result + ";" + calib_value_result
        # print("third_column")
        # print(third_column)

        final_result = first_column + m_xuanyuanDelim + second_column + m_xuanyuanDelim + third_column
        # print("final_result")
        # print(final_result)

        return final_result
        

    def select_best_dimension_for_adid(self, adid, train_df, val_df, candidate_dims):
        """
        为单个adid选择最佳校准维度
        标准：训练集和验证集的样本数目以及正负样本数满足阈值，训练集CTR差异提升，验证集ECE提升
        扩展：细化未选择维度的具体原因
        """
        # 筛选当前adid的数据
        adid_train_df = train_df[train_df['ad_id'] == adid].copy()
        adid_val_df = val_df[val_df['ad_id'] == adid].copy()
        
        # 记录所有未选择原因
        reasons = []
        
        # 如果数据不足，记录原因
        if len(adid_train_df) < (self.min_positive_samples + self.min_negative_samples) or \
           len(adid_val_df) < (self.min_positive_samples + self.min_negative_samples):
            reason = "样本不足（训练或验证）"
            reasons.append(reason)
            print(f"adid={adid} {reason}，无法选择校准维度")
            return None, reasons
        
        # 检查正负样本数量是否满足阈值
        train_pos = (adid_train_df['label'] == self.positive_class).sum()
        train_neg = len(adid_train_df) - train_pos
        
        if train_pos < self.min_positive_samples or train_neg < self.min_negative_samples:
            reason = "正负样本数不满足阈值"
            reasons.append(reason)
            print(f"adid={adid} {reason}，无法选择校准维度")
            return None, reasons
        
        # 对每个候选维度进行评估
        has_valid_dim = False
        for dim in candidate_dims:
            # 检查维度是否存在于训练信息中
            if dim not in self.bin_info:
                reason = f"维度不存在于训练信息中({dim})"
                reasons.append(reason)
                continue
                
            has_valid_dim = True  # 标记存在有效维度
            # 提取该adid在当前维度的分组键
            fields = self._parse_dimension(dim)
            # 确保有数据
            if len(adid_train_df) == 0:
                continue
            group_key = self._get_group_key(adid_train_df.iloc[0], fields)
            
                
            # 检查该分组是否有校准信息
            if group_key not in self.bin_info[dim] or len(self.bin_info[dim][group_key]) == 0:
                reason = f"维度{dim}无有效校准信息"
                reasons.append(reason)
                continue
            
            # 1. 在训练集上使用该维度进行校准并评估CTR差异提升
            temp_train = adid_train_df.copy()
            temp_train['calibrated_prob'] = temp_train['pred_prob']  # 初始化
            for idx, row in temp_train.iterrows():
                current_prob = row["pred_prob"]
                for bin_id, (bin_left, bin_right, calib_value, _) in self.bin_info[dim][group_key].items():
                    if bin_left <= current_prob < bin_right:
                        temp_train.at[idx, "calibrated_prob"] = calib_value
                        break
            
            ctr_diff_improve, _, _ = self._calculate_ctr_diff(temp_train, 'pred_prob', 'calibrated_prob', 'label')
            if ctr_diff_improve < self.ctr_diff_threshold:
                reason = f"训练集CTR差异提升不足({dim})"
                reasons.append(reason)
                print(f"adid={adid} 维度={dim} 训练集CTR差异提升不足({ctr_diff_improve:.4f} < {self.ctr_diff_threshold})")
                continue
            
            # 2. 在验证集上使用该维度进行校准并评估ECE提升
            temp_val = adid_val_df.copy()
            temp_val['calibrated_prob'] = temp_val['pred_prob']  # 初始化
            for idx, row in temp_val.iterrows():
                current_prob = row["pred_prob"]
                for bin_id, (bin_left, bin_right, calib_value, _) in self.bin_info[dim][group_key].items():
                    if bin_left <= current_prob < bin_right:
                        temp_val.at[idx, "calibrated_prob"] = calib_value
                        break
            
            ece_before = self._calculate_ece(temp_val, 'pred_prob', 'label')
            if ece_before == -1.0:
                reason = f"ECE计算异常({dim})"
                reasons.append(reason)
                print("ece_before计算异常")
                continue
            ece_after = self._calculate_ece(temp_val, 'calibrated_prob', 'label')
            if ece_after == -1.0:
                reason = f"ECE计算异常({dim})"
                reasons.append(reason)
                print("ece_after计算异常")
                continue
            ece_improve = ece_before - ece_after
            
            if ece_improve < self.ece_improvement_threshold:
                reason = f"验证集ECE提升不足({dim})"
                reasons.append(reason)
                print(f"adid={adid} 维度={dim} 验证集ECE提升不足({ece_improve:.4f} < {self.ece_improvement_threshold})")
                continue
            
            # 如果所有条件都满足，选择该维度
            print(f"adid={adid} 选择最佳校准维度: {dim} (CTR提升={ctr_diff_improve:.4f}, ECE提升={ece_improve:.4f})")
            return dim, reasons
        
        # 如果没有找到满足条件的维度
        if not has_valid_dim:
            reason = "无有效训练维度"
            reasons.append(reason)
        else:
            reason = "所有维度均不满足条件"
            reasons.append(reason)
            
        print(f"adid={adid} {reason}")
        return None, reasons

    def select_best_dimensions(self, train_df, val_df, candidate_dims):
        """为所有adid选择最佳校准维度，增强统计功能"""
        print("开始为每个adid选择最佳校准维度...")
        
        # 重置统计信息
        self.dim_selection_stats = {
            "total_adids": 0,
            "dimension_counts": defaultdict(int),
            "no_dimension_reason": defaultdict(int)
        }
        
        # 获取所有唯一的adid
        all_adids = train_df['ad_id'].unique()
        total_adids = len(all_adids)
        self.dim_selection_stats["total_adids"] = total_adids
        print(f"共发现 {total_adids} 个唯一adid需要处理")
        
        # 为每个adid选择最佳维度
        for adid in all_adids:
            best_dim, reasons = self.select_best_dimension_for_adid(adid, train_df, val_df, candidate_dims)
            self.adid_best_dim_mapping[adid] = best_dim
            
            # 更新统计信息
            if best_dim is not None:
                self.dim_selection_stats["dimension_counts"][best_dim] += 1
            else:
                # 为每个原因计数
                for reason in reasons:
                    self.dim_selection_stats["no_dimension_reason"][reason] += 1
        
        # 打印统计结果
        self._print_dimension_selection_stats()
        
        return self.adid_best_dim_mapping

    def _print_dimension_selection_stats(self):
        """打印维度选择统计结果，包含更多原因分析"""
        total = self.dim_selection_stats["total_adids"]
        if total == 0:
            print("没有adid数据可统计")
            return
            
        print("\n===== 维度选择统计结果 =====")
        print(f"总adid数量: {total}")
        
        # 打印各维度选择情况
        print("\n各维度被选择的数量及占比:")
        dim_counts = self.dim_selection_stats["dimension_counts"]
        for dim, count in sorted(dim_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            print(f"  {dim}: {count} 个 ({percentage:.2f}%)")
        
        # 打印未选择维度的情况
        no_dim_count = total - sum(dim_counts.values())
        no_dim_percentage = (no_dim_count / total) * 100
        print(f"\n未选择任何维度: {no_dim_count} 个 ({no_dim_percentage:.2f}%)")
        
        # 归类原因以便汇总显示
        reason_counts = self.dim_selection_stats["no_dimension_reason"]
        
        # 按原因类型分组统计
        categorized_reasons = defaultdict(int)
        for reason, count in reason_counts.items():
            categorized_reasons[reason] += count
        
        # 打印未选择维度的原因分布
        print("\n未选择维度的原因及占比:")
        for reason, count in sorted(categorized_reasons.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            reason_percentage = (count / no_dim_count) * 100 if no_dim_count > 0 else 0
            print(f"  {reason}: {count} 个 ({percentage:.2f}% 总占比, {reason_percentage:.2f}% 未选择占比)")

    def save_adid_best_dim_mapping(self, file_path=None):
        """保存adid与最佳维度的映射关系到txt文件"""
        if not self.adid_best_dim_mapping:
            print("没有adid最佳维度映射信息可保存")
            return
        
        # 如果未指定路径，使用基础文件名
        if file_path is None:
            file_path = f"{self.base_bin_info_file}_adid_best_dim_mapping.txt"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # 写入表头
                f.write("ad_id|best_dimension\n")
                # 写入每条映射
                for adid, best_dim in self.adid_best_dim_mapping.items():
                    dim_str = best_dim if best_dim is not None else "None"
                    f.write(f"{adid}|{dim_str}\n")
            
            print(f"adid最佳维度映射已保存至: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"保存adid最佳维度映射失败: {str(e)}")
            return None

    def update_bin_info_with_used_flag(self, train_df):
        """
        更新分箱信息文件，标记实际使用的维度分箱
        修改：直接读取文件并更新，避免使用内存存储临时数据
        """
        if not self.adid_best_dim_mapping:
            print("没有adid最佳维度映射信息，无法更新分箱使用标记")
            return
        
        # 遍历每个维度的分箱信息文件
        for dim in self.train_dims:
            dim_sanitized = dim.replace('*', '_')
            info_file = f"{self.base_bin_info_file}_{dim_sanitized}.txt"
            fields = self._parse_dimension(dim)
            
            # 检查文件是否存在
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    pass  # 仅检查文件是否存在
            except FileNotFoundError:
                print(f"分箱信息文件 {info_file} 不存在，跳过更新")
                continue
            
            # 收集该维度中被使用的分组键
            used_group_keys = set()
            
            # 遍历所有adid的最佳维度映射
            for adid, best_dim in self.adid_best_dim_mapping.items():
                if best_dim != dim:
                    continue
                    
                adid_data = train_df[train_df['ad_id'] == adid]
                if len(adid_data) == 0:
                    continue
                    
                group_key = self._get_group_key(adid_data.iloc[0], fields)
                used_group_keys.add(group_key)
            
            # 读取文件内容并更新used标记
            try:
                # 读取所有行
                with open(info_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 第一行是表头，不处理
                header = lines[0]
                content_lines = lines[1:]
                
                # 更新内容行的used标记
                updated_lines = []
                for line in content_lines:
                    line = line.strip()
                    if not line:
                        updated_lines.append(line)
                        continue
                        
                    # 分割分组键和分箱详情
                    parts = line.split('|', 1)
                    if len(parts) < 2:
                        updated_lines.append(line)
                        continue
                        
                    group_key_str, bin_details = parts
                    
                    # 转换回原始的group_key类型
                    if '*' in dim:
                        group_key = tuple(group_key_str.split('|'))
                    else:
                        group_key = group_key_str
                        
                    # 检查是否为被使用的分组
                    if group_key in used_group_keys:
                        # 将used从0改为1
                        bin_parts = bin_details.rsplit(',', 1)
                        if len(bin_parts) == 2:
                            updated_details = f"{bin_parts[0]},1"
                            updated_lines.append(f"{group_key_str}|{updated_details}")
                        else:
                            updated_lines.append(line)
                    else:
                        updated_lines.append(line)
                
                # 写回更新后的内容
                with open(info_file, 'w', encoding='utf-8') as f:
                    f.write(header)
                    f.write('\n'.join(updated_lines) + '\n')
                
                print(f"已更新分箱信息文件 {info_file}，标记了使用状态")
                
            except Exception as e:
                print(f"更新分箱信息文件 {info_file} 失败: {str(e)}")

    def calibrate(self, df, dim_priority=None, use_adid_mapping=False, data_type="test"):
        """
        使用训练好的模型校准测试数据，生成calibrated_prob
        新增：use_adid_mapping参数，是否使用为每个adid选择的最佳维度
        """
        print(f"开始{data_type}数据校准（算法: {self.calibration_method}）：")
        df_calibrated = df.copy()
        # 初始化校准列和跟踪校准来源的列
        df_calibrated["calibrated_prob"] = np.nan
        df_calibrated["calibration_dim"] = "null"  # 记录使用的校准维度
        df_calibrated["calibration_bin_id"] = -1   # 记录使用的箱子ID

        # 如果使用adid映射，则忽略dim_priority
        if use_adid_mapping and self.adid_best_dim_mapping:
            print("使用为每个adid选择的最佳校准维度进行校准")
            return self._calibrate_with_adid_mapping(df_calibrated, data_type)
        
        # 按维度优先级进行校准
        return self._calibrate_by_dim_priority(df_calibrated, dim_priority, data_type)

    def _calibrate_with_adid_mapping(self, df, data_type):
        """使用为每个adid选择的最佳维度进行校准"""
        total_samples = len(df)
        calibrated_count = 0
        uncalibrated_count = total_samples

        # 按adid分组处理
        adid_groups = df.groupby('ad_id', observed=True)
        
        for adid, group_df in adid_groups:
            # 获取为该adid选择的最佳维度
            best_dim = self.adid_best_dim_mapping.get(adid)
            
            if best_dim is None:
                # 没有找到合适的维度，不进行校准
                continue
                
            # 检查维度是否有效
            if best_dim not in self.bin_info:
                print(f"校准{data_type}的时候, adid={adid} 选择的维度 {best_dim} 不存在校准信息，跳过")
                continue
                
            fields = self._parse_dimension(best_dim)
            
            # 检查该维度所需的字段是否存在
            missing_fields = [f for f in fields if f not in df.columns]
            if missing_fields:
                print(f"校准{data_type}的时候, adid={adid} 维度 {best_dim} 所需字段 {missing_fields} 不存在，跳过")
                continue
                
            # 获取该adid在当前维度的分组键
            group_key = self._get_group_key(group_df.iloc[0], fields)
            
                
            # 检查该分组是否有校准信息
            if group_key not in self.bin_info[best_dim] or len(self.bin_info[best_dim][group_key]) == 0:
                print(f"校准{data_type}的时候, adid={adid} 维度 {best_dim} 没有有效的校准信息，跳过")
                continue
                
            # 使用最佳维度进行校准
            bins = self.bin_info[best_dim][group_key]
            group_calibrated = 0
            
            for idx, row in group_df.iterrows():
                current_prob = row["pred_prob"]
                for bin_id, (bin_left, bin_right, calib_value, _) in bins.items():
                    if bin_left <= current_prob < bin_right:
                        df.at[idx, "calibrated_prob"] = calib_value
                        df.at[idx, "calibration_dim"] = best_dim
                        df.at[idx, "calibration_bin_id"] = bin_id
                        group_calibrated += 1
                        break
            
            calibrated_count += group_calibrated
            uncalibrated_count -= group_calibrated
            print(f"校准{data_type}的时候, adid={adid} 使用维度 {best_dim} 校准完成 {group_calibrated} 条记录，剩余未校准: {len(group_df) - group_calibrated}")
        
        # 打印校准统计
        print(f"\n校准{data_type}的时候, 使用adid维度映射校准完成：共校准 {calibrated_count} 条记录，剩余未校准: {uncalibrated_count}")
        
        # 填充未校准的NaN值（用原始pred_prob替代）
        na_mask = df["calibrated_prob"].isna()
        na_count = na_mask.sum()
        if na_count > 0:
            df.loc[na_mask, "calibrated_prob"] = df.loc[na_mask, "pred_prob"]
            percentage = (na_count / total_samples) * 100 if total_samples > 0 else 0
            print(f"{data_type} 数据中有 {na_count} 条记录未匹配到任何校准模型 ({percentage:.2f}%)，用原始预测概率填充")
        
        return df

    def _calibrate_by_dim_priority(self, df_calibrated, dim_priority, data_type):
        """按维度优先级进行校准的子函数"""
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
            uncalibrated_mask = df_calibrated["calibrated_prob"].isna()
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
                    continue

                # 获取分箱信息并进行校准
                bins = self.bin_info[dim][group_key]
                group_calibrated = 0

                for idx, row in group_df.iterrows():
                    current_prob = row["pred_prob"]
                    # 遍历分箱以找到包含当前pred_prob的分箱
                    for bin_id, (bin_left, bin_right, calib_value, _) in bins.items():
                        if bin_left <= current_prob < bin_right:
                            # 记录校准值
                            df_calibrated.at[idx, "calibrated_prob"] = calib_value
                            # 记录校准维度和箱子ID
                            df_calibrated.at[idx, "calibration_dim"] = dim
                            df_calibrated.at[idx, "calibration_bin_id"] = bin_id
                            group_calibrated += 1
                            dim_calibrated_count += 1
                            break

                if group_calibrated > 0:
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

        # 填充未校准的NaN值（用原始pred_prob替代）
        na_mask = df_calibrated["calibrated_prob"].isna()
        na_count = na_mask.sum()
        if na_count > 0:
            df_calibrated.loc[na_mask, "calibrated_prob"] = df_calibrated.loc[na_mask, "pred_prob"]
            percentage = (na_count / total_samples) * 100 if total_samples > 0 else 0
            print(f"\n{data_type} 数据中有 {na_count} 条记录未匹配到任何校准模型 ({percentage:.2f}%)，用原始预测概率填充")
        else:
            print("\n所有记录均已通过校准模型校准，无使用原始预测概率的记录")

        return df_calibrated

    def evaluate(self, df, evaluate_dim=None, is_train_data=False, is_val_data=False, is_test_data=False):
        """评估校准效果：支持按不同维度进行评估"""
        data_type = "训练数据" if is_train_data else "验证数据" if is_val_data else "测试数据" if is_test_data else "未知数据"

        # 确定评估维度
        eval_dims = evaluate_dim if evaluate_dim and isinstance(evaluate_dim, list) and len(evaluate_dim) > 0 else []
        results = {}

        # 计算整体评估
        print(f"开始{data_type}整体评估：")
        overall_df, overall_metrics = self._evaluate_single_dim(df, "overall", data_type)
        results["overall"] = (overall_df, overall_metrics)

        # 按指定维度进行评估
        for dim in eval_dims:
            print(f"开始{data_type}按{dim}维度评估：")
            # 检查维度是否有效
            fields = self._parse_dimension(dim)
            missing_fields = [f for f in fields if f not in df.columns]
            if missing_fields:
                print(f"警告：评估维度 {dim} 所需字段 {missing_fields} 不存在，跳过该维度评估")
                continue

            eval_df, eval_metrics = self._evaluate_single_dim(df, dim, data_type)
            results[dim] = (eval_df, eval_metrics)

        return results

    def _evaluate_single_dim(self, df, dim, data_type):
        """评估单个维度的校准效果"""
        group_dim = dim
        eval_results = []

        if dim == "overall":
            # 整体评估，只有一个分组
            groups = [("overall", df)]
        else:
            # 按维度分组
            fields = self._parse_dimension(dim)
            groups = df.groupby(fields[0], observed=True) if len(fields) == 1 else df.groupby(fields, observed=True)
            # 转换为列表以便处理
            groups = [(k, v) for k, v in groups]

        # 遍历每个分组计算评估指标
        for group_key, group_data in groups:
            # 没有label列则无法评估，跳过
            if "label" not in group_data.columns:
                print(f"{data_type} 警告：{group_dim} 维度的 {group_key} 没有label列，跳过评估")
                continue

            try:
                # 1. 计算分组核心统计量
                group_sample_count = len(group_data)
                group_positive_count = (group_data["label"] == self.positive_class).sum()
                group_negative_count = group_sample_count - group_positive_count
                group_positive_rate = group_positive_count / group_sample_count if group_sample_count > 0 else 0

                # 2. 计算校准前后的ctr_diff
                ctr_diff_improvement, pre_ctr_diff, post_ctr_diff = self._calculate_ctr_diff(group_data, "pred_prob", "calibrated_prob", "label", group_key, data_type)
                
                # 3. 计算ECE
                group_ece = self._calculate_ece(group_data, "pred_prob", "label")
                group_calibrated_ece = self._calculate_ece(group_data, "calibrated_prob", "label")
                ece_improvement = group_ece - group_calibrated_ece

                # 整理该分组的评估结果
                if dim == "overall":
                    result = {
                        "group": "overall",
                        "sample_count": group_sample_count,
                        "positive_count": group_positive_count,
                        "negative_count": group_negative_count,
                        "positive_rate": round(group_positive_rate, 6),
                        "orig_ctr_diff": round(pre_ctr_diff, 6),
                        "calib_ctr_diff": round(post_ctr_diff, 6),
                        "ctr_diff_improvement": round(ctr_diff_improvement, 6),
                        "orig_ece_score": round(group_ece, 6),
                        "calib_ece_score": round(group_calibrated_ece, 6),
                        "ece_improvement": round(ece_improvement, 6)
                    }
                else:
                    result = {
                        group_dim: group_key if not isinstance(group_key, tuple) else "*".join(map(str, group_key)),
                        "sample_count": group_sample_count,
                        "positive_count": group_positive_count,
                        "negative_count": group_negative_count,
                        "positive_rate": round(group_positive_rate, 6),
                        "orig_ctr_diff": round(pre_ctr_diff, 6),
                        "calib_ctr_diff": round(post_ctr_diff, 6),
                        "ctr_diff_improvement": round(ctr_diff_improvement, 6),
                        "orig_ece_score": round(group_ece, 6),
                        "calib_ece_score": round(group_calibrated_ece, 6),
                        "ece_improvement": round(ece_improvement, 6)
                    }

                eval_results.append(result)

            except Exception as e:
                print(f"{data_type} 的 {group_dim} 的 {group_key} 评估失败：{str(e)}, 跳过该分组")
                continue

        # 转换为DataFrame并按sample_count降序排序
        eval_df = pd.DataFrame(eval_results)
        if not eval_df.empty and dim != "overall":
            eval_df = eval_df.sort_values(by="sample_count", ascending=False).reset_index(drop=True)

        # 提取整体评估指标
        overall_metrics = eval_results[0] if eval_results else {}
        overall_metrics[f"{group_dim}_evaluation"] = True

        return eval_df, overall_metrics

def read_classification_data(txt_path, data_type="data", train_dims=None, calibrate_dims=None, evaluate_dims=None):
    """读取TSV格式的分类数据，从第一行获取列名，并展示关键列信息"""
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
            low_memory=False,
            dtype={
                "day": str,
                "pcvr": float,
                "real_cvr": int,
                "ad_id": str,
                "ad_group_id": str,
                "app_id": str,
                "industry_id": str,
                "day": str,
                "alg": str,
                "spent": float,
                "finall_cvrtarget": str
            }
        )

        # 列名映射：将pcvr改为pred_prob，real_cvr改为label
        column_mapping = {
            "pcvr": "pred_prob",
            "real_cvr": "label",
            "finall_cvrtarget": "cvType"
        }
        df = df.rename(columns=column_mapping)

        # 验证所需列是否存在
        required_columns = ["pred_prob", "label", "ad_id"]  # 新增ad_id作为必需列
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据文件缺少必要的列: {', '.join(missing_columns)}")

        # 检查预测概率是否在[0,1]范围内
        prob_out_of_range = ((df["pred_prob"] < 0) | (df["pred_prob"] > 1)).sum()
        if prob_out_of_range > 0:
            print(f"警告：有 {prob_out_of_range} 条记录的预测概率不在[0,1]范围内")

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
        print(f"  正样本数：{sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df):.2%})")
        print(f"  负样本数：{sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df):.2%})")
        print(f"  唯一adid数量：{df['ad_id'].nunique()}")
        
        # 统计不同day的数量及每天的数据量
        day_counts = df['day'].value_counts().sort_index()  # 按day排序顺序排序
        print(f"不同day的数量：{len(day_counts)}")
        print("每天的数据量：")
        for day, count in day_counts.items():
            print(f"  {day}: {count} 行")


        # 打印维度相关字段信息
        if existing_dim_fields:
            print(f"  维度相关字段 ({len(existing_dim_fields)} 个):")
            for field in existing_dim_fields:
                unique_count = df[field].nunique()
                print(f"    {field}: {unique_count} 个不同值")

        # 打印缺失的维度字段警告
        if missing_dim_fields:
            print(f"  警告：以下维度字段不存在于数据中: {', '.join(missing_dim_fields)}")

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
    TRAIN_DATA_PATH = "/Users/11179767/Code/test/py/Calib/data/tmp_fast_pcvr_offline_Calibration_input_v1_data_1014.txt" #!!!
    TEST_DATA_PATH = "/Users/11179767/Code/test/py/Calib/data/tmp_fast_pcvr_offline_Calibration_test_v1_data_1015.txt" #!!!
    # 校准算法选择："histogram_binning"（直方图分箱）或 "isotonic_regression"（保序回归）
    CALIBRATION_METHOD = "histogram_binning"  # 在这里配置选择的校准算法, #!!!
    OUTPUT_TRAIN = f"/Users/11179767/Code/test/py/Calib/result/{CALIBRATION_METHOD}_train_calibrated.txt"
    OUTPUT_VAL = f"/Users/11179767/Code/test/py/Calib/result/{CALIBRATION_METHOD}_val_calibrated.txt"
    OUTPUT_TEST = f"/Users/11179767/Code/test/py/Calib/result/{CALIBRATION_METHOD}_test_calibrated.txt"
    BASE_BIN_INFO_FILE = f"/Users/11179767/Code/test/py/Calib/result/{CALIBRATION_METHOD}_bin_details"  # 分箱/模型信息文件的基础名称
    ADID_BEST_DIM_FILE = f"/Users/11179767/Code/test/py/Calib/result/{CALIBRATION_METHOD}_adid_best_dim.txt"  # adid最佳维度映射文件
    ONLINE_DICT_FILE = f"/Users/11179767/Code/test/py/Calib/result/{CALIBRATION_METHOD}_online_dict.txt"  # 生成在线使用的离线词典
    fast_or_single_game = "f" #!!!
    flow_type_id = 1 #!!!,0-cpd，1-cpc，2-广告联盟

    # 保序回归参数配置
    ISOTONIC_INCREASING = True  # 保序回归是否为递增（预测概率与正例率正相关）
    POSITIVE_CLASS = 1  # 正样本标签

    # 新增配置：评估结果显示的最大行数，-1表示显示所有
    MAX_EVAL_DISPLAY_ROWS = 25
    
    # 新增配置：训练集与验证集拆分比例
    TRAIN_VAL_SPLIT_RATIO = 0.8  # 80%训练，20%验证
    SPLIT_RANDOM_STATE = 42  # 随机种子，保证结果可复现
    
    # 新增配置：评估阈值
    CTR_DIFF_THRESHOLD = 0.001  # CTR差异提升阈值
    ECE_IMPROVEMENT_THRESHOLD = 0.001  # ECE提升阈值
    MIN_POSITIVE_SAMPLES = 0  # 最小正样本数阈值
    MIN_NEGATIVE_SAMPLES = 50  # 最小负样本数阈值
    
    # 新增配置：目标adid，设置为None则处理所有adid
    TARGET_ADID = None  # 例如：None, "239662521"

    # 训练维度配置
    TRAIN_DIMS = ["ad_id", "ad_group_id", "app_id"]  #"ad_id", "ad_group_id", "app_id", "industry_id"
    # 每个训练维度的最小样本数（同时用于保序回归等算法的区间最小样本数）
    DIM_MIN_SAMPLES = {
        "ad_id": 100, #!!!100，1000
        "ad_group_id": 100, #!!!100，1000
        "app_id": 100, #!!!100，1000
    }
    # 校准维度优先级配置（用于维度选择）
    CALIBRATE_DIM_PRIORITY = ["ad_id", "ad_group_id", "app_id"]
    # 评估维度配置
    EVALUATE_DIMS = ["ad_id"] #

    # 2. 读取数据
    all_train_data = read_classification_data(
        TRAIN_DATA_PATH,
        data_type="训练数据",
        train_dims=TRAIN_DIMS,
        calibrate_dims=CALIBRATE_DIM_PRIORITY,
        evaluate_dims=EVALUATE_DIMS
    )
    
    # 如果指定了目标adid，保留该adid相关的数据
    if TARGET_ADID is not None:
        original_train_count = len(all_train_data)
        
        # 获取目标adid在训练数据中TRAIN_DIMS各列的对应值
        target_adid_train_data = all_train_data[all_train_data['ad_id'] == TARGET_ADID].iloc[0] if len(all_train_data[all_train_data['ad_id'] == TARGET_ADID]) > 0 else None
        
        if target_adid_train_data is None:
            print(f"警告：训练数据中未找到adid={TARGET_ADID}的记录，将使用全部训练数据")
        else:
            # 构建筛选条件：只要匹配TRAIN_DIMS中任意一列的对应值即可
            # 初始化为False，使用逻辑或累积条件
            train_filter_conditions = False
            matched_dims = []  # 记录实际匹配的维度
            
            for dim in TRAIN_DIMS:
                # 处理组合维度（如果有的话）
                if '*' in dim:
                    fields = dim.split('*')
                    for field in fields:
                        if field in target_adid_train_data.index:
                            # 对每个字段使用逻辑或，只要有一个匹配即可
                            train_filter_conditions |= (all_train_data[field] == target_adid_train_data[field])
                            matched_dims.append(field)
                else:
                    if dim in target_adid_train_data.index:
                        # 对每个维度使用逻辑或，只要有一个匹配即可
                        train_filter_conditions |= (all_train_data[dim] == target_adid_train_data[dim])
                        matched_dims.append(dim)
            
            # 应用筛选条件
            all_train_data = all_train_data[train_filter_conditions].copy()
            filtered_train_count = len(all_train_data)
            
            if filtered_train_count == 0:
                print(f"警告：训练数据中未找到与adid={TARGET_ADID}在{TRAIN_DIMS}任一维度上匹配的记录，将使用全部训练数据")
            else:
                # 去重并显示实际参与匹配的维度
                unique_matched_dims = list(set(matched_dims))
                print(f"已从{original_train_count}条训练数据中筛选出与adid={TARGET_ADID}在{unique_matched_dims}任一维度上匹配的记录，共 {filtered_train_count} 条")

    # 拆分训练集和验证集（按adid分层抽样）
    print(f"将训练数据按 {TRAIN_VAL_SPLIT_RATIO}:{1-TRAIN_VAL_SPLIT_RATIO} 拆分为训练集和验证集")
    train_df, val_df = train_test_split(
        all_train_data,
        test_size=1-TRAIN_VAL_SPLIT_RATIO,
        random_state=SPLIT_RANDOM_STATE
    )
    print(f"训练集样本数: {len(train_df)}, 验证集样本数: {len(val_df)}")

    # 3. 初始化校准器并设置评估阈值和属性
    calibrator = ClassifierCalibrator(
        calibration_method=CALIBRATION_METHOD,
        base_bin_info_file=BASE_BIN_INFO_FILE,
        increasing=ISOTONIC_INCREASING,
        positive_class=POSITIVE_CLASS
    )
    calibrator.set_evaluation_thresholds(
        ctr_diff=CTR_DIFF_THRESHOLD,
        ece_improve=ECE_IMPROVEMENT_THRESHOLD,
        min_pos=MIN_POSITIVE_SAMPLES,
        min_neg=MIN_NEGATIVE_SAMPLES
    )
    calibrator.set_evaluation_attribute(
        fast_or_single_game=fast_or_single_game,
        flow_type_id=flow_type_id
    )

    # 4. 在训练集上训练模型（可指定目标adid）
    calibrator.train(
        train_df,
        train_dims=TRAIN_DIMS,
        dim_min_samples=DIM_MIN_SAMPLES,
        data_type="train"
    )

    # 5. 在验证集上为每个adid选择最佳校准维度
    calibrator.select_best_dimensions(train_df, val_df, CALIBRATE_DIM_PRIORITY)
    # 保存adid最佳维度映射、更新分箱信息文件的used标记、生成在线使用的词典
    calibrator.save_adid_best_dim_mapping(ADID_BEST_DIM_FILE)
    calibrator.update_bin_info_with_used_flag(train_df)
    calibrator.generate_online_dict(train_df, ONLINE_DICT_FILE)


    # 6. 使用选择的最佳维度校准训练集、验证集和测试集
    train_calibrated = calibrator.calibrate(
        train_df,
        use_adid_mapping=True,
        data_type="train"
    )
    
    val_calibrated = calibrator.calibrate(
        val_df,
        use_adid_mapping=True,
        data_type="validation"
    )

    # 7. 读取并校准测试数据
    test_df = read_classification_data(
        TEST_DATA_PATH,
        data_type="测试数据",
        train_dims=TRAIN_DIMS,
        calibrate_dims=CALIBRATE_DIM_PRIORITY,
        evaluate_dims=EVALUATE_DIMS
    )
    
    # 如果指定了目标adid，仅保留该adid的测试数据
    if TARGET_ADID is not None:
        test_df = test_df[test_df['ad_id'] == TARGET_ADID].copy()
        print(f"已筛选出目标adid={TARGET_ADID}的测试数据，共 {len(test_df)} 条记录")

    test_calibrated = calibrator.calibrate(
        test_df,
        use_adid_mapping=True,
        data_type="test"
    )

    # 8. 评估校准效果
    train_eval_results = calibrator.evaluate(
        train_calibrated,
        evaluate_dim=EVALUATE_DIMS,
        is_train_data=True
    )
    
    val_eval_results = calibrator.evaluate(
        val_calibrated,
        evaluate_dim=EVALUATE_DIMS,
        is_val_data=True
    )

    test_eval_results = calibrator.evaluate(
        test_calibrated,
        evaluate_dim=EVALUATE_DIMS,
        is_test_data=True
    )

    # 9. 输出训练数据评估结果
    print("\n=== 训练数据整体评估结果 ===")
    overall_eval_train, overall_metrics_train = train_eval_results["overall"]
    for key, value in overall_metrics_train.items():
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

    # 10. 输出验证数据评估结果
    print("\n=== 验证数据整体评估结果 ===")
    overall_eval_val, overall_metrics_val = val_eval_results["overall"]
    for key, value in overall_metrics_val.items():
        if key != "overall_evaluation":
            print(f"{key}: {value}")
    
    for dim in EVALUATE_DIMS:
        if dim in val_eval_results:
            print(f"\n=== 验证数据按{dim}维度评估结果 ===")
            eval_df, eval_metrics = val_eval_results[dim]
            if not eval_df.empty:
                # 根据配置决定显示的行数
                if MAX_EVAL_DISPLAY_ROWS > 0 and len(eval_df) > MAX_EVAL_DISPLAY_ROWS:
                    print(eval_df.head(MAX_EVAL_DISPLAY_ROWS).to_string(index=False))
                    print(f"... 仅显示前{MAX_EVAL_DISPLAY_ROWS}行，共{len(eval_df)}行 ...")
                else:
                    print(eval_df.to_string(index=False))


    # 11. 输出测试数据评估结果
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

    # 12. 保存结果
    train_calibrated.to_csv(OUTPUT_TRAIN, sep="\t", index=False, header=True)
    val_calibrated.to_csv(OUTPUT_VAL, sep="\t", index=False, header=True)
    test_calibrated.to_csv(OUTPUT_TEST, sep="\t", index=False, header=True)
    print(f"\n校准结果已保存至：\n{OUTPUT_TRAIN}\n{OUTPUT_VAL}\n{OUTPUT_TEST}")

    # 计算并打印总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n测试数据集时间")
    print(f"\n程序总运行时间：{total_time:.2f}秒")

    # 再次打印整体评估结果
    print("\n===== 最终整体评估结果汇总 =====")
    print(CALIBRATION_METHOD)
    print("\n--- 训练数据整体评估结果 ---")
    for key, value in overall_metrics_train.items():
        if key != "overall_evaluation":
            print(f"{key}: {value}")

    print("\n--- 验证数据整体评估结果 ---")
    for key, value in overall_metrics_val.items():
        if key != "overall_evaluation":
            print(f"{key}: {value}")

    print("\n--- 测试数据整体评估结果 ---")
    for key, value in overall_metrics_test.items():
        if key != "overall_evaluation":
            print(f"{key}: {value}")

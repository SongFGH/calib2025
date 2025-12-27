#版本v2:初始版本

#版本v2d1:#针对v2，优化相关细节

#版本v2d2:针对v2d1，优化相关细节,保存箱子中实际用到的信息

#版本v2d3:针对v2d2，准备开发:打印出来训练、验证以及测试数据的时间和样本量；使用评估指标追加；优化相关细节；生成在线使用词典；离线例行任务；多个文件合并例行
需求：业务指标：偏差分段后，不同段下包/adgroupid/adid对应的数量，消耗占比
    模型指标：ece，field-ece，bias，logloss 、auc/gini、gini
ece: ece-all，ece-adId，ece-groupId， ece-appId，
bias: bias-all，
偏差分布: 不同区间下面adId/groupId/appId对应的数量、消耗以及消耗占比
auc: auc-all
ad-id:ece,bias
在线词典格式：0/1/2^Af/s_cvType_1/2/3/4_adId/groupId/appId^A箱子边界1，箱子边界2，箱子边界3，箱子边界4；纠偏值1，纠偏值2，纠偏值3


#v2d3d1:继承上面的v2d3
#v2d3d2:继承上面的v2d3

v2d3：针对v2d3d1，改动传入参数运行函数

v2d3：针对v2d3d2，改动选取校准维度的规则（新增验证集的ctr也要有提升）；训练集合验证集拆分要正负样本单独拆分；

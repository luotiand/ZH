import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import geom, weibull_min
from scipy.special import gamma

class special_Charts:
    def __init__(self, data, chart_type='G'):
        """
        初始化控制图类
        :param data: 输入的数据，必须是一个Series，表示时间间隔或产品数间隔
        :param chart_type: 控制图类型，'G' 或 'T'
        """
        if not isinstance(data, pd.Series):
            raise ValueError("输入数据必须是一个 pandas Series")
        
        self.data = data
        self.chart_type = chart_type.upper()
        
        if self.chart_type not in ['G', 'T']:
            raise ValueError("chart_type 必须是 'G' 或 'T'")
        
        self.mean = data.mean()
        self.std_dev = data.std()

    def calculate_control_limits(self):
        """
        计算控制限
        """
        n = len(self.data)
        if self.chart_type == 'G':
            # G控制图基于几何分布
            p = (n-1)/(n*(self.mean + 1 ))  # 几何分布的概率参数
            UCL = geom.ppf(0.99865, p)-1  # 上控制限，使用几何分布的分位数函数
            CL = geom.ppf(0.5, p)-1  # 中心线
            LCL = geom.ppf(0.00135, p)-1  # 下控制限
        else:
            # T控00图基于威布尔分布
            shape, loc, scale = weibull_min.fit(self.data, floc=0)  # 拟合威布尔分布
            UCL = weibull_min.ppf(0.99865, shape, loc, scale)  # 上控制限，使用威布尔分布的分位数函数
            CL = scale * gamma(1 + 1/shape)  # 中心线
            LCL = weibull_min.ppf(0.00135, shape, loc, scale)  # 下控制限
        
        return UCL, CL, LCL

    def plot_chart(self):
        """
        绘制控制图
        """
        UCL, CL, LCL = self.calculate_control_limits()

        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data, marker='o', linestyle='-', color='b', label='Data')
        plt.axhline(UCL, color='r', linestyle='--', label='UCL')
        plt.axhline(LCL, color='r', linestyle='--', label='LCL')
        plt.axhline(CL, color='g', linestyle='--', label='Center Line (Mean)')

        plt.title(f'{self.chart_type} Control Chart')
        plt.xlabel('Sample')
        plt.ylabel('Interval')
        plt.legend()
        plt.show()

# 示例数据
data = pd.Series([14,11,36,8,13,3,23,4,8,23,14,5,1,7,8,6,1,18,2,11,4,12,5,6,10,12,0,0,0,0,2,7,6,6,13,16])

# 创建并绘制G控制图
g_chart = special_Charts(data, chart_type='G')
g_chart.plot_chart()

# 创建并绘制T控制图
t_chart = special_Charts(data, chart_type='T')
t_chart.plot_chart()

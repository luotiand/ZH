import numpy as np
import pandas as pd
from scipy import stats

class OutlierDetection:
    def __init__(self, data):
        """
        初始化类
        :param data: numpy array 或 pandas Series，待检测的数据
        """
        self.data = data

    def sigma_rule(self, sigma_level=3):
        """
        使用 sigma 法则检测异常值
        :param sigma_level: int，sigma 水平（默认为 3）
        :return: 异常值
        """
        mean = np.mean(self.data)
        std = np.std(self.data)
        lower_bound = mean - sigma_level * std
        upper_bound = mean + sigma_level * std
        outliers = [x for x in self.data if x < lower_bound or x > upper_bound]
        return outliers

    def boxplot_method(self):
        """
        使用箱线图检测异常值
        :return: 异常值
        """
        q1 = np.percentile(self.data, 25)
        q3 = np.percentile(self.data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [x for x in self.data if x < lower_bound or x > upper_bound]
        return outliers

    def grubbs_test(self, alpha=0.05):
        """
        使用 Grubbs 检验检测异常值
        :param alpha: float，显著性水平（默认为 0.05）
        :return: 异常值
        """
        data = self.data
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        G = max(abs(data - mean)) / std
        t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        G_critical = (n - 1) / np.sqrt(n) * np.sqrt(t_dist ** 2 / (n - 2 + t_dist ** 2))
        outliers = [x for x in data if abs(x - mean) / std > G_critical]
        return outliers

    def dixon_test(self, alpha=0.05):
        """
        使用 Dixon 检验检测异常值
        :param alpha: float，显著性水平（默认为 0.05）
        :return: 异常值
        """
        data = sorted(self.data)
        n = len(data)
        if n < 3 or n > 30:
            raise ValueError("Dixon 检验适用于样本量在 3 到 30 之间的数据")
        q_critical = {
            0.10: [0.941, 0.765, 0.642, 0.560, 0.507, 0.468, 0.437, 0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329, 0.320, 0.313, 0.306, 0.300, 0.295, 0.290, 0.285, 0.281, 0.277, 0.273, 0.269, 0.266, 0.263, 0.260],
            0.05: [0.970, 0.829, 0.710, 0.625, 0.568, 0.526, 0.493, 0.466, 0.444, 0.426, 0.410, 0.396, 0.384, 0.374, 0.364, 0.356, 0.348, 0.341, 0.334, 0.328, 0.322, 0.317, 0.312, 0.307, 0.302, 0.298, 0.294, 0.290],
            0.02: [0.994, 0.926, 0.821, 0.740, 0.682, 0.634, 0.598, 0.568, 0.543, 0.521, 0.503, 0.488, 0.474, 0.462, 0.451, 0.441, 0.432, 0.423, 0.415, 0.408, 0.402, 0.396, 0.391, 0.385, 0.380, 0.376, 0.371, 0.367]
        }
        Q = (data[1] - data[0]) / (data[-1] - data[0])
        q_val = q_critical[alpha][n - 3]
        if Q > q_val:
            return [data[0]]
        Q = (data[-1] - data[-2]) / (data[-1] - data[0])
        if Q > q_val:
            return [data[-1]]
        return []

    def quantile_method(self, lower_quantile=0.05, upper_quantile=0.95):
        """
        使用分位数方法检测异常值
        :param lower_quantile: float，下分位数（默认为 0.05）
        :param upper_quantile: float，上分位数（默认为 0.95）
        :return: 异常值
        """
        lower_bound = np.quantile(self.data, lower_quantile)
        upper_bound = np.quantile(self.data, upper_quantile)
        outliers = [x for x in self.data if x < lower_bound or x > upper_bound]
        return outliers

# 示例用法:
if __name__ == "__main__":
    data = np.array([10, 12, 12, 12, 13, 13, 14, 14, 14, 15, 16, 17, 100])
    detector = OutlierDetection(data)
    
    print("Sigma 法则异常值:", detector.sigma_rule())
    print("箱线图异常值:", detector.boxplot_method())
    print("Grubbs 检验异常值:", detector.grubbs_test())
    print("Dixon 检验异常值:", detector.dixon_test())
    print("分位数异常值:", detector.quantile_method())

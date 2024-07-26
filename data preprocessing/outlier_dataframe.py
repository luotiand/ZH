
import numpy as np
import pandas as pd
from scipy import stats
class OutlierDetection:
    def __init__(self, data):
        """
        初始化类
        :param data: pandas DataFrame，待检测的数据
        """
        if isinstance(data, pd.Series):
            self.data = data.to_frame()
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError("输入数据必须是 pandas Series 或 DataFrame")

    def sigma_rule(self, sigma_level=3):
        """
        使用 sigma 法则检测异常值
        :param sigma_level: int，sigma 水平（默认为 3）
        :return: 异常值的 DataFrame
        """
        outliers = pd.DataFrame()
        for column in self.data.columns:
            mean = self.data[column].mean()
            std = self.data[column].std()
            lower_bound = mean - sigma_level * std
            upper_bound = mean + sigma_level * std
            outliers[column] = self.data[column][(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        return outliers.dropna()

    def boxplot_method(self,IQR =1.5):        
        """
        使用箱线图检测异常值
        :return: 异常值的 DataFrame
        """
        outliers = pd.DataFrame()
        for column in self.data.columns:
            q1 = self.data[column].quantile(0.25)
            q3 = self.data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - IQR * iqr
            upper_bound = q3 + IQR * iqr
            outliers[column] = self.data[column][(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        return outliers.dropna()

    def grubbs_test(self, alpha=0.05):
        """
        使用 Grubbs 检验检测异常值
        :param alpha: float，显著性水平（默认为 0.05）
        :return: 异常值的 DataFrame
        """
        outliers = pd.DataFrame()
        for column in self.data.columns:
            data = self.data[column].values
            n = len(data)
            mean = np.mean(data)
            std = np.std(data)
            G = max(abs(data - mean)) / std
            t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            G_critical = (n - 1) / np.sqrt(n) * np.sqrt(t_dist ** 2 / (n - 2 + t_dist ** 2))
            outliers[column] = self.data[column][abs(data - mean) / std > G_critical]
        return outliers.dropna()

    def dixon_test(self, alpha=0.05):
        """
        使用 Dixon 检验检测异常值
        :param alpha: float，显著性水平（默认为 0.05）
        :return: 异常值的 DataFrame
        """
        outliers = pd.DataFrame()
        for column in self.data.columns:
            data = sorted(self.data[column].values)
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
                outliers[column] = self.data[column][self.data[column] == data[0]]
            else:
                Q = (data[-1] - data[-2]) / (data[-1] - data[0])
                if Q > q_val:
                    outliers[column] = self.data[column][self.data[column] == data[-1]]
                else:
                    outliers[column] = pd.Series([])
        return outliers.dropna()

    def quantile_method(self, lower_quantile=0.05, upper_quantile=0.95):
        """
        使用分位数方法检测异常值
        :param lower_quantile: float，下分位数（默认为 0.05）
        :param upper_quantile: float，上分位数（默认为 0.95）
        :return: 异常值的 DataFrame
        """
        outliers = pd.DataFrame()
        for column in self.data.columns:
            lower_bound = self.data[column].quantile(lower_quantile)
            upper_bound = self.data[column].quantile(upper_quantile)
            outliers[column] = self.data[column][(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        return outliers.dropna()

# 示例用法:
if __name__ == "__main__":
    data = pd.DataFrame({
        'col1': [10, 12, 12, 12, 13, 13, 14, 14, 14, 15, 16, 17, 100],
        'col2': [5, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 50]
    })
    detector = OutlierDetection(data)
    
    print("Sigma 法则异常值:\n", detector.sigma_rule(sigma_level=3))
    print("箱线图异常值:\n", detector.boxplot_method(IQR = 1.5))
    print("Grubbs 检验异常值:\n", detector.grubbs_test(alpha=0.05))
    print("Dixon 检验异常值:\n", detector.dixon_test(alpha=0.05))
    print("分位数异常值:\n", detector.quantile_method(lower_quantile=0.05, upper_quantile=0.95))

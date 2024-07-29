import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm

class ProcessCapability:
    def __init__(self, data):
        self.data = data
        self.mean = np.mean(data)
        self.std_dev = np.std(data,ddof=1)

    def calculate_cpk(self, USL=None, LSL=None, method='s/c4', num_groups=1):
        """
        计算 Cpk 值
        :param USL: float，上规格限
        :param LSL: float，下规格限
        :param method: str，方差计算方法，'r/d2' 或 's/c4' 
        :param num_groups: int，将数据样本分成的组数（默认为 1）
        :return: dict，包含计算结果
        """
        n = len(self.data)
        group_size = n // num_groups

        if group_size == 0:
            raise ValueError("num_groups 太多，导致每组数据样本数不足")

        # 将数据分组
        grouped_data = [self.data[i:i + group_size] for i in range(0, n, group_size)]

        if method == 'r/d2':
            d2 = stats.norm.ppf((1 + (1 - 1 / group_size)) / 2)  # d2 修正因子
            group_std_devs = [(g.max() - g.min()) / d2 for g in grouped_data]
            avg_std_dev = np.mean(group_std_devs)
        elif method == 's/c4':
            c4 = (4 * group_size - 4) / (4 * group_size - 3)  # c4 修正因子
            group_std_devs = [g.std(ddof=1) / c4 for g in grouped_data]
            avg_std_dev = np.mean(group_std_devs)
        else:
            raise ValueError("method 必须是 'r/d2' 或 's/c4'")


        calculations = {
            'n': n,
            'mean': self.mean,
            'std_dev': self.std_dev,
            'avg_std_dev': avg_std_dev,
        }

        if LSL is not None:
            proportion_below_lsl = norm.cdf(LSL, loc=self.mean, scale=self.std_dev)
            ppm_low = proportion_below_lsl * 1e6
            cpl = (self.mean - LSL) / (3 * avg_std_dev)
            ppl = (self.mean - LSL) / (3 * self.std_dev)
            calculations.update({
                'LSL': LSL,
                'ppm_low': ppm_low,
                'cpl': cpl,
                'ppl': ppl,
            })

        if USL is not None:
            proportion_above_usl = 1 - norm.cdf(USL, loc=self.mean, scale=self.std_dev)
            ppm_up = proportion_above_usl * 1e6
            cpu = (USL - self.mean) / (3 * avg_std_dev)
            ppu = (USL - self.mean) / (3 * self.std_dev)
            calculations.update({
                'USL': USL,
                'ppm_up': ppm_up,
                'cpu': cpu,
                'ppu': ppu,
            })

        if USL is not None and LSL is not None:
            cp = (USL - LSL) / (6 * avg_std_dev)
            pp = (USL - LSL) / (6 * self.std_dev)
            cpk = min(cpu, cpl)
            ppk = min(ppu, ppl)
            ca = (self.mean-(USL+LSL)/2)/((USL-LSL)/2)
            calculations.update({
                'cp': cp,
                'pp': pp,
                'cpk': cpk,
                'ppk': ppk,
                'ca':ca,
            })

        return calculations

if __name__ == "__main__":
# 示例数据
    

# 多行数据，分隔为单独的列表
    data = [
        [0.297, 0.336, 0.291, 0.270, 0.285, 0.351, 0.305, 0.350, 0.301, 0.320, 0.289, 0.269, 0.292, 0.306, 0.337, 0.329, 0.312, 0.328, 0.306, 0.293, 0.319, 0.318, 0.328, 0.319, 0.316, 0.293, 0.285, 0.283, 0.334, 0.299],
        [0.319, 0.268, 0.291, 0.282, 0.332, 0.282, 0.296, 0.298, 0.337, 0.290, 0.339, 0.298, 0.311, 0.308, 0.301, 0.292, 0.322, 0.347, 0.329, 0.288, 0.316, 0.342, 0.320, 0.316, 0.291, 0.310, 0.292, 0.277, 0.336, 0.305],
        [0.304, 0.304, 0.336, 0.317, 0.348, 0.325, 0.288, 0.292, 0.307, 0.291, 0.328, 0.344, 0.290, 0.299, 0.296, 0.298, 0.322, 0.312, 0.307, 0.312, 0.298, 0.306, 0.272, 0.280, 0.284, 0.284, 0.303, 0.333, 0.269, 0.328],
        [0.356, 0.310, 0.297, 0.305, 0.300, 0.303, 0.334, 0.317, 0.330, 0.282, 0.260, 0.312, 0.256, 0.282, 0.319, 0.298, 0.293, 0.318, 0.318, 0.312, 0.323, 0.298, 0.308, 0.306, 0.307, 0.309, 0.296, 0.300, 0.319, 0.312],
        [0.288, 0.300, 0.303, 0.299, 0.278, 0.268, 0.299, 0.292, 0.290, 0.328, 0.319, 0.322, 0.319, 0.314, 0.322, 0.300, 0.308, 0.324, 0.302, 0.277, 0.318, 0.310, 0.306, 0.308, 0.325, 0.290, 0.305, 0.302, 0.266, 0.305]
    ]

    # 按列展平数据
    flattened_data = [item for col in zip(*data) for item in col]

    # 转换为 pandas Series
    data = pd.Series(flattened_data)

    print(data)

    USL = 0.4  # 上规格限
    LSL = 0.2  # 下规格限
    num_groups = 30 # 分成 2 组

    process_capability = ProcessCapability(data)
    cpk_value_r_d2 = process_capability.calculate_cpk(USL, LSL, method='r/d2', num_groups=num_groups)
    print(f'Cpk (R/d2): {cpk_value_r_d2}')

    cpk_value_s_c4 = process_capability.calculate_cpk(USL, LSL, method='s/c4', num_groups=num_groups)
    print(f'Cpk (S/c4): {cpk_value_s_c4}')

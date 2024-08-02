import math
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm

class ProcessCapability:
    def __init__(self, data,num_groups):
        self.data = data
        self.mean = np.mean(data)
        self.std_dev = np.std(data,ddof=1)
        self.num_groups=num_groups

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
        group_size = n // self.num_groups

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
    
    
   
    def plot_xr_chart(self):
        df = pd.DataFrame({'data': self.data})
        n = len(self.data)
        sample_size = n // self.num_groups
        # 将数据划分为样本
        samples = [data[i:i + sample_size] for i in range(0, len(data), sample_size)]

        # 计算每个样本的均值和极差
        x_bars = [np.mean(sample) for sample in samples]
        r_values = [np.max(sample) - np.min(sample) for sample in samples]

        # 计算均值和平均极差
        x_bar = np.mean(x_bars)
        r_bar = np.mean(r_values)

        # 控制常数
        A2 = {
            2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483,
            7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308, 11: 0.286, 12: 0.266,
            13: 0.249, 14: 0.235, 15: 0.223, 16: 0.213, 17: 0.204, 18: 0.196,
            19: 0.189, 20: 0.182, 21: 0.176, 22: 0.171, 23: 0.166, 24: 0.161,
            25: 0.157, 26: 0.153, 27: 0.149, 28: 0.146, 29: 0.143, 30: 0.140,
            31: 0.137, 32: 0.134, 33: 0.132, 34: 0.129, 35: 0.127, 36: 0.125,
            37: 0.123, 38: 0.121, 39: 0.120, 40: 0.118
        }.get(sample_size)  # 默认适用于样本量为5

        D3 = {
            2: 0, 3: 0, 4: 0, 5: 0, 6: 0.076, 7: 0.136, 8: 0.184, 9: 0.223, 10: 0.256,
            11: 0.278, 12: 0.295, 13: 0.311, 14: 0.325, 15: 0.338, 16: 0.350, 17: 0.361,
            18: 0.370, 19: 0.379, 20: 0.386, 21: 0.393, 22: 0.399, 23: 0.405, 24: 0.410,
            25: 0.415, 26: 0.419, 27: 0.423, 28: 0.427, 29: 0.430, 30: 0.433, 31: 0.436,
            32: 0.439, 33: 0.442, 34: 0.444, 35: 0.446, 36: 0.448, 37: 0.450, 38: 0.452,
            39: 0.453, 40: 0.455
        }.get(sample_size)  

        D4 = {
            2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 
            9: 1.816, 10: 1.777, 11: 1.746, 12: 1.718, 13: 1.694, 14: 1.672, 
            15: 1.652, 16: 1.634, 17: 1.618, 18: 1.604, 19: 1.591, 20: 1.579, 
            21: 1.568, 22: 1.558, 23: 1.549, 24: 1.541, 25: 1.534, 26: 1.527, 
            27: 1.521, 28: 1.516, 29: 1.511, 30: 1.507, 31: 1.503, 32: 1.500, 
            33: 1.497, 34: 1.495, 35: 1.493, 36: 1.491, 37: 1.490, 38: 1.489, 
            39: 1.488, 40: 1.488
        }.get(sample_size)  

        # 计算控制界限
        UCLx = x_bar + A2 * r_bar
        LCLx = x_bar - A2 * r_bar
        UCLr = D4 * r_bar
        LCLr = D3 * r_bar
        calculations = {
            'x_bar': x_bar,
            'r_bar': r_bar,
            'mean': self.mean,
            'std_dev': self.std_dev,
            'UCLx': UCLx,
            'LCLr': LCLr,
        }

        # 绘制 X-bar 图
        plt.figure(figsize=(14, 7))
        plt.subplot(2, 1, 1)
        plt.plot(x_bars, marker='o', linestyle='-')
        plt.axhline(x_bar, color='r', linestyle='--', label='Center Line (X-bar)')
        plt.axhline(UCLx, color='g', linestyle='--', label='UCL (X-bar)')
        plt.axhline(LCLx, color='b', linestyle='--', label='LCL (X-bar)')
        plt.title('X-bar Chart')
        plt.xlabel('Sample')
        plt.ylabel('X-bar')
        plt.legend()

        # 绘制 R 图
        plt.subplot(2, 1, 2)
        plt.plot(r_values, marker='o', linestyle='-')
        plt.axhline(r_bar, color='r', linestyle='--', label='Center Line (R)')
        plt.axhline(UCLr, color='g', linestyle='--', label='UCL (R)')
        plt.axhline(LCLr, color='b', linestyle='--', label='LCL (R)')
        plt.title('R Chart')
        plt.xlabel('Sample')
        plt.ylabel('Range')
        plt.legend()

        plt.tight_layout()
        plt.show()
        return calculations
        

    def plot_xs_chart(self):
        n = len(self.data)
        df = pd.DataFrame({'data': self.data})
        group_size = n // self.num_groups

        if group_size == 0:
            raise ValueError("num_groups 太多，导致每组数据样本数不足")

        # 将数据分组
        grouped_data = [self.data[i:i + group_size] for i in range(0, n, group_size)]
        sample_size = n // self.num_groups
        samples = [data[i:i + sample_size] for i in range(0, len(data), sample_size)]
        x_bars = [np.mean(sample) for sample in samples]
        group_std_devs = [g.std(ddof=1) for g in grouped_data]
        avg_std_dev = np.mean(group_std_devs)
        
        s_values = group_std_devs
        x_bar = np.mean(x_bars)
        s_bar = avg_std_dev
        


        B3 = {
            2: 0, 3: 0, 4: 0, 5: 0, 6: 0.03, 
            7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284,
            11: 0.321, 12: 0.354, 13: 0.382, 14: 0.406, 15: 0.428,
            16: 0.448, 17: 0.466, 18: 0.482, 19: 0.497, 20: 0.510,
            21: 0.523, 22: 0.534, 23: 0.545, 24: 0.555,25:0.565
        }.get(sample_size, 0)  # 默认适用于样本量为2

        B4 = {
            2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 
            7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716, 11: 1.679,
            12: 1.646, 13: 1.618, 14: 1.594, 15: 1.572, 16: 1.552,
            17: 1.543, 18: 1.518, 19: 1.503, 20: 1.490,
            21: 1.477, 22: 1.466, 23: 1.455, 24: 1.445,25:1.435
        }.get(sample_size, 3.267)  # 默认适用于样本量为2

        A3 = {
            2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 
            7: 1.182, 8: 1.099, 9: 1.032, 10: 0.975, 11: 0.927,
            12: 0.886, 13: 0.850, 14: 0.817, 15: 0.789, 16: 0.763,
            17: 0.739, 18: 0.718, 19: 0.698, 20: 0.680,
            21: 0.663, 22: 0.647, 23: 0.633, 24: 0.619,25:0.606
        }.get(sample_size, 0.729)  # 默认适用于样本量为2

        UCLx = x_bar + A3 * s_bar
        LCLx = x_bar - A3 * s_bar
        UCLs = B4 * s_bar
        LCLs = B3 * s_bar
        calculations = {
            'x_bar': x_bar,
            's_bar': s_bar,
            'mean': self.mean,
            'std_dev': self.std_dev,
            'UCLx': UCLx,
            'LCLs': LCLs,
        }

        plt.figure(figsize=(14, 7))
        plt.subplot(2, 1, 1)
        plt.plot(x_bars, marker='o', linestyle='-')
        plt.axhline(x_bar, color='r', linestyle='--', label='Center Line (X-bar)')
        plt.axhline(UCLx, color='g', linestyle='--', label='UCL (X-bar)')
        plt.axhline(LCLx, color='b', linestyle='--', label='LCL (X-bar)')
        plt.title('X-bar Chart')
        plt.xlabel('Sample')
        plt.ylabel('X-bar')
        plt.legend()

        # 绘制 R 图
        plt.subplot(2, 1, 2)
        plt.plot(s_values, marker='o', linestyle='-')
        plt.axhline(s_bar, color='r', linestyle='--', label='Center Line (R)')
        plt.axhline(UCLs, color='g', linestyle='--', label='UCL (S)')
        plt.axhline(LCLs, color='b', linestyle='--', label='LCL (S)')
        plt.title('S Chart')
        plt.xlabel('Sample')
        plt.ylabel('Range')
        plt.legend()

        plt.tight_layout()
        plt.show()
        return calculations
    def plot_imr_chart(self):
        df = pd.DataFrame({'data': self.data})
        n = len(df)
        sample_size = n // self.num_groups
        moving_range = df['data'].diff().abs()[1:]

        x_bar = df['data'].mean()
        mr_bar = moving_range.mean()

        A2 = {
            2: 1.88, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483,
            7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308, 11: 0.287,
            12: 0.270, 13: 0.256, 14: 0.243, 15: 0.232, 16: 0.222,
            17: 0.213, 18: 0.205, 19: 0.198, 20: 0.191,
            25: 0.155, 30: 0.141, 35: 0.133, 40: 0.127
        }.get(sample_size, 0.577)  # 默认适用于样本量为5

        d3 = {
            2: 0, 3: 0, 4: 0, 5: 0, 6: 0.076,
            7: 0.136, 8: 0.184, 9: 0.223, 10: 0.256, 11: 0.284,
            12: 0.308, 13: 0.329, 14: 0.347, 15: 0.362, 16: 0.375,
            17: 0.387, 18: 0.397, 19: 0.407, 20: 0.416,
            25: 0.447, 30: 0.468, 35: 0.482, 40: 0.492
        }.get(sample_size, 0)  # 默认适用于样本量为2

        d4 = {
            2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004,
            7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777, 11: 1.746,
            12: 1.721, 13: 1.699, 14: 1.681, 15: 1.665, 16: 1.651,
            17: 1.639, 18: 1.628, 19: 1.619, 20: 1.611,
            25: 1.576, 30: 1.550, 35: 1.531, 40: 1.515
        }.get(sample_size, 3.267)  # 默认适用于样本量为2

        # 使用 MR（移动范围）图计算控制限
        UCLx = x_bar + 3 * mr_bar / np.sqrt(2)
        LCLx = x_bar - 3 * mr_bar / np.sqrt(2)
        UCLmr = mr_bar * d4
        LCLmr = mr_bar * d3
        calculations = {
            'x_bar': x_bar,
            'mr_bar': mr_bar,
            'mean': self.mean,
            'std_dev': self.std_dev,
            'UCLx': UCLx,
            'LCLmr': LCLmr,
        }


        plt.figure(figsize=(10, 6))
        plt.subplot(211)
        plt.plot(df.index, df['data'], marker='o', linestyle='-', color='b', label='Individual Values')
        plt.axhline(x_bar, color='g', linestyle='--', label='Center Line (X̄)')
        plt.axhline(UCLx, color='r', linestyle='--', label='UCL')
        plt.axhline(LCLx, color='r', linestyle='--', label='LCL')
        plt.title('I Chart')
        plt.xlabel('Sample')
        plt.ylabel('Individual Values')
        plt.legend()

        plt.subplot(212)
        plt.plot(moving_range.index, moving_range, marker='o', linestyle='-', color='b', label='Moving Range')
        plt.axhline(mr_bar, color='g', linestyle='--', label='Center Line (MR̄)')
        plt.axhline(UCLmr, color='r', linestyle='--', label='UCL')
        plt.axhline(LCLmr, color='r', linestyle='--', label='LCL')
        plt.title('MR Chart')
        plt.xlabel('Sample')
        plt.ylabel('Moving Range')
        plt.legend()
        plt.tight_layout()
        plt.show()
        return calculations
    

    def _calculate_cp_cpk(self,usl,lsl):
        cp = (usl -lsl) / (6 * self.std_dev)
        cpu = (usl - self.mean) / (3 * self.std_dev)
        cpl = (self.mean - lsl) / (3 * self.std_dev)
        cpk = min(cpu, cpl)
        return cp, cpk

    def plot_capability_chart(self,usl,lsl):
        cp, cpk = self._calculate_cp_cpk(usl,lsl)

        # 绘制直方图
        plt.figure(figsize=(12, 6))
        plt.hist(self.data, bins=30, alpha=0.7, color='b', edgecolor='black', density=True)

        # 绘制正态分布曲线
        x = np.linspace(min(self.data), max(self.data), 1000)
        p = norm.pdf(x, self.mean, self.std_dev)
        plt.plot(x, p, 'k', linewidth=2)

        # 标记USL和LSL
        plt.axvline(usl, color='r', linestyle='--', label='USL')
        plt.axvline(lsl, color='r', linestyle='--', label='LSL')

        # 标记均值
        plt.axvline(self.mean, color='g', linestyle='-', label='Mean')

        # 显示Cp和Cpk值
        plt.text(self.mean, max(p) * 0.8, f'Cp = {cp:.2f}\nCpk = {cpk:.2f}', 
                 horizontalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        plt.title('Process Capability Chart')
        plt.xlabel('Measurements')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    def plot_moving_average_chart(self,size = 3):
        """
        绘制移动平均控制图，包括数据点、移动平均线和控制界限。
        """
        df = pd.DataFrame({'data': self.data})
        n = len(df)
        sample_size = n // self.num_groups
        df['group'] = (df.index // sample_size) + 1
        group_means = df.groupby('group')['data'].mean().reset_index()
        group_means = group_means.rename(columns={'data': 'group_mean'})

        # 初始化加权移动平均序列
        moving_avg = group_means['group_mean'].rolling(window=size).mean()
        mv = pd.DataFrame(moving_avg)
        # 计算移动平均线的控制限
        mean_ma = moving_avg.mean()
        std_dev_ma = moving_avg.std()
        d2 = {
            1:1.128,
            2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
            7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078, 11: 3.173,
            12: 3.258, 13: 3.336, 14: 3.407, 15: 3.472, 16: 3.532,
            17: 3.588, 18: 3.640, 19: 3.689, 20: 3.735,
            25: 3.971, 30: 4.136, 35: 4.251, 40: 4.337
            }.get(sample_size) 
        # UCL = mean + L * std_dev * ((lambda_value / (2 - lambda_value)) * (1 - (1 - lambda_value)**(2 * (np.arange(1, n+1)))))**0.5
        # LCL = mean - L * std_dev * ((lambda_value / (2 - lambda_value)) * (1 - (1 - lambda_value)**(2 * (np.arange(1, n+1)))))**0.5
        UCL = mean_ma + 3 * std_dev_ma
        LCL = mean_ma - 3 * std_dev_ma

        # 绘制移动平均图
        plt.figure(figsize=(12, 6))
        plt.plot(moving_avg.index, moving_avg,  marker='o', linestyle='-', color='g', label='Moving Average')
        plt.axhline(UCL, linestyle='--', color='r', label='UCL')
        plt.axhline(LCL, linestyle='--', color='r', label='LCL')
        plt.axhline(mean_ma, color='r', linestyle='--', label='Mean')

        plt.title('Moving Average Control Chart')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        calculations = {
            'mean_ma': mean_ma,
            'std_dev_ma': std_dev_ma,
            'UCL': UCL,
            'LCL': LCL,
        }
        return calculations
    
    def plot_standard_chart(self):
        df = pd.DataFrame({'data': self.data})
        n = len(df)
        sample_size = n // self.num_groups
        
        # 如果样本数量不能整除分组数量，则最后一组的样本数可能会少于其他组
        df['group'] = (df.index // sample_size) + 1
        group_means = df.groupby('group')['data'].mean().reset_index()
        group_means = group_means.rename(columns={'data': 'group_mean'})

        # 标准化组均值
        group_mean_overall_mean = group_means['group_mean'].mean()
        group_mean_overall_std = group_means['group_mean'].std()
        group_means['group_mean_standard'] = (group_means['group_mean'] - group_mean_overall_mean) / group_mean_overall_std * np.sqrt(sample_size)

        # 查找 d2 值
        d2 = {1: 1.128, 
            2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
            7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078, 11: 3.173,
            12: 3.258, 13: 3.336, 14: 3.407, 15: 3.472, 16: 3.532,
            17: 3.588, 18: 3.640, 19: 3.689, 20: 3.735,
            25: 3.971, 30: 4.136, 35: 4.251, 40: 4.337
        }.get(sample_size)

        # 计算标准化后的 UCL 和 LCL
        UCL = group_means['group_mean_standard'].mean() + 3
        LCL = group_means['group_mean_standard'].mean() - 3

        # 绘制标准化控制图
        plt.figure(figsize=(12, 6))
        plt.plot(group_means.index, group_means['group_mean_standard'], marker='o', linestyle='-', color='g', label='standard')
        plt.axhline(UCL, linestyle='--', color='r', label='UCL')
        plt.axhline(LCL, linestyle='--', color='r', label='LCL')
        plt.axhline(group_means['group_mean_standard'].mean(), color='r', linestyle='--', label='Mean')
        plt.title('Standard Control Chart')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        calculations = {
            'UCL': UCL,
            'LCL': LCL,
        }
        return calculations

    
    def plot_ewma_chart(self, r=0.2):
        """
        绘制移动平均控制图，包括数据点、移动平均线和控制界限。
        """
        df = pd.DataFrame({'data': self.data})
        n = len(df)
        sample_size = n // self.num_groups
        df['group'] = (df.index // sample_size) + 1
        group_means = df.groupby('group')['data'].mean().reset_index()
        group_means = group_means.rename(columns={'data': 'group_mean'})

        # 初始化加权移动平均序列
        avg = group_means['group_mean'].mean() 
        moving_avg = [self.data[0]]
        # 计算后续值
        for i in range(1, len(group_means)):
            prev_z = moving_avg[-1]
            current_x = group_means['group_mean'].iloc[i]
            new_z = prev_z * (1 - r) + current_x * r
            moving_avg.append(new_z)

        # 将结果添加到 DataFrame 中
        group_means['moving_avg'] = moving_avg
        # 计算移动平均线的控制限
        mean_ma = group_means['moving_avg'].mean()
        std_dev_ma =group_means['moving_avg'].std()
        d2 = {1: 1.128,
            2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
            7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078, 11: 3.173,
            12: 3.258, 13: 3.336, 14: 3.407, 15: 3.472, 16: 3.532,
            17: 3.588, 18: 3.640, 19: 3.689, 20: 3.735,
            25: 3.971, 30: 4.136, 35: 4.251, 40: 4.337
            }.get(sample_size) 
        # UCL = mean + L * std_dev * ((lambda_value / (2 - lambda_value)) * (1 - (1 - lambda_value)**(2 * (np.arange(1, n+1)))))**0.5
        # LCL = mean - L * std_dev * ((lambda_value / (2 - lambda_value)) * (1 - (1 - lambda_value)**(2 * (np.arange(1, n+1)))))**0.5

        group_means['UCL'] = mean_ma + 3 * std_dev_ma*np.sqrt((r/(2-r))*(1-(1-r)**(2*group_means.index)))
        group_means['LCL'] = mean_ma - 3 * std_dev_ma*np.sqrt((r/(2-r))*(1-(1-r)**(2*group_means.index)))
        UCL = mean_ma + 3 * std_dev_ma
        LCL = mean_ma - 3 * std_dev_ma

        # 绘制移动平均图
        plt.figure(figsize=(12, 6))
        plt.plot(group_means.index, group_means['moving_avg'],  marker='o', linestyle='-', color='g', label='EWMA')
        plt.step(group_means.index, group_means['UCL'] , where='pre', linestyle='--', color='r', label='UCL')
        plt.step(group_means.index, group_means['LCL'] , where='pre', linestyle='--', color='r', label='LCL')
        plt.axhline(mean_ma, color='r', linestyle='--', label='Mean')

        plt.title('EWMA Control Chart')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        calculations = {
            'mean_ma': mean_ma,
            'std_dev_ma': std_dev_ma,
            'UCL': UCL,
            'LCL': LCL,
        }
        return calculations
    
    def _calculate_cv(self):
        return self.std_dev / self.mean

    def plot_cv_control_chart(self, mean_known=False, std_known=False, mean_value=None, std_value=None):
        if mean_known and mean_value is None:
            raise ValueError("Mean value must be provided if mean is known.")
        if std_known and std_value is None:
            raise ValueError("Standard deviation value must be provided if standard deviation is known.")
        df = pd.DataFrame({'data': self.data})
        n = len(df)
        sample_size = n // self.num_groups
        df['group'] = (df.index // sample_size) + 1
        group_stats = df.groupby('group')['data'].agg(['mean', 'std']).reset_index() 
        # df = pd.DataFrame({'data': self.data})
        # n = len(df)
        # sample_size = n // self.num_groups
        C2 = {1: 0.7979, 
        2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.94, 6: 0.9515,
        7: 0.9594, 8: 0.965, 9: 0.9693, 10: 0.9727, 11: 0.9754,
        12: 0.9776, 13: 0.9794, 14: 0.981, 15: 0.9823, 16: 0.9835,
        17: 0.9845, 18: 0.9854, 19: 0.9862, 20: 0.9869,
        25: 0.989, 30: 0.9904, 35: 0.9914, 40: 0.9922
        }.get(sample_size)

        cv_values = C2*group_stats['std']/group_stats['mean']

        if mean_known and std_known:
            mean_cv = std_value / mean_value
        else:
            mean_cv = cv_values.mean()

        if std_known:
            std_dev_cv = std_value / mean_value
        else:
            std_dev_cv = cv_values.std()

        B3_table = {1:0,
        2: 0, 3: 0, 4: 0, 5: 0, 6: 0.030,
        7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284,
        11: 0.322, 12: 0.354, 13: 0.382, 14: 0.406,
        15: 0.428, 16: 0.448, 17: 0.466, 18: 0.482,
        19: 0.497, 20: 0.510, 25: 0.565, 30: 0.605,
        35: 0.634, 40: 0.659
    }

        B4_table = {1:3.267,
        2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970,
        7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716,
        11: 1.679, 12: 1.646, 13: 1.618, 14: 1.594,
        15: 1.572, 16: 1.553, 17: 1.536, 18: 1.521,
        19: 1.508, 20: 1.496, 25: 1.435, 30: 1.388,
        35: 1.352, 40: 1.322
    }


        B3 = B3_table.get(sample_size)
        B4 = B4_table.get(sample_size)

        UCL = B4 * mean_cv
        LCL = max(0, B3 * mean_cv)  # 控制下限不能小于0

        # 绘制CV控制图
        plt.figure(figsize=(12, 6))
        plt.plot(cv_values.index, cv_values, marker='o', linestyle='-', color='b', label='CV')
        plt.axhline(mean_cv, color='r', linestyle='--', label='Mean CV')
        plt.axhline(UCL, color='r', linestyle='--', label='UCL')
        plt.axhline(LCL, color='r', linestyle='--', label='LCL')
        plt.title('CV Control Chart')
        plt.xlabel('Sample')
        plt.ylabel('CV')
        plt.legend()
        plt.show()

        calculations = {
            'mean_cv': mean_cv,
            'std_dev_cv': std_dev_cv,
            'UCL': UCL,
            'LCL': LCL,
        }
        return calculations
    
    def plot_cusum_chart(self):
        """
        绘制CUSUM控制图，包括累积和和控制界限
        """
        group_size = len(self.data) // self.num_groups
        df = pd.DataFrame({'data': self.data})
        n = len(df)
        sample_size = n // self.num_groups
        df['group'] = (df.index // sample_size) + 1
        group_stats = df.groupby('group')['data'].agg(['mean', 'std']).reset_index() 
        a = self.std_dev*math.sqrt(group_size)
        if a < 0.75:
            f,h = 8,0.25
        elif a>1.5:
            f,h = 2.5,1
        else:
            f,h = 5,0.5
        F = f * self.std_dev
        H = h * self.std_dev
        K1 = self.mean + F
        K2 = self.mean - F
        group_stats['adjusted_mean1'] = group_stats['mean'] - K1
        group_stats['adjusted_mean2'] = group_stats['mean'] - K2
        group_stats['cumsum_adjusted_mean1'] = group_stats['adjusted_mean1'].cumsum()
        group_stats['cumsum_adjusted_mean2'] = group_stats['adjusted_mean2'].cumsum()
        calculations = {
            'K1': K1,
            'K2': K2,
        }

        plt.figure(figsize=(12, 6))
        plt.plot(group_stats.index,group_stats['cumsum_adjusted_mean1'], marker='o', linestyle='-', color='b', label='CUSUM-')
        plt.plot(group_stats.index,group_stats['cumsum_adjusted_mean2'], marker='o', linestyle='-', color='r', label='CUSUM+')
        plt.axhline(H, color='g', linestyle='--', label='UCL')
        plt.axhline(-H, color='g', linestyle='--', label='LCL')
        plt.axhline(0, color='k', linestyle='-', label='Target')
        plt.title('CUSUM Control Chart')
        plt.xlabel('Sample')
        plt.ylabel('Cumulative Sum')
        plt.legend()
        plt.show()
        return calculations
        

    # def _get_characteristic_value(self,P0,P1):
    #     """
    #     根据 P1 / P0 查找特性值
    #     """
    #     ratio = P1 /P0
    #     characteristic_values = {
    #         1.1: 0.4,
    #         1.2: 0.5,
    #         1.3: 0.6,
    #         1.4: 0.7,
    #         1.5: 0.8
    #     }
    #     return characteristic_values.get(ratio, 0.5)  # 默认值为0.5

    # def _get_T0(self, characteristic_value):
    #     """
    #     根据特性值查找 T0
    #     """
    #     T0_values = {
    #         0.4: 20,
    #         0.5: 25,
    #         0.6: 30,
    #         0.7: 35,
    #         0.8: 40
    #     }
    #     return T0_values.get(characteristic_value, 25)  # 默认值为25

    # def _calculate_cusum(self, K1, K2):
    #     """
    #     计算CUSUM控制图的累积和
    #     """
    #     cusum_pos = np.zeros(len(self.data))
    #     cusum_neg = np.zeros(len(self.data))

    #     for i in range(1, len(self.data)):
    #         cusum_pos[i] = max(0, cusum_pos[i - 1] + self.data[i] - K1)
    #         cusum_neg[i] = min(0, cusum_neg[i - 1] + self.data[i] - K2)

    #     return cusum_pos, cusum_neg

    # def plot_count_cusum_chart(self,P0 ,P1):
    #     """
    #     绘制计数型CUSUM控制图，包括累积和和控制界限
    #     """
    #     characteristic_value = self._get_characteristic_value(P0,P1)
    #     T0 = self._get_T0(characteristic_value)
    #     sample_size = T0 /P0

    #     group_size = len(self.data) // self.num_groups
    #     a = self.std_dev * np.sqrt(group_size)
        
    #     if a < 0.75:
    #         f, h = 8, 0.25
    #     elif a > 1.5:
    #         f, h = 2.5, 1
    #     else:
    #         f, h = 5, 0.5
        
    #     F = f * self.std_dev
    #     H = h * self.std_dev
    #     K1 = self.mean + F
    #     K2 = self.mean - F

    #     cusum_pos, cusum_neg = self._calculate_cusum(K1, K2)

    #     calculations = {
    #         'K1': K1,
    #         'K2': K2,
    #         'cusum_pos': cusum_pos,
    #         'cusum_neg': cusum_neg,
    #         'sample_size': sample_size
    #     }

    #     plt.figure(figsize=(12, 6))
    #     plt.plot(cusum_pos, marker='o', linestyle='-', color='b', label='CUSUM+')
    #     plt.plot(cusum_neg, marker='o', linestyle='-', color='r', label='CUSUM-')
    #     plt.axhline(H, color='g', linestyle='--', label='UCL')
    #     plt.axhline(-H, color='g', linestyle='--', label='LCL')
    #     plt.axhline(0, color='k', linestyle='-', label='Target')
    #     plt.title('CUSUM Control Chart')
    #     plt.xlabel('Sample')
    #     plt.ylabel('Cumulative Sum')
    #     plt.legend()
    #     plt.show()
        
        return calculations
    
    def plot_chain_chart(self):
        """
        绘制链图（Cumulative Sum Control Chart）
        """
        df = pd.DataFrame({'data': self.data})
        n = len(df)
        sample_size = n // self.num_groups
        df['group'] = (df.index // sample_size) + 1
        group_stats = df.groupby('group')['data'].agg(['mean', 'std']).reset_index() 
        chain_sum = np.cumsum(group_stats['mean'] - self.mean)

        plt.figure(figsize=(12, 6))
        plt.plot(chain_sum, marker='o', linestyle='-', color='b', label='Chain Sum')
        plt.axhline(0, color='k', linestyle='-', label='Target')
        plt.title('Chain Chart')
        plt.xlabel('Sample')
        plt.ylabel('Cumulative Sum')
        plt.legend()
        plt.show()
        
        return chain_sum




if __name__ == "__main__":
# 示例数据
    

# 多行数据，分隔为单独的列表
    data = [
       18.5,
20.9,
20.8,
19.5,
20.7,
21.1,
19.1,
19.8,
20.2,
20.2,
19.7,
21.2,
20.4,
21,
19.4,
20.2,
21.3,
20.3,
22.1,
21.5,
20,
20.7,
21.7,
22.2,
22.9

    ]

    # 按列展平数据
    # flattened_data = [item for col in zip(*data) for item in col]

    # 转换为 pandas Series
    # data = pd.Series(flattened_data)
    data = pd.Series(data)

    USL = 1  # 上规格限
    LSL = 0# 下规格限
    num_groups = 25
     # 分成 2 组

    process_capability = ProcessCapability(data, num_groups=num_groups)
    # cpk_value_r_d2 = process_capability.calculate_cpk(USL, LSL, method='r/d2')
    # print(f'Cpk (R/d2): {cpk_value_r_d2}')

    # cpk_value_s_c4 = process_capability.calculate_cpk(USL, LSL, method='s/c4')
    # print(f'Cpk (S/c4): {cpk_value_s_c4}')

    #   # 绘制 P 图
    # a=process_capability.plot_p_chart()

    # # 绘制 NP 图
    # b=process_capability.plot_np_chart()

    # # 绘制 U 图
    # c=process_capability.plot_u_chart()
    # # print(c)
    # # 绘制 C 图
    # d=process_capability.plot_c_chart()

    #  # 绘制 X-R 图
    # e=process_capability.plot_xr_chart()

    # 绘制 X-S 图
    # f=process_capability.plot_xs_chart()

    # 绘制 I-MR 图
    # g=process_capability.plot_imr_chart()
    z = process_capability.plot_standard_chart()

    h = process_capability.plot_capability_chart(USL, LSL)

    # 绘制移动平均控制图
    i = process_capability.plot_moving_average_chart()

    # 绘制EWMA控制图
    j = process_capability.plot_ewma_chart(r=0.2)

    k = process_capability.plot_cv_control_chart()

    l = process_capability.plot_cusum_chart()
    print(l)
    # m = process_capability.plot_count_cusum_chart(P0=0.2,P1=0.4)

    n = process_capability.plot_chain_chart()
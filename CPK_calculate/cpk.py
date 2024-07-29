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
        self.std_dev = np.std(data, ddof=1)

    def calculate_cpk(self, USL, LSL, method='s/c4', num_groups=1):
        """
        计算 Cpk 值
        :param USL: float，上规格限
        :param LSL: float，下规格限
        :param method: str，方差计算方法，'r/d2' 或 's/c4' 或 'std'
        :param num_groups: int，将数据样本分成的组数（默认为 1）
        :return: float，Cpk 值
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
             # 计算平均标准差
            avg_std_dev = np.mean(group_std_devs)
        elif method == 's/c4':
            c4 = (4 * group_size - 4) / (4 * group_size - 3)  # c4 修正因子
            group_std_devs = [g.std(ddof=1) / c4 for g in grouped_data]
             # 计算平均标准差
            avg_std_dev = np.mean(group_std_devs)
        elif method == 'std':
            avg_std_dev = np.std(data, ddof=1)
        else:
            raise ValueError("method 必须是 'r/d2' 或 's/c4'")

       
        self.std_dev = avg_std_dev
        self.USL = USL
        self.LSL = LSL
        cpk = min((USL - self.mean) / (3 * avg_std_dev), (self.mean - LSL) / (3 * avg_std_dev))
        return cpk

    def plot(self):
        """
        绘制概率图和概率密度曲线图
        :param USL: float，上规格限
        :param LSL: float，下规格限
        """
        mean = self.mean
        std_dev = self.std_dev
        USL = self.USL
        LSL = self.LSL
        Us_Ls_Center = (USL + LSL) / 2
        
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文标签乱码
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建概率密度曲线的 x 值范围
        x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 100)
        # 计算概率密度函数的值
        pdf = norm.pdf(x, loc=mean, scale=std_dev)
        
        plt.figure(figsize=(6, 6))
        plt.subplot(2, 1, 1)
        # 绘制概率图（probability plot）
        stats.probplot(self.data, plot=plt, dist='norm', fit=True, rvalue=True)
        plt.title('Probability Plot (Q-Q Plot)')
        
        plt.subplot(2, 1, 2)
        # 绘制直方图
        plt.hist(self.data, bins=10, density=True, alpha=0.6, color='b', label='Generated Data')
        # 绘制概率密度曲线图
        plt.plot(x, pdf, color='r', label='概率密度曲线')
        plt.xlabel('观测值')
        plt.ylabel('概率密度')
        plt.title('概率密度曲线图')
        
        # 添加规范上限、规范下限、规格中心线和3倍标准差线
        plt.axvline(x=USL, color='g', linestyle='--', label='USL规范上限')
        plt.axvline(x=LSL, color='g', linestyle='--', label='LSL规范下限')
        plt.axvline(x=Us_Ls_Center, color='r', linestyle='--', label='规范中心')
        plt.axvline(x=mean, color='orange', linestyle='--', label='过程中心线')
        plt.axvline(x=mean + 3 * std_dev, color='orange', linestyle='-', label='3sigma')
        plt.axvline(x=mean - 3 * std_dev, color='orange', linestyle='-', label='3sigma')
        
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # 显示图形
        plt.show()

    def calculate_acceptance_rate(self, USL, LSL):
        """
        计算规范上限和规范下限之间的合格率
        :param USL: float，上规格限
        :param LSL: float，下规格限
        :return: float，合格率
        """
        def pdf_function(x):
            return norm.pdf(x, loc=self.mean, scale=self.std_dev)
        
        acceptance_rate = self._calculate_acceptance_rate(pdf_function, LSL, USL)
        return acceptance_rate

    def _calculate_acceptance_rate(self, pdf_func, LSL, USL):
        area, _ = quad(pdf_func, LSL, USL)
        return area

if __name__ == "__main__":
# 示例数据
    data = pd.Series([1.1, 1.3, 1.2, 1.4, 1.5, 1.2, 1.3, 1.1, 1.2, 1.4])
    USL = 1.6  # 上规格限
    LSL = 1.0  # 下规格限
    num_groups = 2  # 分成 2 组

    process_capability = ProcessCapability(data)
    cpk_value_r_d2 = process_capability.calculate_cpk(USL, LSL, method='r/d2', num_groups=num_groups)
    print(f'Cpk (R/d2): {cpk_value_r_d2}')

    cpk_value_s_c4 = process_capability.calculate_cpk(USL, LSL, method='s/c4', num_groups=num_groups)
    print(f'Cpk (S/c4): {cpk_value_s_c4}')

    # 绘制图形
    process_capability.plot()

    # 计算合格率
    acceptance_rate = process_capability.calculate_acceptance_rate(USL, LSL)
    print(f'合格率: {acceptance_rate}')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
class LaneyControlChart:
    def __init__(self, df, chart_type='u'):
        self.df = df
        self.chart_type = chart_type
        self.df['defects'] = self.df.iloc[:, 0]
        self.df['total'] = self.df.iloc[:, 1]

        if chart_type == 'p':
            self.df['values'] = self.df['defects'] / self.df['total']
            self.mean_value = np.mean(self.df['values'])
            self.sigma_z = self.calculate_sigma_z_p()
        elif chart_type == 'u':
            self.df['values'] = self.df['defects'] / self.df['total']
            self.mean_value = np.mean(self.df['values'])
            self.sigma_z = self.calculate_sigma_z_u()
        else:
            raise ValueError("chart_type must be either 'p' or 'u'")

    def calculate_sigma_z_p(self):
        z = (self.df['values'] - self.mean_value) / np.sqrt((self.mean_value * (1 - self.mean_value)) / self.df['total'])
        mr = np.abs(z.diff().dropna())
        mr_bar = np.mean(mr)
        sigma_z = mr_bar / 1.128  # 1.128 是移动极差分布的常数
        return sigma_z

    def calculate_sigma_z_u(self):
        z = (self.df['values'] - self.mean_value) / np.sqrt(self.mean_value / self.df['total'])
        mr = np.abs(z.diff().dropna())
        mr_bar = np.mean(mr)
        sigma_z = mr_bar / 1.128  # 1.128 是移动极差分布的常数
        return sigma_z

    def calculate_control_limits(self):
        if self.chart_type == 'p':
            UCL = self.mean_value + 3 * self.sigma_z * np.sqrt((self.mean_value * (1 - self.mean_value)) / self.df['total'])
            LCL = self.mean_value - 3 * self.sigma_z * np.sqrt((self.mean_value * (1 - self.mean_value)) / self.df['total'])
        elif self.chart_type == 'u':
            UCL = self.mean_value + 3 * self.sigma_z * np.sqrt(self.mean_value / self.df['total'])
            LCL = self.mean_value - 3 * self.sigma_z * np.sqrt(self.mean_value / self.df['total'])

        LCL = np.clip(LCL, 0, None)
        return UCL, LCL

    # def calculate_variance_ratio(self):
    #     if self.chart_type == 'p':
    #         expected_variance = self.mean_value * (1 - self.mean_value) / self.df['total']
    #     elif self.chart_type == 'u':
    #         expected_variance = self.mean_value / self.df['total']

    #     observed_variance = np.var(self.df['values'])
    #     variance_ratio = observed_variance / expected_variance
    #     return variance_ratio

    def calculate_confidence_interval(self):
        if self.chart_type == 'p':

            m = len(self.df)
            u_bar = self.mean_value
            
            # 计算置信区间的上限和下限
            ci_upper = np.exp(0.185 + 5.62 / m + 0.274 / (self.df['defects'].mean()))*100
            ci_lower = 60 # Minitab 使用保守的固定值 60%
        else:
            m = len(self.df)
            u_bar = self.mean_value
            
            # 计算置信区间的上限和下限
            ci_upper = np.exp(0.182 + 5.75 / m + 0.195 / (self.df['defects'].mean()*self.df['total'].mean()))*100
            ci_lower = 60  # Minitab 使用保守的固定值 60%

        return ci_lower, ci_upper

    def plot_chart(self):

        if self.chart_type == 'p':

            total = self.df['total'].mean()
            ai = self.df['values']*total
            x = np.arcsin((np.sqrt((ai+0.375)/(total+0.75)).sort_values()).reset_index(drop = True))
            # x = x.iloc[0]
            z = x.rank(method='first')
                # 计算 Z 分数
            y = stats.norm.ppf((x.rank(method='first')-(3/8)) /(len(x)+1-3/4))
            x = x.to_numpy()
            y = y.reshape(-1, 1)
            slope, intercept = np.polyfit(x, y, 1)
            # model = LinearRegression()
 
            # # 训练模型
            # model.fit(x, y)
            ratio = (1/slope)/(1/np.sqrt(total*4))
            ratio =  ratio.item()*100
            ci_lower, ci_upper = self.calculate_confidence_interval()
            # 预测回归线

        
        else:

            total = self.df['total'].mean()
            ai = self.df['values']*total
            x = ((np.sqrt(ai)+np.sqrt(ai + 1)).sort_values()).reset_index(drop = True)
            # x = x.iloc[0]

                # 计算 Z 分数
            y = stats.norm.ppf((x.rank(method='first')-(3/8)) /(len(x)+1-3/4))
            x = x.to_numpy()
            y = y.reshape(-1, 1)
            slope, intercept = np.polyfit(x, y, 1)
            # model = LinearRegression()
 
            # # 训练模型
            # model.fit(x, y)
            ratio = 1/slope
            ratio =  ratio.item()*100
            ci_lower, ci_upper = self.calculate_confidence_interval()
        
        # 绘制散点图
        plt.scatter(x, y, label='Data Points', color='blue')

        # 绘制线性回归线
        plt.plot(x, slope * x + intercept, color='red', label=f'Linear ')
        if ratio > ci_upper:
            plt.text(0.5, 0.9, 'High: Use Laney', transform=plt.gca().transAxes, fontsize=12, color='red')
        elif ratio < ci_lower:
            plt.text(0.5, 0.9, 'Low: Use Laney', transform=plt.gca().transAxes, fontsize=12, color='green')
        else:
            plt.text(0.5, 0.9, 'anyway', transform=plt.gca().transAxes, fontsize=12, color='blue')
        textstr = f'CI Lower: {ci_lower:.2f}\nCI Upper: {ci_upper:.2f}\nRatio: {ratio:.2f}'
        plt.text(0.02, 0.02, textstr, transform=plt.gca().transAxes, fontsize=12, color='black',
         verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
        # 添加图表标题和标签
        plt.title('Scatter Plot with Linear Regression')
        plt.xlabel('Sorted Data')
        plt.ylabel('Z-Score')
        plt.legend()

        # 显示图表
        plt.show()



        UCL, LCL = self.calculate_control_limits()
        # variance_ratio = self.calculate_variance_ratio()

        plt.figure(figsize=(12, 6))
        plt.plot(self.df['values'], marker='o', linestyle='-', color='b', label=self.chart_type)
        plt.step(range(len(UCL)), UCL , where='mid', linestyle='--', color='r', label='UCL')
        plt.step(range(len(LCL)), LCL , where='mid', linestyle='--', color='r', label='LCL')
        plt.axhline(self.mean_value, color='g', linestyle='--', label=f'Center Line ({self.chart_type}̄)')
        plt.title(f'{self.chart_type.upper()} Control Chart Diagnosis')
        plt.xlabel('Sample')
        plt.ylabel('Proportion Defective' if self.chart_type == 'p' else 'Defects per Unit')
        plt.legend()
        plt.show()
if __name__ == 'main':

        # 示例数据
        data_str = """
        71	5750
        15	9010
        84	7179
        56	6830
        18	7134
        69	8478
        12	8858
        20	7412
        39	7537
        93	8957
        62	8330
        33	9810
        3	8645
        79	5716
        12	9240
        28	7243
        91	8846
        42	6215
        31	4718
        13	4993
        68	9356
        15	7654
        147	4535
        67	5659
        52	5593
        94	9550
        82	7589
        91	8520
        7	9606
        58	6808
        61	8876
        16	5355

        """

        # 解析数据
        data_lines = data_str.strip().split('\n')
        defects = []
        total = []

        for line in data_lines:
            defect, total_val = map(int, line.split())
            defects.append(defect)
            total.append(total_val)

        data = {
            'defects': defects,
            'total': total
        }
        df = pd.DataFrame(data)

        # 创建并绘制Laney P'控制图
        laney_chart_p = LaneyControlChart(df, chart_type='p')
        calculations_p = laney_chart_p.plot_chart()

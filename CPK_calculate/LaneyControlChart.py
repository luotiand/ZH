import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LaneyControlChart:
    def __init__(self, df, chart_type='p'):
        self.df = df
        self.chart_type = chart_type

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
        n = len(self.df)
        z = (self.df['values'] - self.mean_value) / np.sqrt((self.mean_value * (1 - self.mean_value)) / self.df['total'])
        z_bar = np.mean(z)
        sigma_z = np.sqrt(np.sum((z - z_bar) ** 2) / (n - 1))
        return sigma_z

    def calculate_sigma_z_u(self):
        n = len(self.df)
        z = (self.df['values'] - self.mean_value) / np.sqrt(self.mean_value / self.df['total'])
        z_bar = np.mean(z)
        sigma_z = np.sqrt(np.sum((z - z_bar) ** 2) / (n - 1))
        return sigma_z

    def calculate_control_limits(self):
        if self.chart_type == 'p':
            z = (self.df['values'] - self.mean_value) / np.sqrt((self.mean_value * (1 - self.mean_value)) / self.df['total'])
        elif self.chart_type == 'u':
            z = (self.df['values'] - self.mean_value) / np.sqrt(self.mean_value / self.df['total'])

        z_bar = np.mean(z)
        UCL_z = z_bar + 3 * self.sigma_z
        LCL_z = z_bar - 3 * self.sigma_z

        if self.chart_type == 'p':
            UCL = self.mean_value + UCL_z * np.sqrt((self.mean_value * (1 - self.mean_value)) / self.df['total'])
            LCL = self.mean_value + LCL_z * np.sqrt((self.mean_value * (1 - self.mean_value)) / self.df['total'])
        elif self.chart_type == 'u':
            UCL = self.mean_value + UCL_z * np.sqrt(self.mean_value / self.df['total'])
            LCL = self.mean_value + LCL_z * np.sqrt(self.mean_value / self.df['total'])

        LCL = np.clip(LCL, 0, None)
        return UCL, LCL

    def plot_chart(self):
        UCL, LCL = self.calculate_control_limits()

        plt.figure(figsize=(10, 6))
        plt.plot(self.df['values'], marker='o', linestyle='-', color='b', label=self.chart_type)
        plt.plot(UCL, linestyle='--', color='r', label='UCL')
        plt.plot(LCL, linestyle='--', color='r', label='LCL')
        plt.axhline(self.mean_value, color='g', linestyle='--', label=f'Center Line ({self.chart_type}̄)')
        
        plt.title(f'Laney {self.chart_type.upper()}\' Control Chart')
        plt.xlabel('Sample')
        plt.ylabel('Proportion Defective' if self.chart_type == 'p' else 'Defects per Unit')
        plt.legend()
        plt.show()

# 示例数据
data = {
    'defects': [16, 11, 18, 6, 11, 15, 14, 12, 15, 16, 17, 19, 14, 17, 5, 11, 5, 15, 17, 7, 18, 9, 20, 7, 9, 36, 7, 20, 9, 9],
    'total': [870, 800, 921, 638, 750, 543, 625, 835, 938, 867, 787, 646, 753, 635, 958, 694, 712, 683, 953, 840, 914, 685, 661, 886, 606, 788, 751, 892, 947, 927]
}

df = pd.DataFrame(data)

# 创建并绘制Laney P'控制图
laney_chart_p = LaneyControlChart(df, chart_type='p')
laney_chart_p.plot_chart()

# 创建并绘制Laney U'控制图
laney_chart_u = LaneyControlChart(df, chart_type='u')
laney_chart_u.plot_chart()

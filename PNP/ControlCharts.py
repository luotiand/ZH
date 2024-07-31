import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ChartRules import ChartRules

class ControlCharts:
    def __init__(self, df):
        # 初始化时接收数据框（DataFrame）并设置列名
        self.df = df
        self.df.columns = ['Defects', 'Total']
        # 计算每个样本的缺陷比例 p 和缺陷数 u
        self.df['p'] = self.df.iloc[:, 0] / self.df.iloc[:, 1]
        self.df['u'] = self.df.iloc[:, 0] / self.df.iloc[:, 1]
        self.df['Defects'] = self.df.iloc[:, 0]
    def plot_p_chart(self):
        # 计算每组的p图的中心线（p̄）和控制限（UCL 和 LCL）
        p_bar = self.df['p'].mean()  # 计算p的均值

        # 计算每组的UCL和LCL
        self.df['UCL'] = self.df.apply(lambda row: p_bar + 3 * np.sqrt((p_bar * (1 - p_bar)) / row['Total']), axis=1)
        self.df['LCL'] = self.df.apply(lambda row: p_bar - 3 * np.sqrt((p_bar * (1 - p_bar)) / row['Total']), axis=1)
        self.df['LCL'] = self.df['LCL'].clip(lower=0)  # 确保LCL不小于0
        chartRules = ChartRules(self.df['p'],self.df['UCL'],self.df['LCL'])
        points_outside_control_limits = chartRules.points_outside_control_limits()
        six_consecutive_points = chartRules.six_consecutive_points_trending()
        fourteen_points_alternating = chartRules.fourteen_points_alternating()
        fifteen_points_in_zone_c = chartRules.fifteen_points_in_zone_c()
        eight_points_outside_zone_c = chartRules.eight_points_outside_zone_c()
        three_points_in_zone_a = chartRules.three_points_in_zone_a()
        four_of_five_points_in_zone_b_or_beyond = chartRules.four_of_five_points_in_zone_b_or_beyond()
        nine_points_on_one_side_of_center_line = chartRules.nine_points_on_one_side_of_center_line()
        anomalies = points_outside_control_limits

        calculations = {
            'p_bar': p_bar,
            'UCL': self.df['UCL'].mean(),
            'LCL': self.df['LCL'].mean(),
        }

        # 绘制p图
        plt.figure(figsize=(10, 6))
        plt.plot(self.df.index, self.df['p'], marker='o', linestyle='-', color='b', label='p')
        plt.step(self.df.index, self.df['UCL'] , where='pre', linestyle='--', color='r', label='UCL')
        plt.step(self.df.index, self.df['LCL'] , where='pre', linestyle='--', color='r', label='LCL')
        # plt.plot(self.df.index, self.df['UCL'], linestyle='--', color='r', label='UCL')
        # plt.plot(self.df.index, self.df['LCL'], linestyle='--', color='r', label='LCL')
        plt.axhline(p_bar, color='g', linestyle='--', label='Center Line (p̄)')
        # 标记异常点
        if not anomalies.empty:
            plt.scatter(anomalies.index, anomalies['p'], color='purple', edgecolor='black', zorder=5, label='Anomalies')
        plt.title('P Chart')
        plt.xlabel('Sample')
        plt.ylabel('Proportion Defective')
        plt.legend()
        plt.show()
        return calculations

    def plot_np_chart(self):
        # 计算每组的np图的中心线（np̄）和控制限（UCL 和 LCL）
        np_bar = self.df['Defects'].mean()
        p_bar = self.df['p'].mean()
        n = self.df.iloc[:, 2]
        # 计算每组的UCL和LCL
        UCL =  np_bar + 3 * np.sqrt(np_bar*(1-p_bar))
        LCL =  np_bar - 3 * np.sqrt(np_bar*(1-p_bar))
        
        calculations = {
            'np_bar': np_bar,
            'UCL': self.df['UCL'].mean(),
            'LCL': self.df['LCL'].mean(),
        }

        # 绘制np图
        plt.figure(figsize=(10, 6))
        plt.plot(self.df.index, self.df['Defects'], marker='o', linestyle='-', color='b', label='Number of Defects')

        plt.axhline(UCL, color='r', linestyle='--', label='UCL')
        plt.axhline(LCL, color='r', linestyle='--', label='LCL')
        plt.axhline(np_bar, color='g', linestyle='--', label='Center Line (np̄)')
        plt.title('NP Chart')
        plt.xlabel('Sample')
        plt.ylabel('Number of Defects')
        plt.legend()
        plt.show()
        return calculations

    def plot_u_chart(self):
        # 计算每组的u图的中心线（ū）和控制限（UCL 和 LCL）
        u_bar = self.df['u'].mean()
        
        # 计算每组的UCL和LCL
        self.df['UCL'] = self.df.apply(lambda row: u_bar + 3 * np.sqrt(u_bar / row['Total']), axis=1)
        self.df['LCL'] = self.df.apply(lambda row: u_bar - 3 * np.sqrt(u_bar / row['Total']), axis=1)
        self.df['LCL'] = self.df['LCL'].clip(lower=0)  # 确保LCL不小于0

        calculations = {
            'u_bar': u_bar,
            'UCL': self.df['UCL'].mean(),
            'LCL': self.df['LCL'].mean(),
        }

        # 绘制u图
        plt.figure(figsize=(10, 6))
        plt.plot(self.df.index, self.df['u'], marker='o', linestyle='-', color='b', label='u')
        plt.step(self.df.index, self.df['UCL'] , where='pre', linestyle='--', color='r', label='UCL')
        plt.step(self.df.index, self.df['LCL'] , where='pre', linestyle='--', color='r', label='LCL')
        # plt.plot(self.df.index, self.df['UCL'], linestyle='--', color='r', label='UCL')
        # plt.plot(self.df.index, self.df['LCL'], linestyle='--', color='r', label='LCL')
        plt.axhline(u_bar, color='g', linestyle='--', label='Center Line (ū)')
        plt.title('U Chart')
        plt.xlabel('Sample')
        plt.ylabel('Defects per Unit')
        plt.legend()
        plt.show()
        return calculations

    def plot_c_chart(self):
        # 计算每组的c图的中心线（c̄）和控制限（UCL 和 LCL）
        c_bar = self.df['Defects'].mean()
        
        # 计算每组的UCL和LCL
        self.df['UCL'] = self.df.apply(lambda row: c_bar + 3 * np.sqrt(c_bar), axis=1)
        self.df['LCL'] = self.df.apply(lambda row: c_bar - 3 * np.sqrt(c_bar), axis=1)
        self.df['LCL'] = self.df['LCL'].clip(lower=0)  # 确保LCL不小于0

        calculations = {
            'c_bar': c_bar,
            'UCL': self.df['UCL'].mean(),
            'LCL': self.df['LCL'].mean(),
        }

        # 绘制c图
        plt.figure(figsize=(10, 6))
        plt.plot(self.df.index, self.df['Defects'], marker='o', linestyle='-', color='b', label='c')
        plt.plot(self.df.index, self.df['UCL'], linestyle='--', color='r', label='UCL')
        plt.plot(self.df.index, self.df['LCL'], linestyle='--', color='r', label='LCL')
        plt.axhline(c_bar, color='g', linestyle='--', label='Center Line (c̄)')
        plt.title('C Chart')
        plt.xlabel('Sample')
        plt.ylabel('Number of Defects')
        plt.legend()
        plt.show()
        return calculations

def main():
    # 示例数据
    data = {
        'defects': [16, 11, 18, 6, 11, 15, 14, 12, 15, 16, 17, 19, 14, 17, 5, 11, 5, 15, 17, 7, 18, 9, 20, 7, 9, 36, 7, 20, 9, 9],
        'total': [870, 800, 921, 638, 750, 543, 625, 835, 938, 867, 787, 646, 753, 635, 958, 694, 712, 683, 953, 840, 914, 685, 661, 886, 606, 788, 751, 892, 947, 927]
    }

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 创建 ControlCharts 类实例
    control_charts = ControlCharts(df)

    # 绘制P图
    p_chart_calculations = control_charts.plot_p_chart()
    print("P Chart Calculations:", p_chart_calculations)

    # 绘制NP图
    np_chart_calculations = control_charts.plot_np_chart()
    print("NP Chart Calculations:", np_chart_calculations)

    # 绘制U图
    u_chart_calculations = control_charts.plot_u_chart()
    print("U Chart Calculations:", u_chart_calculations)

    # 绘制C图
    c_chart_calculations = control_charts.plot_c_chart()
    print("C Chart Calculations:", c_chart_calculations)

if __name__ == "__main__":
    main()

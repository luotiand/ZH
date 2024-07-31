import numpy as np
import pandas as pd

class ChartRules:
    def __init__(self, p, UCL, LCL):
        # 如果UCL不是列表，则将其转换为与p长度相同的列表
        if isinstance(UCL, (int, float)):
            UCL = [UCL] * len(p)
        
        # 如果LCL不是列表，则将其转换为与p长度相同的列表
        if isinstance(LCL, (int, float)):
            LCL = [LCL] * len(p)
        
        self.df = pd.DataFrame({'p': p, 'UCL': UCL, 'LCL': LCL})
        self.p_bar = np.mean(p)
        self.df['p_bar'] = [self.p_bar] * len(p)

    def points_outside_control_limits(self):
        # 方法1: 超过UCL和LCL的点
        print(self.df.head())  
        return self.df[(self.df['p'] > self.df['UCL']) | (self.df['p'] < self.df['LCL'])]

    def six_consecutive_points_trending(self):
        # 方法2: 6连升/降
        trends = self.df['p'].diff().apply(np.sign)
        indices = [i for i in range(5, len(trends)) if all(trends[i-5:i+1] == 1) or all(trends[i-5:i+1] == -1)]
        return self.df.iloc[indices]

    def fourteen_points_alternating(self):
        # 方法3: 连续14个点在中线上下交替
        alternating = ((self.df['p'] - self.df['p_bar']).diff().apply(np.sign).diff().abs() == 2).rolling(window=14).sum() == 14
        result = self.df[alternating].index.to_series().apply(lambda idx: self.df.iloc[idx-13:idx+1] if idx >= 13 else pd.DataFrame())
        return result


    def fifteen_points_in_zone_c(self):
        # 方法4: 连续15个点都是C区
        zone_c_upper = self.p_bar + (self.df['UCL'] - self.p_bar) / 3
        zone_c_lower = self.p_bar - (self.p_bar - self.df['LCL']) / 3
        zone_c = (self.df['p'] < zone_c_upper) & (self.df['p'] > zone_c_lower)
        indices = [i for i in range(14, len(zone_c)) if all(zone_c[i-14:i+1])]
        result = pd.DataFrame()
        for start in indices:
            if start + 14 <= len(self.df):  # 确保索引不会超出范围
                result = pd.concat([result, self.df.iloc[start-14:start+1]])

        return result

    def eight_points_outside_zone_c(self):
        # 方法5: 连续8个点不在C区
        zone_c_upper = self.p_bar + (self.df['UCL'] - self.p_bar) / 3
        zone_c_lower = self.p_bar - (self.p_bar - self.df['LCL']) / 3
        outside_zone_c = (self.df['p'] >= zone_c_upper) | (self.df['p'] <= zone_c_lower)
        indices = [i for i in range(7, len(outside_zone_c)) if all(outside_zone_c[i-7:i+1])]
        result = pd.DataFrame()
        for start in indices:
            if start + 7 <= len(self.df):  # 确保索引不会超出范围
                result = pd.concat([result, self.df.iloc[start-7:start+1]])
        return result

    def three_points_in_zone_a(self):
        # 方法6: 连续3个点有2个点在A区
        zone_a_upper = self.p_bar + 2 * (self.df['UCL'] - self.p_bar) / 3
        zone_a_lower = self.p_bar - 2 * (self.p_bar - self.df['LCL']) / 3
        zone_a = (self.df['p'] > zone_a_upper) | (self.df['p'] < zone_a_lower)
        indices = [i for i in range(2, len(zone_a)) if sum(zone_a[i-2:i+1]) >= 2]
        result = pd.DataFrame()
        for start in indices:
            if start + 2 <= len(self.df):  # 确保索引不会超出范围
                result = pd.concat([result, self.df.iloc[start-2:start+1]])

        return result

    def four_of_five_points_in_zone_b_or_beyond(self):
        # 方法7: 连续5个点有4个不在C区
        zone_c_upper = self.p_bar + (self.df['UCL'] - self.p_bar) / 3
        zone_c_lower = self.p_bar - (self.p_bar - self.df['LCL']) / 3
        not_in_zone_c = (self.df['p'] >= zone_c_upper) | (self.df['p'] <= zone_c_lower)
        result = pd.DataFrame()
        indices = [i for i in range(4, len(not_in_zone_c)) if sum(not_in_zone_c[i-4:i+1]) >= 4]
        for start in indices:
            if start + 4 <= len(self.df):  # 确保索引不会超出范围
                result = pd.concat([result, self.df.iloc[start-4:start+1]])

        return result

    def nine_points_on_one_side_of_center_line(self):
        # 方法8: 连续9个点在中线同一侧
        above_center = self.df['p'] > self.p_bar
        below_center = self.df['p'] < self.p_bar
        result = pd.DataFrame()
        indices = [i for i in range(8, len(above_center)) if all(above_center[i-8:i+1]) or all(below_center[i-8:i+1])]
        for start in indices:
            if start + 8 <= len(self.df):  # 确保索引不会超出范围
                result = pd.concat([result, self.df.iloc[start-8:start+1]])
        return result

# # 示例数据
# p = [0.01, 0.02, 0.01, 0.00, 0.03, 0.02, 0.01, 0.00, 0.01, 0.02, 0.01, 0.03, 0.02, 0.01, 0.00]
# UCL = 0.03
# LCL = 0.00

# control_rules = ChartRules(p, UCL, LCL)

# # 调用异常检测方法
# print("Points outside control limits:", control_rules.points_outside_control_limits())
# print("Six consecutive points trending:", control_rules.six_consecutive_points_trending())
# print("Fourteen points alternating:", control_rules.fourteen_points_alternating())
# print("Fifteen points in zone C:", control_rules.fifteen_points_in_zone_c())
# print("Eight points outside zone C:", control_rules.eight_points_outside_zone_c())
# print("Three points in zone A:", control_rules.three_points_in_zone_a())
# print("Four of five points in zone B or beyond:", control_rules.four_of_five_points_in_zone_b_or_beyond())
# print("Nine points on one side of center line:", control_rules.nine_points_on_one_side_of_center_line())

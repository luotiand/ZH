import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportions_chisquare

class HypothesisTesting:
    def __init__(self, dataframe):
        """
        初始化类
        :param dataframe: pandas DataFrame，待处理的数据框
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("输入数据必须是 pandas DataFrame")
        self.dataframe = dataframe

    
    def one_sample_mean(self, column, popmean, sigma=None):
        """
        单样本均值检验
        :param column: str，待检验的列名
        :param popmean: float，假设总体均值
        :param sigma: float，总体方差已知（默认为 None，表示使用 t 检验）
        :return: t-statistic/z-statistic, p-value
        """
        data = self.dataframe[column].dropna()
        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        
        if sigma is None:
            # 使用 t 检验
            t_stat, p_val = stats.ttest_1samp(data, popmean)
            return t_stat, p_val
        else:
            # 使用 z 检验
            z_stat = (sample_mean - popmean) / (sigma / np.sqrt(n))
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            return z_stat, p_val

    def two_sample_mean(self, column1, column2, equal_var=True):
        """
        双样本均值检验
        :param column1: str，待检验的第一列名
        :param column2: str，待检验的第二列名
        :param equal_var: bool，是否假设方差相等
        :return: t-statistic, p-value
        """
        data1 = self.dataframe[column1].dropna()
        data2 = self.dataframe[column2].dropna()
        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=equal_var)
        return t_stat, p_val
    def multiple_sample_mean(self, *columns):
        """
        多样本均值检验（单因素方差分析）
        :param columns: str，可变数量的待检验的列名
        :return: F-statistic, p-value
        """
        data = [self.dataframe[column].dropna() for column in columns]
        f_stat, p_val = stats.f_oneway(*data)
        return f_stat, p_val


    def one_sample_variance(self, column, popvar):
        """
        单样本方差检验
        :param column: str，待检验的列名
        :param popvar: float，假设总体方差
        :return: chi2-statistic, p-value
        """
        data = self.dataframe[column].dropna()
        df = len(data) - 1
        sample_var = np.var(data, ddof=1)
        chi2_stat = df * sample_var / popvar
        p_val = stats.chi2.sf(chi2_stat, df)
        return chi2_stat, p_val

    def two_sample_variance(self, column1, column2):
        """
        双样本方差检验
        :param column1: str，待检验的第一列名
        :param column2: str，待检验的第二列名
        :return: F-statistic, p-value
        """
        data1 = self.dataframe[column1].dropna()
        data2 = self.dataframe[column2].dropna()
        f_stat, p_val = stats.levene(data1, data2)
        return f_stat, p_val
    
    def multiple_sample_variance(self, *columns):
        """
        多样本方差检验
        :param columns: str，可变数量的待检验的列名
        :return: F-statistic, p-value
        """
        data = [self.dataframe[column].dropna() for column in columns]
        f_stat, p_val = stats.bartlett(*data)
        return f_stat, p_val


    def one_sample_proportion(self, count_column, nobs_column, prop):
        """
        单样本比例检验
        :param count_column: str，成功次数的列名
        :param nobs_column: str，总样本量的列名
        :param prop: float，假设比例
        :return: z-statistic, p-value
        """
        count = self.dataframe[count_column].sum()
        nobs = self.dataframe[nobs_column].sum()
        stat, p_val = proportions_ztest(count, nobs, prop)
        return stat, p_val

    def two_sample_proportion(self, count1_column, nobs1_column, count2_column, nobs2_column,prop):
        """
        双样本比例检验
        :param count1_column: str，样本1中成功次数的列名
        :param nobs1_column: str，样本1的样本量的列名
        :param count2_column: str，样本2中成功次数的列名
        :param nobs2_column: str，样本2的样本量的列名
        :return: z-statistic, p-value
        """
        count1 = self.dataframe[count1_column].sum()
        nobs1 = self.dataframe[nobs1_column].sum()
        count2 = self.dataframe[count2_column].sum()
        nobs2 = self.dataframe[nobs2_column].sum()
        stat, p_val = proportions_ztest([count1, count2], [nobs1, nobs2],prop)
        return stat, p_val

    def multiple_sample_proportion(self, count_columns, nobs_columns,prop):
        """
        多样本比例检验（列联表）
        :param count_columns: list，包含每个样本成功次数的列名列表
        :param nobs_columns: list，包含每个样本总样本量的列名列表
        :return: chi2-statistic, p-value
        """
        counts = [self.dataframe[col].sum() for col in count_columns]
        nobs = [self.dataframe[col].sum() for col in nobs_columns]
        stat, p_val, _ = proportions_chisquare(counts, nobs,prop)
        return stat, p_val


# 示例用法:
if __name__ == "__main__":
    # 创建示例 DataFrame
    df = pd.DataFrame({
        'sample1': np.random.normal(0, 1, 100),
        'sample2': np.random.normal(0.5, 1, 100),
        'sample3': np.random.normal(1, 1, 100),
        'success_count': np.random.randint(0, 100, 100),
        'total_count': np.random.randint(100, 200, 100)
    })

    ht = HypothesisTesting(df)
    
    # 单样本均值检验
    t_stat, p_val = ht.one_sample_mean('sample1', 0)
    print("单样本均值检验:", t_stat, p_val)

    # 双样本均值检验
    t_stat, p_val = ht.two_sample_mean('sample1', 'sample2')
    print("双样本均值检验:", t_stat, p_val)

    # 单样本方差检验
    chi2_stat, p_val = ht.one_sample_variance('sample1', 1)
    print("单样本方差检验:", chi2_stat, p_val)

    # 双样本方差检验
    f_stat, p_val = ht.two_sample_variance('sample1', 'sample2')
    print("双样本方差检验:", f_stat, p_val)

    # 单样本比例检验
    stat, p_val = ht.one_sample_proportion('success_count', 'total_count', 0.3)
    print("单样本比例检验:", stat, p_val)

    #双样本比例检验
    stat, p_val = ht.two_sample_proportion('success_count', 'total_count', 'success_count', 'total_count', 0.3)
    print("双样本比例检验:", stat, p_val)

    # 多样本比例检验（列联表）
    stat, p_val = ht.multiple_sample_proportion(['success_count', 'success_count'], ['total_count', 'total_count'], 0.3)
    print("多样本比例检验:", stat, p_val)

    # 多样本方差检验
    f_stat, p_val = ht.multiple_sample_variance('sample1', 'sample2', 'sample3')
    print("多样本方差检验:", f_stat, p_val)

    # 多样本均值检验
    f_stat, p_val = ht.multiple_sample_mean('sample1', 'sample2', 'sample3')
    print("多样本均值检验:", f_stat, p_val)

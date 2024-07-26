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

    
    def one_sample_mean(self, column, popmean, sigma=None, alternative='two-sided', alpha=0.05):
        """
        单样本均值检验
        :param column: str，待检验的列名
        :param popmean: float，假设总体均值
        :param sigma: float，总体方差已知（默认为 None，表示使用 t 检验）
        :param alternative: str，假设检验类型（'two-sided'、'less' 或 'greater'）
        :param alpha: float，显著性水平（默认为 0.05）
        :return: t-statistic/z-statistic, p-value, critical value
        """
        data = self.dataframe[column].dropna()
        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        df = n - 1  # 自由度
        
        if sigma is None:
            # 使用 t 检验
            t_stat, p_val = stats.ttest_1samp(data, popmean)
            if alternative == 'two-sided':
                crit_value = stats.t.ppf(1 - alpha / 2, df)
                p_val = 2 * min(stats.t.cdf(t_stat, df), 1 - stats.t.cdf(t_stat, df))
            elif alternative == 'less':
                crit_value = stats.t.ppf(alpha, df)
                p_val = stats.t.cdf(t_stat, df)
            elif alternative == 'greater':
                crit_value = stats.t.ppf(1 - alpha, df)
                p_val = 1 - stats.t.cdf(t_stat, df)
            else:
                raise ValueError("alternative 必须是 'two-sided'、'less' 或 'greater'")
            return t_stat, p_val, crit_value
        else:
            # 使用 z 检验
            z_stat = (sample_mean - popmean) / (sigma / np.sqrt(n))
            if alternative == 'two-sided':
                crit_value = stats.norm.ppf(1 - alpha / 2)
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            elif alternative == 'less':
                crit_value = stats.norm.ppf(alpha)
                p_val = stats.norm.cdf(z_stat)
            elif alternative == 'greater':
                crit_value = stats.norm.ppf(1 - alpha)
                p_val = 1 - stats.norm.cdf(z_stat)
            else:
                raise ValueError("alternative 必须是 'two-sided'、'less' 或 'greater'")
            return z_stat, p_val, crit_value


    def two_sample_mean(self, column1, column2, sigma1=None, sigma2=None, equal_var=True, alternative='two-sided', alpha=0.05):
        """
        双样本均值检验
        :param column1: str，待检验的第一列名
        :param column2: str，待检验的第二列名
        :param sigma1: float，第一组的已知标准差（可选）
        :param sigma2: float，第二组的已知标准差（可选）
        :param equal_var: bool，是否假设方差相等（默认为True，仅在sigma1和sigma2未提供时有效）
        :param alternative: str，假设检验类型（'two-sided'、'less' 或 'greater'）
        :param alpha: float，显著性水平（默认为 0.05）
        :return: t-statistic, p-value, critical value
        """
        data1 = self.dataframe[column1].dropna()
        data2 = self.dataframe[column2].dropna()

        n1 = len(data1)
        n2 = len(data2)
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)

        if sigma1 is not None and sigma2 is not None:
            # 情况1：σ1和σ2都已知
            se = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
            t_stat = (mean1 - mean2) / se
            df = n1 + n2 - 2  # 自由度
        elif sigma1 is None and sigma2 is None:
            # 情况2和3：σ1和σ2未知
            if equal_var:
                # σ1 = σ2
                pooled_var = ((n1 - 1) * np.var(data1, ddof=1) + (n2 - 1) * np.var(data2, ddof=1)) / (n1 + n2 - 2)
                se = np.sqrt(pooled_var * (1 / n1 + 1 / n2))
                t_stat = (mean1 - mean2) / se
                df = n1 + n2 - 2
            else:
                # σ1 ≠ σ2
                se1 = np.var(data1, ddof=1) / n1
                se2 = np.var(data2, ddof=1) / n2
                se = np.sqrt(se1 + se2)
                t_stat = (mean1 - mean2) / se
                df = (se1 + se2)**2 / ((se1**2 / (n1 - 1)) + (se2**2 / (n2 - 1)))
        else:
            raise ValueError("sigma1 和 sigma2 必须同时提供或同时不提供")

        if alternative == 'two-sided':
            crit_value = stats.t.ppf(1 - alpha / 2, df)
            p_val = 2 * min(stats.t.cdf(t_stat, df), 1 - stats.t.cdf(t_stat, df))
        elif alternative == 'less':
            crit_value = stats.t.ppf(alpha, df)
            p_val = stats.t.cdf(t_stat, df)
        elif alternative == 'greater':
            crit_value = stats.t.ppf(1 - alpha, df)
            p_val = 1 - stats.t.cdf(t_stat, df)
        else:
            raise ValueError("alternative 必须是 'two-sided'、'less' 或 'greater'")
        
        return t_stat, p_val, crit_value
    
    def multiple_sample_mean(self, *columns, alternative='two-sided', alpha=0.05):
        """
        多样本均值检验（单因素方差分析）
        :param columns: str，可变数量的待检验的列名
        :param alternative: str，假设检验类型（'two-sided'、'less' 或 'greater'）
        :param alpha: float，显著性水平（默认为 0.05）
        :return: F-statistic, p-value, critical value
        """
        da = []
        for column in columns:
            da.append(self.dataframe[column].dropna())
        # 取出第一个 DataFrame
        df = da[0]

        # 将 DataFrame 转换为列表
        data = df.values.tolist()
        data = list(map(list, zip(*data)))
        k = len(data)

        f_stat, p_val = stats.f_oneway(*data)
        df_between = len(columns) - 1
        df_within = sum([len(d) for d in data]) - len(columns)

        if alternative == 'two-sided':
            crit_value = stats.f.ppf(1 - alpha / 2, df_between, df_within)
            p_val = 2 * min(stats.f.cdf(f_stat, df_between, df_within), 1 - stats.f.cdf(f_stat, df_between, df_within))
        elif alternative == 'less':
            crit_value = stats.f.ppf(alpha, df_between, df_within)
            p_val = stats.f.cdf(f_stat, df_between, df_within)
        elif alternative == 'greater':
            crit_value = stats.f.ppf(1 - alpha, df_between, df_within)
            p_val = 1 - stats.f.cdf(f_stat, df_between, df_within)
        else:
            raise ValueError("alternative 必须是 'two-sided'、'less' 或 'greater'")
        
        return f_stat, p_val, crit_value



    def one_sample_variance(self, column, popvar, mean_known=None, alternative='two-sided', alpha=0.05):
        """
        单样本方差检验
        :param column: str，待检验的列名
        :param popvar: float，假设总体方差
        :param mean_known: float，已知均值（如果已知，默认None）
        :param alternative: str，假设检验类型（'two-sided'、'less' 或 'greater'）
        :param alpha: float，显著性水平（默认为 0.05）
        :return: chi2-statistic, p-value, critical value
        """
        data = self.dataframe[column].dropna()
        n = len(data)
        
        if mean_known is not None:
            # 均值已知的情况
            sample_var = np.sum((data - mean_known) ** 2) / n
            df = n
        else:
            # 均值未知的情况
            sample_var = np.var(data, ddof=1)
            df = n - 1
        
        chi2_stat = df * sample_var / popvar

        if alternative == 'two-sided':
            crit_value_low = stats.chi2.ppf(alpha / 2, df)
            crit_value_high = stats.chi2.ppf(1 - alpha / 2, df)
            p_val = 2 * min(stats.chi2.cdf(chi2_stat, df), 1 - stats.chi2.cdf(chi2_stat, df))
        elif alternative == 'less':
            crit_value = stats.chi2.ppf(alpha, df)
            p_val = stats.chi2.cdf(chi2_stat, df)
        elif alternative == 'greater':
            crit_value = stats.chi2.ppf(1 - alpha, df)
            p_val = 1 - stats.chi2.cdf(chi2_stat, df)
        else:
            raise ValueError("alternative 必须是 'two-sided'、'less' 或 'greater'")

        if alternative == 'two-sided':
            return chi2_stat, p_val, (crit_value_low, crit_value_high)
        else:
            return chi2_stat, p_val, crit_value

    def two_sample_variance(self, column1, column2, mean1=None, mean2=None, alternative='two-sided', alpha=0.05):
        """
        双样本方差检验
        :param column1: str，待检验的第一列名
        :param column2: str，待检验的第二列名
        :param mean1: float，第一列的已知均值（可选）
        :param mean2: float，第二列的已知均值（可选）
        :param alternative: str，假设检验类型（'two-sided'、'less' 或 'greater'）
        :param alpha: float，显著性水平（默认为 0.05）
        :return: F-statistic, p-value, critical value
        """
        data1 = self.dataframe[column1].dropna()
        data2 = self.dataframe[column2].dropna()
        n1, n2 = len(data1), len(data2)
        
        if mean1 is not None and mean2 is not None:
            # 均值已知的情况
            var1 = np.sum((data1 - mean1) ** 2) / n1
            var2 = np.sum((data2 - mean2) ** 2) / n2
            df1, df2 = n1, n2
        else:
            # 均值未知的情况
            var1 = np.var(data1, ddof=1)
            var2 = np.var(data2, ddof=1)
            df1, df2 = n1 - 1, n2 - 1

        f_stat = var1 / var2

        if alternative == 'two-sided':
            crit_value_low = stats.f.ppf(alpha / 2, df1, df2)
            crit_value_high = stats.f.ppf(1 - alpha / 2, df1, df2)
            p_val = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
        elif alternative == 'less':
            crit_value = stats.f.ppf(alpha, df1, df2)
            p_val = stats.f.cdf(f_stat, df1, df2)
        elif alternative == 'greater':
            crit_value = stats.f.ppf(1 - alpha, df1, df2)
            p_val = 1 - stats.f.cdf(f_stat, df1, df2)
        else:
            raise ValueError("alternative 必须是 'two-sided'、'less' 或 'greater'")

        if alternative == 'two-sided':
            return f_stat, p_val, (crit_value_low, crit_value_high)
        else:
            return f_stat, p_val, crit_value
    
    def multiple_sample_variance(self, *columns, alternative='two-sided', alpha=0.05):
        """
        多样本方差检验
        :param columns: str，可变数量的待检验的列名
        :param alternative: str，假设检验类型（'two-sided'、'less' 或 'greater'）
        :param alpha: float，显著性水平（默认为 0.05）
        :return: F-statistic, p-value, critical value
        """
        da = []
        for column in columns:
            da.append(self.dataframe[column].dropna())
        # 取出第一个 DataFrame
        df = da[0]

        # 将 DataFrame 转换为列表
        data = df.values.tolist()
        k = len(data)

        
        # 计算 Bartlett's test 统计量
        f_stat, p_val = stats.bartlett(*data)
        
        # 计算 F 分布的临界值
        df1 = k - 1  # 自由度1：样本组数减去1
        df2 = sum(len(d) for d in data) - k  # 自由度2：样本总数减去样本组数

        if alternative == 'two-sided':
            crit_value_low = stats.f.ppf(alpha / 2, df1, df2)
            crit_value_high = stats.f.ppf(1 - alpha / 2, df1, df2)
            crit_value = (crit_value_low, crit_value_high)
            # For two-sided, you may need to compare the absolute f_stat to critical values.
            p_val = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
        elif alternative == 'less':
            crit_value = stats.f.ppf(alpha, df1, df2)
            # For less, p-value is directly from the F distribution CDF.
            p_val = stats.f.cdf(f_stat, df1, df2)
        elif alternative == 'greater':
            crit_value = stats.f.ppf(1 - alpha, df1, df2)
            # For greater, p-value is the complement of the F distribution CDF.
            p_val = 1 - stats.f.cdf(f_stat, df1, df2)
        else:
            raise ValueError("alternative 必须是 'two-sided'、'less' 或 'greater'")

        return f_stat, p_val, crit_value



    def one_sample_proportion(self, count_column, nobs_column, prop, alternative='two-sided', alpha=0.05):
        """
        单样本比例检验
        :param count_column: str，成功次数的列名
        :param nobs_column: str，总样本量的列名
        :param prop: float，假设比例
        :param alternative: str，假设检验类型（'two-sided'、'less' 或 'greater'）
        :param alpha: float，显著性水平（默认为 0.05）
        :return: z-statistic, p-value, critical value
        """
        count = self.dataframe[count_column].sum()
        nobs = self.dataframe[nobs_column].sum()
        
        # 计算 z 统计量和 p 值
        stat, p_val = proportions_ztest(count, nobs, prop, alternative=alternative)
        
        # 计算临界值
        if alternative == 'two-sided':
            crit_value_low = stats.norm.ppf(alpha / 2)
            crit_value_high = stats.norm.ppf(1 - alpha / 2)
            crit_value = (crit_value_low, crit_value_high)
            # 对于双尾检验，p-value 是 z 统计量绝对值的两侧面积
            p_val = 2 * (1 - stats.norm.cdf(np.abs(stat)))
        elif alternative == 'less':
            crit_value = stats.norm.ppf(alpha)
            # 对于单尾检验（小于），p-value 是 z 统计量的左侧面积
            p_val = stats.norm.cdf(stat)
        elif alternative == 'greater':
            crit_value = stats.norm.ppf(1 - alpha)
            # 对于单尾检验（大于），p-value 是 z 统计量的右侧面积
            p_val = 1 - stats.norm.cdf(stat)
        else:
            raise ValueError("alternative 必须是 'two-sided'、'less' 或 'greater'")
        
        return stat, p_val, crit_value

    def two_sample_proportion(self, count1_column, nobs1_column, count2_column, nobs2_column, prop, alternative='two-sided', alpha=0.05):
        """
        双样本比例检验
        :param count1_column: str，样本1中成功次数的列名
        :param nobs1_column: str，样本1的样本量的列名
        :param count2_column: str，样本2中成功次数的列名
        :param nobs2_column: str，样本2的样本量的列名
        :param prop: float，假设比例
        :param alternative: str，假设检验类型（'two-sided'、'less' 或 'greater'）
        :param alpha: float，显著性水平（默认为 0.05）
        :return: z-statistic, p-value, critical value
        """
        count1 = self.dataframe[count1_column].sum()
        nobs1 = self.dataframe[nobs1_column].sum()
        count2 = self.dataframe[count2_column].sum()
        nobs2 = self.dataframe[nobs2_column].sum()
        
        # 计算 z 统计量和 p 值
        stat, p_val = proportions_ztest([count1, count2], [nobs1, nobs2], value=prop, alternative=alternative)
        
        # 计算临界值
        if alternative == 'two-sided':
            crit_value_low = stats.norm.ppf(alpha / 2)
            crit_value_high = stats.norm.ppf(1 - alpha / 2)
            crit_value = (crit_value_low, crit_value_high)
            # 对于双尾检验，p-value 是 z 统计量绝对值的两侧面积
            p_val = 2 * (1 - stats.norm.cdf(np.abs(stat)))
        elif alternative == 'less':
            crit_value = stats.norm.ppf(alpha)
            # 对于单尾检验（小于），p-value 是 z 统计量的左侧面积
            p_val = stats.norm.cdf(stat)
        elif alternative == 'greater':
            crit_value = stats.norm.ppf(1 - alpha)
            # 对于单尾检验（大于），p-value 是 z 统计量的右侧面积
            p_val = 1 - stats.norm.cdf(stat)
        else:
            raise ValueError("alternative 必须是 'two-sided'、'less' 或 'greater'")
        
        return stat, p_val, crit_value
    def multiple_sample_proportion(self, count_columns, nobs_columns, prop, alternative='two-sided', alpha=0.05):
        """
        多样本比例检验（列联表）
        :param count_columns: list，包含每个样本成功次数的列名列表
        :param nobs_columns: list，包含每个样本总样本量的列名列表
        :param prop: float，假设比例
        :param alternative: str，假设检验类型（'two-sided'、'less' 或 'greater'）
        :param alpha: float，显著性水平（默认为 0.05）
        :return: chi2-statistic, p-value, critical value
        """
        # 计算每个样本的成功次数和总样本量
        counts = [self.dataframe[col].sum() for col in count_columns]
        nobs = [self.dataframe[col].sum() for col in nobs_columns]
        
        # 使用 proportions_chisquare 进行多样本比例检验
        stat, p_val, _ = proportions_chisquare(counts, nobs, value=prop)
        
        # 计算临界值
        if alternative == 'two-sided':
            crit_value_low = stats.chi2.ppf(alpha / 2, len(counts) - 1)
            crit_value_high = stats.chi2.ppf(1 - alpha / 2, len(counts) - 1)
            crit_value = (crit_value_low, crit_value_high)
            # 对于双尾检验，p-value 是 chi2 统计量绝对值的两侧面积
            p_val = 2 * (1 - stats.chi2.cdf(np.abs(stat), len(counts) - 1))
        elif alternative == 'less':
            crit_value = stats.chi2.ppf(alpha, len(counts) - 1)
            # 对于单尾检验（小于），p-value 是 chi2 统计量的左侧面积
            p_val = stats.chi2.cdf(stat, len(counts) - 1)
        elif alternative == 'greater':
            crit_value = stats.chi2.ppf(1 - alpha, len(counts) - 1)
            # 对于单尾检验（大于），p-value 是 chi2 统计量的右侧面积
            p_val = 1 - stats.chi2.cdf(stat, len(counts) - 1)
        else:
            raise ValueError("alternative 必须是 'two-sided'、'less' 或 'greater'")
        
        return stat, p_val, crit_value


# 示例用法:
if __name__ == "__main__":
    # 创建示例 DataFrame
    df = pd.DataFrame({
        'sample1': np.random.normal(0, 1, 100),
        'sample2': np.random.normal(0.5, 1, 100),
        'sample3': np.random.normal(1, 1, 100),
        'success_count1': np.random.randint(0, 100, 100),
        'total_count1': np.random.randint(100, 200, 100),
        'success_count2': np.random.randint(0, 100, 100),
        'total_count2': np.random.randint(100, 200, 100)
    })

    ht = HypothesisTesting(df)
    
    # 单样本均值检验
    t_stat, p_val, crit_value = ht.one_sample_mean('sample1', 0, alternative='two-sided')
    print("单样本均值检验:", t_stat, p_val, crit_value)

    # 双样本均值检验
    t_stat, p_val, crit_value = ht.two_sample_mean('sample1', 'sample2', sigma1=None, sigma2=None, equal_var=True, alternative='two-sided', alpha=0.05)
    print("双样本均值检验:", t_stat, p_val, crit_value)
    #多样本均值检验
    f_stat, p_val, crit_value = ht.multiple_sample_mean(['sample1', 'sample2', 'sample3'], alternative='two-sided')
    print("多样本均值检验:", f_stat, p_val, crit_value)

    # 单样本方差检验
    chi2_stat, p_val, crit_value = ht.one_sample_variance('sample1', 1, mean_known=None, alternative='two-sided', alpha=0.05)
    print("单样本方差检验:", chi2_stat, p_val, crit_value)

    # 双样本方差检验
    f_stat, p_val, crit_value = ht.two_sample_variance('sample1', 'sample2', mean1=None, mean2=None, alternative='two-sided', alpha=0.05)
    print("双样本方差检验:", f_stat, p_val, crit_value)

    # 多样本方差检验
    f_stat, p_val, crit_value = ht.multiple_sample_variance(['sample1', 'sample2', 'sample3'], alternative='two-sided')
    print("多样本方差检验:", f_stat, p_val, crit_value)

    # 单样本比例检验
    z_stat, p_val, crit_value = ht.one_sample_proportion('success_count1', 'total_count1', 0.3, alternative='two-sided')
    print("单样本比例检验:", z_stat, p_val, crit_value)

    # 双样本比例检验
    z_stat, p_val, crit_value = ht.two_sample_proportion('success_count1', 'total_count1', 'success_count2', 'total_count2', 0.3, alternative='two-sided')
    print("双样本比例检验:", z_stat, p_val, crit_value)

    # 多样本比例检验（列联表）
    chi2_stat, p_val, crit_value = ht.multiple_sample_proportion(['success_count1', 'success_count2'], ['total_count1', 'total_count2'],0.3,alternative='two-sided')
    print("多样本比例检验:", chi2_stat, p_val, crit_value)

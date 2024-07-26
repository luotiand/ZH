## HypothesisTesting

`HypothesisTesting` 是一个用于执行各种假设检验的类。它包含单样本、双样本和多样本的均值、方差和比例检验方法。

### 初始化类

- **Description**: 初始化 HypothesisTesting 类。
- **Parameters**: 
  - `dataframe` (pd.DataFrame): 待处理的数据框。

### one_sample_mean

- **Description**: 单样本均值检验。
- **Parameters**: 
  - `column` (str): 待检验的列名。
  - `popmean` (float): 假设总体均值。
  - `sigma` (float): 假设总体方差，默认值为None
- **Returns**: t-statistic, p-value

### two_sample_mean

- **Description**: 双样本均值检验。
- **Parameters**: 
  - `column1` (str): 待检验的第一列名。
  - `column2` (str): 待检验的第二列名。
  - `equal_var` (bool): 是否假设方差相等，默认值为 True。
- **Returns**: t-statistic, p-value


### multiple_sample_mean

- **Description**: 多样本均值检验（单因素方差分析）。
- **Parameters**: 
  - `columns` (list of str): 可变数量的待检验的列名。
- **Returns**: F-statistic, p-value


### one_sample_variance

- **Description**: 单样本方差检验。
- **Parameters**: 
  - `column` (str): 待检验的列名。
  - `popvar` (float): 假设总体方差。
- **Returns**: chi2-statistic, p-value

### two_sample_variance

- **Description**: 双样本方差检验。
- **Parameters**: 
  - `column1` (str): 待检验的第一列名。
  - `column2` (str): 待检验的第二列名。
- **Returns**: F-statistic, p-value

### multiple_sample_variance

- **Description**: 多样本方差检验（单因素方差分析）。
- **Parameters**: 
  - `columns` (list of str): 可变数量的待检验的列名。
- **Returns**: F-statistic, p-value

### one_sample_proportion

- **Description**: 单样本比例检验。
- **Parameters**: 
  - `count_column` (str): 成功次数的列名。
  - `nobs_column` (str): 总样本量的列名。
  - `prop` (float): 假设比例。
- **Returns**: z-statistic, p-value

### two_sample_proportion

- **Description**: 双样本比例检验。
- **Parameters**: 
  - `count1_column` (str): 样本1中成功次数的列名。
  - `nobs1_column` (str): 样本1的样本量的列名。
  - `count2_column` (str): 样本2中成功次数的列名。
  - `nobs2_column` (str): 样本2的样本量的列名。
- **Returns**: z-statistic, p-value


### chi2_contingency

- **Description**: 多样本比例检验（列联表）。
- **Parameters**: 
  - `columns` (list of str): 列联表的列名列表。
- **Returns**: chi2-statistic, p-value, degrees of freedom, expected frequencies
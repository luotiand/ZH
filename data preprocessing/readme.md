# 异常值处理

## OutlierDetection 


### sigma_rule
- **Description**: 使用 sigma 法则检测异常值
- **Parameters**:
  - `sigma_level` (int): sigma 水平，默认值为 3
- **Returns**: pandas DataFrame 包含异常值

### boxplot_method
- **Description**: 使用箱线图方法检测异常值
- **Parameters**:
  - `IQR` (float): 箱线图系数，默认值为 1.5
- **Returns**: pandas DataFrame 包含异常值

### grubbs_test
- **Description**: 使用 Grubbs 检验检测异常值
- **Parameters**:
  - `alpha` (float): 显著性水平，默认值为 0.05
- **Returns**: pandas DataFrame 包含异常值

### dixon_test
- **Description**: 使用 Dixon 检验检测异常值
- **Parameters**:
  - `alpha` (float): 显著性水平，默认值为 0.05
- **Returns**: pandas DataFrame 包含异常值

### quantile_method
- **Description**: 使用分位数方法检测异常值
- **Parameters**:
  - `lower_quantile` (float): 下分位数，默认值为 0.05
  - `upper_quantile` (float): 上分位数，默认值为 0.95
- **Returns**: pandas DataFrame 包含异常值


## DataFrameTransformer

### str_to_num
- **Description**: 将指定列中的 'num' 字符串转化为数字
- **Parameters**:
  - `column` (str): 待处理的列名
- **Returns**: 处理后的 pandas DataFrame

### num_to_str
- **Description**: 将指定列中的数字转化为 'num' 字符串
- **Parameters**:
  - `column` (str): 待处理的列名
- **Returns**: 处理后的 pandas DataFrame

### transpose_matrix
- **Description**: 将 DataFrame 进行转置
- **Parameters**: 无
- **Returns**: 转置后的 pandas DataFrame
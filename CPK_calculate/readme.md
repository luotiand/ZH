## ProcessCapability

`ProcessCapability` 是一个用于计算过程能力指数（Cpk）以及绘制概率图和概率密度曲线图的类。它还可以计算数据在规范限内的合格率。

### 初始化类

- **Description**: 初始化 `ProcessCapability` 类。
- **Parameters**:
  - `data` (pd.Series): 待处理的数据。

### 方法

#### calculate_cpk

- **Description**: 计算 Cpk 值。
- **Parameters**:
  - `USL` (float): 上规格限。
  - `LSL` (float): 下规格限。
  - `method` (str): 方差计算方法，'r/d2' 或 's/c4' 或 'std'。
  - `num_groups` (int): 将数据样本分成的组数（默认为 1）。
- **Returns**: float，Cpk 值。

#### plot

- **Description**: 绘制概率图和概率密度曲线图。
- **Parameters**: 无。

#### calculate_acceptance_rate

- **Description**: 计算规范上限和规范下限之间的合格率。
- **Parameters**:
  - `USL` (float): 上规格限。
  - `LSL` (float): 下规格限。
- **Returns**: float，合格率。

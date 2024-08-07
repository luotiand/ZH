# ProcessCapability 类文档

`ProcessCapability` 类用于计算和可视化过程能力指标，支持多种控制图和能力图的绘制。该类包括以下主要功能：

## 类定义

### `ProcessCapability`

#### 方法

##### `calculate_cpk(self, USL=None, LSL=None, method='s/c4', num_groups=1)`

计算过程能力指标 Cpk 和相关的统计数据。

- `USL` (float): 上规格限。
- `LSL` (float): 下规格限。
- `method` (str): 方差计算方法，支持 `'r/d2'`、`'s/c4'` 和 `'merged'`。
- `num_groups` (int): 数据样本分成的组数（默认为 1）。

返回一个包含以下键的字典：
- `n`: 数据样本数量。
- `mean`: 数据均值。
- `std_dev`: 数据标准差。
- `avg_std_dev`: 平均标准差。
- `LSL`: 下规格限。
- `ppm_low`: 低于下规格限的百万分率。
- `cpl`: 下规格限能力指数。
- `ppl`: 下规格限的预期能力指数。
- `USL`: 上规格限。
- `ppm_up`: 高于上规格限的百万分率。
- `cpu`: 上规格限能力指数。
- `ppu`: 上规格限的预期能力指数。
- `cp`: 过程能力指数。
- `pp`: 过程预期能力指数。
- `cpk`: 过程能力指数 (调整后的)。
- `ppk`: 过程预期能力指数 (调整后的)。
- `ca`: 偏移系数。

##### `plot_xr_chart(self)`

绘制 X-bar 和 R 图（均值图和极差图）。

返回一个包含以下键的字典：
- `x_bar`: X-bar 均值。
- `r_bar`: R 均值。
- `mean`: 数据均值。
- `std_dev`: 数据标准差。
- `UCLx`: X-bar 图的上控制限。
- `LCLx`: X-bar 图的下控制限。
- `UCLr`: R 图的上控制限。
- `LCLr`: R 图的下控制限。

##### `plot_xs_chart(self)`

绘制 X-bar 和 S 图（均值图和标准差图）。

返回一个包含以下键的字典：
- `x_bar`: X-bar 均值。
- `s_bar`: S 均值。
- `mean`: 数据均值。
- `std_dev`: 数据标准差。
- `UCLx`: X-bar 图的上控制限。
- `LCLx`: X-bar 图的下控制限。
- `UCLs`: S 图的上控制限。
- `LCLs`: S 图的下控制限。

##### `plot_imr_chart(self)`

绘制 I 图和 MR 图（个体值图和移动范围图）。

返回一个包含以下键的字典：
- `x_bar`: I 图均值。
- `mr_bar`: MR 图均值。
- `mean`: 数据均值。
- `std_dev`: 数据标准差。
- `UCLx`: I 图的上控制限。
- `LCLx`: I 图的下控制限。
- `UCLmr`: MR 图的上控制限。
- `LCLmr`: MR 图的下控制限。

##### `plot_capability_chart(self, usl, lsl)`

绘制过程能力图。

- `usl` (float): 上规格限。
- `lsl` (float): 下规格限。

返回一个包含以下键的字典：
- `cp`: 过程能力指数。
- `cpk`: 过程能力指数 (调整后的)。
- `pp`: 过程预期能力指数。
- `ppk`: 过程预期能力指数 (调整后的)。
- `usl`: 上规格限。
- `lsl`: 下规格限。
- `mean`: 数据均值。
- `std_dev`: 数据标准差。
- `ppm_low`: 低于下规格限的百万分率。
- `ppm_up`: 高于上规格限的百万分率。


# special_Charts 类文档

`special_Charts` 类用于绘制特殊类型的控制图，包括基于几何分布的 G 控制图和基于威布尔分布的 T 控制图。该类包括以下主要功能：

## 类定义

### `special_Charts`

#### 方法

##### `__init__(self, data, chart_type='G')`

初始化 `special_Charts` 类。

- `data` (pandas.Series): 输入的数据，必须是一个 Series，表示时间间隔或产品数间隔。
- `chart_type` (str): 控制图类型，支持 `'G'` 或 `'T'`（默认为 `'G'`）。

##### `calculate_control_limits(self)`

计算控制限。

返回：
- `UCL` (float): 上控制限。
- `CL` (float): 中心线（均值）。
- `LCL` (float): 下控制限。

##### `plot_chart(self)`

绘制控制图。

该方法使用 `matplotlib` 绘制数据点、上控制限、下控制限和中心线。图表标题根据 `chart_type` 自动设置为 G 控制图或 T 控制图。


# ControlCharts 类文档

`ControlCharts` 类用于生成并绘制四种类型的控制图：p 图、np 图、u 图和 c 图。以下是该类的主要功能和方法：

## 类定义

### `ControlCharts`

#### 方法

##### `__init__(self, df)`

初始化 `ControlCharts` 类。

- `df` (pandas.DataFrame): 输入的数据框，包含缺陷数和总数，列名应为 `['Defects', 'Total']`。

##### `plot_p_chart(self)`

绘制 p 图，并计算中心线（p̄）和控制限（UCL 和 LCL）。

返回：
- `calculations` (dict): 包含 p̄、UCL 和 LCL 的平均值。

##### `plot_np_chart(self)`

绘制 np 图，并计算中心线（np̄）和控制限（UCL 和 LCL）。

返回：
- `calculations` (dict): 包含 np̄、UCL 和 LCL 的平均值。

##### `plot_u_chart(self)`

绘制 u 图，并计算中心线（ū）和控制限（UCL 和 LCL）。

返回：
- `calculations` (dict): 包含 ū、UCL 和 LCL 的平均值。

##### `plot_c_chart(self)`

绘制 c 图，并计算中心线（c̄）和控制限（UCL 和 LCL）。

返回：
- `calculations` (dict): 包含 c̄、UCL 和 LCL 的平均值。


# LaneyControlChart 类文档

`LaneyControlChart` 类用于生成并绘制 Laney 控制图，支持两种类型：p 图和 u 图。Laney 控制图是一种改进的控制图，用于处理数据的变异性。

## 类定义

### `LaneyControlChart`

#### 方法

##### `__init__(self, df, chart_type='u')`

初始化 `LaneyControlChart` 类。

- `df` (pandas.DataFrame): 输入的数据框，包含缺陷数和总数，列名应为 `['Defects', 'Total']`。
- `chart_type` (str): 控制图类型，'p' 或 'u'。默认为 'u'。

##### `calculate_sigma_z_p(self)`

计算 p 图的 `sigma_z`，用于确定控制限。

返回：
- `sigma_z` (float): 计算得到的 `sigma_z` 值。

##### `calculate_sigma_z_u(self)`

计算 u 图的 `sigma_z`，用于确定控制限。

返回：
- `sigma_z` (float): 计算得到的 `sigma_z` 值。

##### `calculate_control_limits(self)`

计算控制图的上下控制限（UCL 和 LCL）。

返回：
- `UCL` (float): 上控制限。
- `LCL` (float): 下控制限（小于 0 的值会被限制为 0）。

##### `calculate_confidence_interval(self)`

计算置信区间的上下限。

返回：
- `ci_lower` (float): 置信区间下限。
- `ci_upper` (float): 置信区间上限。

##### `plot_chart(self)`

绘制控制图和散点图，并进行线性回归分析。

功能：
- 计算并绘制控制图的上下控制限和中心线。
- 根据控制图类型，绘制相应的散点图与线性回归线。
- 显示置信区间及相关统计信息。



# ChartRules 类文档

`ChartRules` 类用于应用不同的控制图规则来检测异常点。此类支持检查各种控制图规则，并能够处理 p 图的异常情况。

## 类定义

### `ChartRules`

#### 方法

##### `__init__(self, p, UCL, LCL)`

初始化 `ChartRules` 类。

- `p` (pandas.Series): 各样本的缺陷比例。
- `UCL` (float, list, or pandas.Series): 上控制限。如果是单一值，则会被扩展为与 `p` 长度相同的列表。
- `LCL` (float, list, or pandas.Series): 下控制限。如果是单一值，则会被扩展为与 `p` 长度相同的列表。

##### `points_outside_control_limits(self)`

检测所有超出控制限（UCL 或 LCL）的点。

返回：
- (pandas.DataFrame): 超出控制限的点。

##### `six_consecutive_points_trending(self)`

检测是否存在 6 连升或 6 连降的趋势。

返回：
- (pandas.DataFrame): 存在 6 连升或 6 连降趋势的点。

##### `fourteen_points_alternating(self)`

检测是否存在连续 14 个点在中线上下交替。

返回：
- (pandas.Series): 连续 14 个点在中线上下交替的结果。

##### `fifteen_points_in_zone_c(self)`

检测是否存在连续 15 个点都在 C 区。

返回：
- (pandas.DataFrame): 连续 15 个点都在 C 区的结果。

##### `eight_points_outside_zone_c(self)`

检测是否存在连续 8 个点不在 C 区。

返回：
- (pandas.DataFrame): 连续 8 个点不在 C 区的结果。

##### `three_points_in_zone_a(self)`

检测是否存在连续 3 个点中有 2 个点在 A 区。

返回：
- (pandas.DataFrame): 连续 3 个点中有 2 个点在 A 区的结果。

##### `four_of_five_points_in_zone_b_or_beyond(self)`

检测是否存在连续 5 个点中有 4 个点不在 C 区。

返回：
- (pandas.DataFrame): 连续 5 个点中有 4 个点不在 C 区的结果。

##### `nine_points_on_one_side_of_center_line(self)`

检测是否存在连续 9 个点在中线的同一侧。

返回：
- (pandas.DataFrame): 连续 9 个点在中线同一侧的结果。


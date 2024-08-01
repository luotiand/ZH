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
  - `USL` (float): 上规格限,默认None。
  - `LSL` (float): 下规格限，默认None。
  - `method` (str): 方差计算方法，'r/d2' 或 's/c4' 或 'std'。
  - `num_groups` (int): 将数据样本分成的组数（默认为 1）。
- **Returns**: calculations = {
            'n':n,
            'mean': self.mean,
            'std_dev': self.std_dev,
            'avg_std_dev': avg_std_dev,
            'USL': self.USL,
            'LSL': self.LSL,
            'cp':cp,
            'pp':pp,
            'cpu':cpu,
            'cpl':cpl,
            'cpk': cpk,
            'ppu':ppu,
            'ppl':ppl,
            'ppk':ppk,
            'ppm_low':ppm_low,
            'ppm_up':ppm_up}
### `plot_p_chart`

绘制缺陷比例 (P Chart) 控制图。

**Description**:
- 计算缺陷比例的中心线 (p̄)、上控制限 (UCL) 和下控制限 (LCL)。
- 绘制缺陷比例图及其中心线、UCL 和 LCL。

**return**:
- 控制图的计算结果，包括 p̄、UCL 和 LCL。

### `plot_np_chart`

绘制缺陷数量 (NP Chart) 控制图。

**Description**:
- 计算缺陷数量的中心线 (np̄)、上控制限 (UCL) 和下控制限 (LCL)。
- 绘制缺陷数量图及其中心线、UCL 和 LCL。

**return**:
- 控制图的计算结果，包括 np̄、UCL 和 LCL。

### `plot_u_chart`

绘制单位缺陷数 (U Chart) 控制图。

**Description**:
- 计算单位缺陷数的中心线 (ū)、上控制限 (UCL) 和下控制限 (LCL)。
- 绘制单位缺陷数图及其中心线、UCL 和 LCL。

**return**:
- 控制图的计算结果，包括 ū、UCL 和 LCL。

### `plot_c_chart`

绘制缺陷总数 (C Chart) 控制图。

**Description**:
- 计算缺陷总数的中心线 (c̄)、上控制限 (UCL) 和下控制限 (LCL)。
- 绘制缺陷总数图及其中心线、UCL 和 LCL。

**return**:
- 控制图的计算结果，包括 c̄、UCL 和 LCL。

### `plot_xr_chart`

绘制均值和极差 (X-bar 和 R Chart) 控制图。

**Description**:
- 计算每个样本的均值和极差。
- 计算均值图和极差图的中心线 (X̄ 和 R̄)、上控制限 (UCLx 和 UCLr) 和下控制限 (LCLx 和 LCLr)。
- 绘制均值图和极差图。

**return**:
- 控制图的计算结果，包括 X̄、R̄、UCLx 和 LCLr。

### `plot_xs_chart`

绘制均值和标准差 (X-bar 和 S Chart) 控制图。

**Description**:
- 计算每个样本的均值和标准差。
- 计算均值图和标准差图的中心线 (X̄ 和 S̄)、上控制限 (UCLx 和 UCLs) 和下控制限 (LCLx 和 LCLs)。
- 绘制均值图和标准差图。

**return**:
- 控制图的计算结果，包括 X̄、S̄、UCLx 和 LCLs。

### `plot_imr_chart`

绘制单个值和移动范围 (I-MR Chart) 控制图。

**Description**:
- 计算每个样本的单个值和移动范围。
- 计算单个值图和移动范围图的中心线 (X̄ 和 MR̄)、上控制限 (UCLx 和 UCLmr) 和下控制限 (LCLx 和 LCLmr)。
- 绘制单个值图和移动范围图。

**return**:
- 控制图的计算结果，包括 X̄、MR̄、UCLx 和 LCLmr。




## `plot_capability_chart`
**Description**:
- 绘制过程能力图，包括直方图、正态分布曲线和控制限。

**Parameters**:
- `usl` (float): 上规格限 (USL)。
- `lsl` (float): 下规格限 (LSL)。

**Returns**:
- `dict`: 包含计算出的 Cp 和 Cpk 值。

---

## `plot_moving_average_chart`
**Description**:
- 绘制移动平均控制图，包括数据点、移动平均线和控制限。

**Returns**:
- `dict`: 包含移动平均线的均值、标准差、UCL 和 LCL。

---

## `plot_ewma_chart`
**Description**:
- 绘制指数加权移动平均 (EWMA) 控制图，包括数据点、EWMA 线和控制限。

**Parameters**:
- `lambda_value` (float): 权重参数，决定新数据点在 EWMA 计算中的权重，默认为 0.2。

**Returns**:
- `dict`: 包含均值、标准差、UCL 和 LCL。

---

## `plot_cv_control_chart`
**Description**:
- 绘制变异系数 (CV) 控制图，包括 CV 值和控制限。

**Parameters**:
- `mean_known` (bool): 是否已知均值，默认为 False。
- `std_known` (bool): 是否已知标准差，默认为 False。
- `mean_value` (float, optional): 已知的均值。
- `std_value` (float, optional): 已知的标准差。

**Returns**:
- `dict`: 包含均值 CV、标准差 CV、UCL 和 LCL。

---

## `plot_cusum_chart`
**Description**:
- 绘制 CUSUM 控制图，包括累积和和控制限。

**Returns**:
- `dict`: 包含 K1、K2、CUSUM+ 和 CUSUM-。

---

## `plot_count_cusum_chart`
**Description**:
- 绘制计数型 CUSUM 控制图，包括累积和和控制限。

**Parameters**:
- `P0` (float): 基线比例。
- `P1` (float): 测量比例。

**Returns**:
- `dict`: 包含 K1、K2、CUSUM+、CUSUM- 和样本大小。

---

## `plot_chain_chart`
**Description**:
- 绘制链图（Cumulative Sum Control Chart）。

**Returns**:
- `numpy.ndarray`: 包含链图的累积和数据。



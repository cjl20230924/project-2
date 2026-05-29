# 监测法内照射剂量计算系统 v1.0

## 概述

本程序基于 **GB/T 16148** 及 **ICRP Publication 54/78/130**，使用 PyQt5 桌面框架实现，
与 `dose_app.py`（空气采样法）保持一致的代码风格和数据路径约定。

---

## 计算方法（GB/T 16148）

```
E = I × e(AMAD)
I = M / m(T)
```

| 符号 | 含义 | 来源 |
|------|------|------|
| M   | 生物样品测量值（Bq 或 Bq/d） | 用户输入 |
| m   | m 值（剂量当量系数，与 M 同量纲） | `m_data/` 数据文件 |
| I   | 放射性核素摄入量（Bq） | I = M / m |
| e   | 剂量系数（Sv/Bq），按 AMAD 双对数插值 | `processed_nuclide_files/` |
| AMAD| 活性中值空气动力学直径（μm） | 用户选择或手动输入 |
| E   | 有效剂量（Sv） | E = I × e |

---

## 目录结构依赖

程序依赖项目根目录（`../`）下的两个数据文件夹：

```
project_01/only2/
├── m_data/                     # m 值数据（按核素分 routine/special CSV）
│   ├── U_238_routine.csv
│   ├── U_238_special.csv
│   └── ...
├── processed_nuclide_files/    # 剂量系数数据（parquet/xlsx）
│   ├── Processed_U-238.parquet
│   └── ...
└── new1/
    ├── monitoring_dose_app.py  ← 主程序
    └── README_monitoring.md
```

---

## 运行方式

```bash
# 在 new1/ 目录下执行（或在项目根目录亦可）
python new1/monitoring_dose_app.py
```

依赖包：
```
PyQt5 >= 5.15
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
scikit-learn >= 1.0
openpyxl    # xlsx 支持
pyarrow     # parquet 支持
```

---

## 功能说明

### 监测类型切换

- **常规监测**：对应 `*_routine.csv`，默认摄入途径为 Inhalation
- **应急监测**：对应 `*_special.csv`，支持 Inhalation / Ingestion 等多途径

### 参数面板（左侧）

| 步骤 | 操作 |
|------|------|
| ① 核素选择 | 先选元素，再选核素（自动联动） |
| ② 监测参数 | 依次选择气溶胶类型、监测方法、时间(d)，唯一确定 m 值 |
| ③ 测量值 M | 输入生物样品测量值，自动显示 I = M/m |
| ④ 粒径 AMAD | 从预设列表（0.001~20 μm）选择或切换为"自定义"手动输入 |
|   | 程序自动对剂量系数进行双对数插值 |
| ⑤ 有效剂量 | 点击「计算剂量」计算 E，再点「添加到列表」记录 |

### 粒径调整

- 预设粒径：`0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0, 20.0 μm`
- 选择"自定义"后出现 QDoubleSpinBox，可输入任意 0.001~100 μm
- 剂量系数按**双对数线性插值**：$\ln e \sim \ln d_p$（与 xf_core 保持一致）

### 结果列表（右侧）

- 表格显示所有已添加项，可单独删除
- 底部实时汇总总有效剂量
- 支持导出为 CSV（UTF-8-BOM，Excel 可直接打开）

---

## m_data 文件格式

### 常规监测（`*_routine.csv`）

```csv
radionuclide, Aerosols type, Monitoring method, Period (d), m(T/2)
U-238, Type F, Urine, 360, 3.1e-5
```

### 应急监测（`*_special.csv`）

```csv
radionuclide, Route of intake, Aerosols type, Monitoring method, Time (d), m(t), fA
U-238, Inhalation, Type F, Urine, 1, 1.8e-1,
```

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2026-05-29 | 初版；GB/T 16148；PyQt5；粒径可调；常规/应急双模式 |

---

## 参考标准

- GB/T 16148-2009《放射性核素摄入量及内照射剂量估算规范》
- ICRP Publication 54 (1988) — Individual Monitoring for Intakes of Radionuclides by Workers
- ICRP Publication 78 (1997) — Individual Monitoring for Internal Exposure of Workers
- ICRP Publication 130 (2015) — Occupational Intakes of Radionuclides Part 1

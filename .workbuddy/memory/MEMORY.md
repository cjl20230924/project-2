# MEMORY.md - 长期记忆

## 项目：剂量计算系统 (dose_app.py)

### 当前版本
v4.0 - 动态分级表格 + 逐级化合物配置（2026-05-27 完成）

### 技术架构
- 基于 PyQt5 桌面应用
- 核心拟合逻辑复用 xf_core.py（通过 _patch_xf_stages / _restore_xf_stages 临时替换全局数组）
- 支持 4 种计算方法：std（5μm固定）、modal（单/双峰拟合）、probit（正态概率图）、stage（逐级独立）
- 双峰模式采用逐级 PDF 权重分配：w1_i = (frac1 * pdf(d_i, a1, g1)) / denom
- 动态效率校正：非标准级数时通过几何平均粒径插值默认 9 级效率值
- 数据结构：self._stages = [{name, low, high, conc, compounds: [{compound, aerosol_type, activity_fraction, nuclides}]}]

### 关键文件
- dose_app.py - 主程序（1210行，v4.0）
- xf_core.py - 拟合核心（不修改，通过 patch 模式适配）
- processed_nuclide_files/ - 核素数据（.parquet / .xlsx）
- new1/monitoring_dose_app.py - 监测法剂量计算系统（v2.3, z(t)函数法）
- z_data/ - z(t)数据目录：U_235/U_238/Pu_239/Pu_240 四个核素的 txt+csv
- z_data/process_z_data.py - 批量解析 *.txt → *_zdata.csv + all_zdata.csv
- z_data/all_zdata.csv - 合并全量（75898 行，4核素×3途径）

### 监测法系统架构（v2.4）
- 公式：E(50) = M × z(t)
- 数据加载：懒加载分文件策略，启动扫描索引 → 选核素时按需加载对应 *_zdata.csv（缓存复用）；兼容模式自动退回 all_zdata.csv 拆分
- Inhalation：material(Type F/M/S) + amad_um(粒径) → 双对数AMAD插值 + 线性时间插值
- Ingestion/Injection：material(含fA，如 "Soluble forms, fA=2E-2") → 直接线性时间插值
- 样本列按途径过滤（Ingestion 无 lungs 列）
- 常规监测：t = T/2；应急监测：t 直接输入
- 时间精度：6位小数（spin_T.setDecimals(6)，最小值0.000001天）
- 核素选择：元素(cb_element) → 核素(cb_nuclide) 两级，同行紧凑布局

### 运行
- 开发：python dose_app.py
- 打包：PyInstaller 单文件，需包含数据文件

### 用户偏好
- 中文交流，结构化输出（表格、编号列表）
- 代码修改提供完整文件和 GitHub 日志摘要
- 评审模式：先优化建议后澄清问题

### 联系方式
- chenjialu08@163.com

"""
监测法内照射剂量计算系统 v2.4 (桌面版)
基于 PyQt5 实现，依据 GB/T 16148-2009 —— z(t) 函数法

计算方法：
  E(50) = M × z(t)

  其中
    M     — 生物样品测量值（Bq 或 Bq/d）
    z(t)  — 剂量-含量转换函数（Sv/Bq），从 z_data 查表插值
    * 时间 t: 常规监测 t = T/2; 应急监测直接输入（支持6位小数精度）
    * AMAD 插值: 双对数 (log-log)  时间插值: 线性 (linear)

数据加载策略（懒加载）：
  启动时仅扫描 z_data/ 目录下 *_zdata.csv 文件名，建立索引。
  选择核素时按需加载对应单文件，已加载的缓存复用，内存友好。

操作步骤：
  ① 监测类型 → ② 元素→核素 + 摄入途径 → ③ 物质/fA
  → ④ 样本类型 → ⑤ (吸入)AMAD粒径 → ⑥ 时间 → ⑦ 测量值M → ⑧ 计算
"""

# =====================================================================
#   导入
# =====================================================================
import sys, os, warnings, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QPushButton,
    QRadioButton, QButtonGroup, QTableWidget, QTableWidgetItem,
    QSplitter, QScrollArea, QFrame, QDoubleSpinBox,
    QFileDialog, QMessageBox, QHeaderView,
    QSizePolicy, QStatusBar, QAbstractItemView,
)
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# =====================================================================
#   路径 & 懒加载数据层
# =====================================================================
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
Z_DATA_DIR = HERE.parent / 'z_data'

# 索引：{核素名 -> csv 路径}，启动时扫描建立
_nuc_index: dict[str, Path] = {}
# 缓存：{核素名 -> DataFrame}，按需加载后复用
_nuc_cache: dict[str, pd.DataFrame] = {}

_META_COLS = frozenset({'radionuclide', 'route_of_intake', 'material', 'fA', 'amad_um', 'time_days'})


def scan_z_data():
    """
    扫描 z_data 目录，建立「核素 → CSV 文件」索引。
    不加载任何数据（懒加载），仅记录可用核素列表。

    文件命名约定：  <nuclide_stem>_zdata.csv
      例：U_235_zdata.csv → 核素名从文件内首行读取
          all_zdata.csv   → 忽略（全量合并文件，不参与懒加载）

    若 z_data/ 下只有 all_zdata.csv 而无分文件，则退回兼容模式：
    将 all_zdata.csv 按核素拆分后缓存。
    """
    global _nuc_index, _nuc_cache
    _nuc_index.clear()
    _nuc_cache.clear()

    csv_files = [p for p in sorted(Z_DATA_DIR.glob('*_zdata.csv'))
                 if p.stem != 'all_zdata']  # 排除合并文件

    if csv_files:
        # ── 正常模式：逐文件扫描首行确认核素名 ──
        for p in csv_files:
            try:
                head = pd.read_csv(p, nrows=1)
                if 'radionuclide' in head.columns:
                    nuc = str(head['radionuclide'].iloc[0]).strip()
                    if nuc and nuc != 'nan':
                        _nuc_index[nuc] = p
                        print(f'[z_data] 索引: {nuc} → {p.name}')
            except Exception as e:
                print(f'[z_data] 跳过 {p.name}: {e}')

        if _nuc_index:
            print(f'[z_data] 扫描完成，可用核素: {sorted(_nuc_index.keys())}')
            return

    # ── 兼容模式：加载 all_zdata.csv 并拆分 ──
    all_csv = Z_DATA_DIR / 'all_zdata.csv'
    if all_csv.exists():
        print(f'[z_data] 分文件未找到，加载 all_zdata.csv 并按核素拆分缓存…')
        df_all = _load_csv(all_csv)
        for nuc, grp in df_all.groupby('radionuclide'):
            _nuc_cache[str(nuc)] = grp.reset_index(drop=True)
            _nuc_index[str(nuc)] = all_csv   # 占位路径
        print(f'[z_data] 兼容模式加载完成: {sorted(_nuc_cache.keys())}')
    else:
        print(f'[警告] z_data 目录下无任何 *_zdata.csv，请先运行 process_z_data.py')


def _load_csv(path: Path) -> pd.DataFrame:
    """读取 CSV 并统一数值列类型。"""
    df = pd.read_csv(path)
    for c in ['time_days', 'amad_um']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def get_nuclide_df(nuclide: str) -> pd.DataFrame:
    """
    按核素名获取 DataFrame（懒加载 + 缓存）。
    返回空 DataFrame 表示数据不可用。
    """
    if nuclide in _nuc_cache:
        return _nuc_cache[nuclide]

    path = _nuc_index.get(nuclide)
    if path is None:
        return pd.DataFrame()

    try:
        df = _load_csv(path)
        # 只保留当前核素（防止 all_zdata.csv 兼容模式下混入其他核素）
        if 'radionuclide' in df.columns:
            df = df[df['radionuclide'] == nuclide].reset_index(drop=True)
        _nuc_cache[nuclide] = df
        print(f'[z_data] 已加载 {nuclide}: {len(df)} 行')
        return df
    except Exception as e:
        print(f'[z_data] 加载 {nuclide} 失败: {e}')
        return pd.DataFrame()


def available_nuclides() -> list[str]:
    """返回所有已索引的核素名（有序）。"""
    return sorted(_nuc_index.keys())


def available_elements() -> list[str]:
    """
    从核素名提取元素符号（取连字符前部分）。
    例：'U-235' → 'U'，'Pu-239' → 'Pu'
    返回有序去重列表。
    """
    elems = []
    for nuc in _nuc_index:
        elem = nuc.split('-')[0].strip()
        if elem and elem not in elems:
            elems.append(elem)
    return sorted(elems)


def nuclides_of_element(element: str) -> list[str]:
    """返回指定元素下所有核素（有序）。"""
    return sorted(n for n in _nuc_index if n.split('-')[0].strip() == element)


# =====================================================================
#   插值函数
# =====================================================================
def _linear_interp(x, xp, yp):
    """线性插值: x 在 xp 中查找, 返回插值 y; xp 需递增排序"""
    x = np.asarray(x, dtype=float); xp = np.asarray(xp, dtype=float); yp = np.asarray(yp, dtype=float)
    if x <= xp[0]:
        return float(yp[0])
    if x >= xp[-1]:
        return float(yp[-1])
    idx = np.searchsorted(xp, x)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = yp[idx - 1], yp[idx]
    if x1 == x0:
        return float(y0)
    return float(y0 + (x - x0) / (x1 - x0) * (y1 - y0))


def _loglog_interp(x, xp, yp):
    """双对数插值 (粒径 AMAD): 对 x 和 xp 取 log10 后线性插值"""
    xp = np.asarray(xp, dtype=float); yp = np.asarray(yp, dtype=float)
    if x <= xp[0]:
        return float(yp[0])
    if x >= xp[-1]:
        return float(yp[-1])
    lx = np.log10(x); lp = np.log10(xp)
    idx = np.searchsorted(lp, lx)
    lx0, lx1 = lp[idx - 1], lp[idx]
    y0, y1 = yp[idx - 1], yp[idx]
    if lx1 == lx0:
        return float(y0)
    return float(10 ** (np.log10(y0) + (lx - lx0) / (lx1 - lx0) * (np.log10(y1) - np.log10(y0))))


def lookup_z(radionuclide: str, route: str, material: str, sample_type: str,
             t: float, ps_um: float = None) -> float | None:
    """
    查 z(t) 值（懒加载版本）

    参数:
        radionuclide, route, material: 元数据筛选
        sample_type: 样本列名 (whole_body / urine_24h / faeces_24h / ...)
        t: 目标时间 (天)，支持任意小数精度
        ps_um: 自定义粒径 (μm), 仅 Inhalation 且非精确匹配 AMAD 时使用
    """
    # 按需加载该核素数据
    zdata = get_nuclide_df(radionuclide)

    # 1) 按元数据筛选
    sub = zdata[
        (zdata['radionuclide'] == radionuclide) &
        (zdata['route_of_intake'] == route) &
        (zdata['material'] == material)
    ].copy()

    if sub.empty:
        print(f'[z] 未找到匹配: {radionuclide}/{route}/{material}')
        return None

    # 2) 判断是否需要 AMAD 插值
    amad_vals = sorted(sub['amad_um'].dropna().unique())

    if len(amad_vals) == 0:
        # 无 AMAD 维度（Ingestion / Injection）: 直接时间线性插值
        times = sub['time_days'].values
        zvals = sub[sample_type].values
        mask = ~np.isnan(zvals)
        if mask.sum() == 0:
            return None
        return _linear_interp(t, times[mask], zvals[mask])

    # 3) 有 AMAD 维度 (Inhalation)
    if ps_um is not None and ps_um not in amad_vals:
        # 双对数插值: 对每个 time_days 在两相邻 AMAD 间插 z
        amad_arr = np.array(amad_vals)
        if ps_um <= amad_arr[0]:
            sub = sub[sub['amad_um'] == amad_arr[0]]
        elif ps_um >= amad_arr[-1]:
            sub = sub[sub['amad_um'] == amad_arr[-1]]
        else:
            idx = np.searchsorted(amad_arr, ps_um)
            lo_amad, hi_amad = amad_arr[idx - 1], amad_arr[idx]
            lo_sub = sub[sub['amad_um'] == lo_amad].set_index('time_days')[sample_type]
            hi_sub = sub[sub['amad_um'] == hi_amad].set_index('time_days')[sample_type]
            # 合并时间点
            common_times = lo_sub.index.intersection(hi_sub.index)
            if len(common_times) < 2:
                # 回退到最近 AMAD
                sub = sub[sub['amad_um'] == lo_amad]
            else:
                # 对每个公共时间点做对数回归插值 z_interp
                interp_z = {}
                for ti in common_times:
                    z_lo = lo_sub.loc[ti]; z_hi = hi_sub.loc[ti]
                    if pd.notna(z_lo) and pd.notna(z_hi):
                        interp_z[ti] = 10 ** (np.log10(z_lo) + (np.log10(ps_um) - np.log10(lo_amad)) /
                                              (np.log10(hi_amad) - np.log10(lo_amad)) *
                                              (np.log10(z_hi) - np.log10(z_lo)))
                if len(interp_z) < 2:
                    sub = sub[sub['amad_um'] == lo_amad]
                else:
                    times = np.array(sorted(interp_z.keys()))
                    zvals = np.array([interp_z[ti] for ti in times])
                    return _linear_interp(t, times, zvals)

    # 4) 精确 AMAD 匹配: 直接时间线性插值
    if ps_um is not None:
        sub = sub[sub['amad_um'] == ps_um]
    times = sub['time_days'].values
    zvals = sub[sample_type].values
    mask = ~np.isnan(zvals)
    if mask.sum() == 0:
        return None
    return _linear_interp(t, times[mask], zvals[mask])


# =====================================================================
#   核心计算
# =====================================================================
def calc_effective_dose(M: float, z: float) -> float:
    """E(50) = M × z(t)"""
    return M * z


# =====================================================================
#   ResultTable —— 带"删除"按钮的结果表格
# =====================================================================
class ResultTable(QTableWidget):
    row_deleted = pyqtSignal(int)

    HEADERS = ['ID', '核素', '摄入途径', '物质/fA', 'AMAD(μm)',
               '样本类型', '时间(d)', 'z(t)(Sv/Bq)', '测量值M', '有效剂量E(Sv)', '操作']

    def __init__(self, parent=None):
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.setColumnWidth(0, 40)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setDefaultSectionSize(28)
        self._items: list = []

    def add_item(self, item: dict):
        self._items.append(item)
        row = self.rowCount()
        self.insertRow(row)
        vals = [
            str(item['id']),
            item['nuclide'],
            item.get('route', '—'),
            item.get('material', '—'),
            f"{item['ps_um']:.3f}" if item.get('ps_um') is not None else '—',
            item.get('sample_type', '—'),
            f"{item['t']:.6f}",
            f"{item['z']:.3e}",
            f"{item['M']:.3e}",
            f"{item['E']:.3e}",
        ]
        for col, v in enumerate(vals):
            cell = QTableWidgetItem(v)
            cell.setTextAlignment(Qt.AlignCenter)
            if col == 0:
                cell.setFont(QFont('monospace', 8))
            self.setItem(row, col, cell)
        # 删除按钮
        btn = QPushButton(' 删除 ')
        btn.setFixedSize(50, 22)
        btn.setStyleSheet('QPushButton{color:#c0392b; font-weight:bold; font-size:10px;} '
                          'QPushButton:hover{background:#e74c3c; color:white;}')
        btn.clicked.connect(lambda _, r=row: self._delete_row(r))
        self.setCellWidget(row, len(self.HEADERS) - 1, btn)
        self.scrollToBottom()

    def _delete_row(self, row: int):
        del self._items[row]
        self.removeRow(row)
        # 重编号
        for i in range(self.rowCount()):
            self.item(i, 0).setText(str(i + 1))
            btn = self.cellWidget(i, len(self.HEADERS) - 1)
            if btn:
                btn.clicked.disconnect()
                btn.clicked.connect(lambda _, r=i: self._delete_row(r))
        self.row_deleted.emit(row)

    def get_items(self):
        return self._items

    def clear_all(self):
        self._items.clear()
        self.setRowCount(0)

    def total_dose(self) -> float:
        return sum(it['E'] for it in self._items)


# =====================================================================
#   SummaryPanel —— 汇总 + 导出
# =====================================================================
class SummaryPanel(QWidget):
    export_requested = pyqtSignal()

    def __init__(self, table: ResultTable, parent=None):
        super().__init__(parent)
        self._table = table
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)

        self.lbl_total = QLabel('总有效剂量: — Sv')
        self.lbl_total.setStyleSheet('font-size:14px; font-weight:bold; color:#2c3e50;')
        layout.addWidget(self.lbl_total)

        layout.addStretch()

        btn_export = QPushButton('导出报告 (CSV)')
        btn_export.setStyleSheet(
            'QPushButton{padding:6px 18px; font-size:12px; '
            'background:#27ae60; color:white; border:none; border-radius:4px;} '
            'QPushButton:hover{background:#2ecc71;}')
        btn_export.clicked.connect(self.export_requested.emit)
        layout.addWidget(btn_export)

        btn_export_xlsx = QPushButton('导出 Excel')
        btn_export_xlsx.setStyleSheet(
            'QPushButton{padding:6px 18px; font-size:12px; '
            'background:#2980b9; color:white; border:none; border-radius:4px;} '
            'QPushButton:hover{background:#3498db;}')
        btn_export_xlsx.clicked.connect(lambda: self.export_requested.emit())
        layout.addWidget(btn_export_xlsx)

        btn_clear = QPushButton('清空表格')
        btn_clear.setStyleSheet(
            'QPushButton{padding:6px 12px; font-size:12px; '
            'background:#e74c3c; color:white; border:none; border-radius:4px;} '
            'QPushButton:hover{background:#c0392b;}')
        btn_clear.clicked.connect(self._table.clear_all)
        btn_clear.clicked.connect(lambda: self.lbl_total.setText('总有效剂量: — Sv'))
        layout.addWidget(btn_clear)

        self._table.row_deleted.connect(self._refresh_total)
        self._table.model().rowsInserted.connect(self._refresh_total)

    def _refresh_total(self):
        total = self._table.total_dose()
        self.lbl_total.setText(f'总有效剂量: {total:.3e} Sv')


# =====================================================================
#   ParamPanel —— 左侧参数面板
# =====================================================================
class ParamPanel(QWidget):
    item_added = pyqtSignal(dict)

    PS_PRESETS = ['0.001', '0.003', '0.01', '0.03', '0.1', '0.3',
                  '1.0', '3.0', '5.0', '10.0', '20.0', '自定义']

    def __init__(self, parent=None):
        super().__init__(parent)
        self._id_counter = 0
        self._items: list = []
        self._setup_ui()
        self._connect_signals()

    # -----------------------------------------------------------------
    #   UI
    # -----------------------------------------------------------------
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ═══ ① 监测类型 ═══
        g_type = QGroupBox('① 监测类型')
        fl0 = QFormLayout(g_type)
        self.rb_routine = QRadioButton('常规监测')
        self.rb_special = QRadioButton('应急监测')
        self.rb_routine.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_routine, 0); bg.addButton(self.rb_special, 1)
        bg.buttonClicked.connect(lambda btn: self._on_type_changed())
        fl0.addRow(self.rb_routine)
        fl0.addRow(self.rb_special)
        main_layout.addWidget(g_type)

        # ═══ ② 核素 & 摄入途径 ═══
        # 元素 + 核素 放在同一行（紧凑布局）
        g_nuc = QGroupBox('② 核素 & 摄入途径')
        fl1 = QFormLayout(g_nuc)
        fl1.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # 元素+核素 同行
        nuc_row = QHBoxLayout()
        nuc_row.setSpacing(4)
        self.cb_element = QComboBox()
        self.cb_element.setMinimumWidth(68)
        self.cb_element.setMaximumWidth(80)
        self.cb_element.setToolTip('选择元素')
        nuc_row.addWidget(self.cb_element)

        nuc_sep = QLabel('—')
        nuc_sep.setAlignment(Qt.AlignCenter)
        nuc_sep.setFixedWidth(14)
        nuc_row.addWidget(nuc_sep)

        self.cb_nuclide = QComboBox()
        self.cb_nuclide.setMinimumWidth(80)
        self.cb_nuclide.setToolTip('选择核素')
        nuc_row.addWidget(self.cb_nuclide, 1)
        fl1.addRow('核素:', nuc_row)

        self.cb_route = QComboBox()
        self.cb_route.setMinimumWidth(160)
        fl1.addRow('摄入途径:', self.cb_route)
        main_layout.addWidget(g_nuc)

        # ═══ ③ 物质 / fA ═══
        g_mat = QGroupBox('③ 物质类型 (fA)')
        fl2 = QFormLayout(g_mat)
        self.cb_material = QComboBox()
        self.cb_material.setMinimumWidth(180)
        fl2.addRow('物质:', self.cb_material)
        main_layout.addWidget(g_mat)

        # ═══ ④ 样本类型 ═══
        g_sample = QGroupBox('④ 样本类型')
        fl3 = QFormLayout(g_sample)
        self.cb_sample = QComboBox()
        self.cb_sample.setMinimumWidth(180)
        fl3.addRow('样本:', self.cb_sample)
        main_layout.addWidget(g_sample)

        # ═══ ⑤ AMAD 粒径 (仅 Inhalation) ═══
        self.g_ps = QGroupBox('⑤ 粒径 AMAD (μm) — 仅吸入途径')
        fl4 = QFormLayout(self.g_ps)
        self.cb_ps_preset = QComboBox()
        self.cb_ps_preset.addItems(self.PS_PRESETS)
        fl4.addRow('预设:', self.cb_ps_preset)
        self.spin_ps_custom = QDoubleSpinBox()
        self.spin_ps_custom.setRange(0.001, 100.0)
        self.spin_ps_custom.setDecimals(3)
        self.spin_ps_custom.setValue(5.0)
        self.spin_ps_custom.setSingleStep(0.1)
        self.spin_ps_custom.setVisible(False)
        fl4.addRow('自定义(μm):', self.spin_ps_custom)
        main_layout.addWidget(self.g_ps)

        # ═══ ⑥ 时间 ═══
        # 支持6位小数精度
        g_time = QGroupBox('⑥ 时间参数')
        fl5 = QFormLayout(g_time)
        self.lbl_time = QLabel('监测周期 T (天):')

        self.spin_T = QDoubleSpinBox()
        self.spin_T.setRange(0.000001, 36500.0)
        self.spin_T.setDecimals(6)           # ← 支持 6 位小数
        self.spin_T.setValue(180.0)
        self.spin_T.setSingleStep(1.0)
        self.spin_T.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        self.spin_T.setToolTip('支持输入最多6位小数，例如 0.500000 天')
        fl5.addRow(self.lbl_time, self.spin_T)

        self.lbl_t_info = QLabel('计算用 t = T/2 = 90.000000 天')
        self.lbl_t_info.setStyleSheet('color:#7f8c8d; font-size:11px;')
        fl5.addRow(self.lbl_t_info)
        main_layout.addWidget(g_time)

        # ═══ ⑦ 测量值 M ═══
        g_M = QGroupBox('⑦ 测量值 M')
        fl6 = QFormLayout(g_M)
        self.spin_M = QDoubleSpinBox()
        self.spin_M.setRange(1e-20, 1e20)
        self.spin_M.setDecimals(6)
        self.spin_M.setValue(1.0)
        self.spin_M.setSingleStep(0.1)
        self.spin_M.setPrefix('M = ')
        self.spin_M.setSuffix(' Bq')
        fl6.addRow(self.spin_M)
        main_layout.addWidget(g_M)

        # ═══ ⑧ 当前 z(t) 显示 ═══
        g_z = QGroupBox('⑧ 剂量转换系数 z(t)')
        fl7 = QFormLayout(g_z)
        self.lbl_z = QLabel('—')
        self.lbl_z.setStyleSheet('font-size:14px; font-weight:bold; color:#2980b9;')
        fl7.addRow('z(t):', self.lbl_z)
        main_layout.addWidget(g_z)

        # ═══ ⑨ 计算 & 添加 ═══
        g_calc = QGroupBox('⑨ 操作')
        fl8 = QFormLayout(g_calc)
        btn_calc = QPushButton('▶  计算并添加到结果表')
        btn_calc.setStyleSheet(
            'QPushButton{padding:8px 20px; font-size:13px; font-weight:bold; '
            'background:#2980b9; color:white; border:none; border-radius:4px;} '
            'QPushButton:hover{background:#3498db;}')
        btn_calc.clicked.connect(self._on_calc)
        fl8.addRow(btn_calc)
        main_layout.addWidget(g_calc)

        main_layout.addStretch()

    # -----------------------------------------------------------------
    #   信号连接
    # -----------------------------------------------------------------
    def _connect_signals(self):
        self.cb_element.currentTextChanged.connect(self._on_element_changed)
        self.cb_nuclide.currentTextChanged.connect(self._on_nuclide_changed)
        self.cb_route.currentTextChanged.connect(self._on_route_changed)
        self.cb_material.currentTextChanged.connect(self._on_material_changed)
        self.cb_ps_preset.currentTextChanged.connect(self._on_ps_preset_changed)
        self.spin_T.valueChanged.connect(self._on_T_changed)
        self.spin_ps_custom.valueChanged.connect(lambda: self._refresh_z())

    # -----------------------------------------------------------------
    #   数据填充
    # -----------------------------------------------------------------
    @staticmethod
    def _fill_cb(cb: QComboBox, items: list, block=False):
        """填充下拉框。
        始终在填充期间阻断信号（避免 clear/addItems 引起的多余级联），
        填充完成后始终解封信号（确保用户操作可触发槽函数）。
        """
        cb.blockSignals(True)
        cb.clear()
        cb.addItems([str(v) for v in items])
        if items:
            cb.setCurrentIndex(0)
        cb.blockSignals(False)

    def init_data(self):
        """初始化元素列表（核素懒加载，不预读所有数据）"""
        elems = available_elements()
        self._fill_cb(self.cb_element, elems)
        self.cb_element.blockSignals(True)
        self.cb_element.setCurrentIndex(0)
        self.cb_element.blockSignals(False)
        self._on_element_changed()

    # -----------------------------------------------------------------
    #   槽函数
    # -----------------------------------------------------------------
    def _on_type_changed(self):
        """监测类型切换"""
        nuc = self.cb_nuclide.currentText()
        if not nuc:
            return
        self._on_nuclide_changed()

    def _on_element_changed(self):
        """元素变化 → 刷新核素列表"""
        elem = self.cb_element.currentText()
        if not elem:
            return
        nucs = nuclides_of_element(elem)
        self._fill_cb(self.cb_nuclide, nucs)
        self.cb_nuclide.blockSignals(True)
        self.cb_nuclide.setCurrentIndex(0)
        self.cb_nuclide.blockSignals(False)
        self._on_nuclide_changed()

    def _on_nuclide_changed(self):
        """核素变化 → 刷新途径、时间标签（触发懒加载）"""
        nuc = self.cb_nuclide.currentText()
        if not nuc:
            return

        is_routine = self.rb_routine.isChecked()
        # 懒加载该核素数据
        zdata = get_nuclide_df(nuc)
        sub = zdata[zdata['radionuclide'] == nuc]
        if sub.empty:
            self._fill_cb(self.cb_route, ['无数据'])
            return

        all_routes = sorted(sub['route_of_intake'].unique())

        if is_routine:
            # 常规监测仅 Inhalation
            if 'Inhalation' in all_routes:
                routes = ['Inhalation']
            else:
                routes = all_routes  # 退化
            # 时间: 周期 T
            self.lbl_time.setText('监测周期 T (天):')
            self.spin_T.setVisible(True)
            self.lbl_t_info.setVisible(True)
            self._on_T_changed()
        else:
            # 应急监测: 三种途径都可选
            routes = all_routes
            # 时间: 直接输入 t
            self.lbl_time.setText('摄入后时间 t (天):')
            self.spin_T.setVisible(True)
            self.lbl_t_info.setText('直接使用输入时间 t')
            self._on_T_changed()

        self._fill_cb(self.cb_route, routes)
        self.cb_route.blockSignals(True)
        if 'Inhalation' in routes:
            self.cb_route.setCurrentText('Inhalation')
        else:
            self.cb_route.setCurrentIndex(0)
        self.cb_route.blockSignals(False)
        self._on_route_changed()

    def _on_route_changed(self):
        """途径变化 → 刷新物质列表"""
        nuc = self.cb_nuclide.currentText()
        route = self.cb_route.currentText()
        if not nuc or not route:
            return

        zdata = get_nuclide_df(nuc)
        sub = zdata[(zdata['radionuclide'] == nuc) &
                    (zdata['route_of_intake'] == route)]
        if sub.empty:
            self._fill_cb(self.cb_material, ['无数据'])
            return

        mat_labels = sub['material'].drop_duplicates().tolist()

        self._fill_cb(self.cb_material, mat_labels)
        self.cb_material.blockSignals(True)
        self.cb_material.setCurrentIndex(0)
        self.cb_material.blockSignals(False)

        # AMAD: 仅 Inhalation 显示
        if route == 'Inhalation':
            self.g_ps.setVisible(True)
            amads = sorted(sub['amad_um'].dropna().unique())
            self.cb_ps_preset.blockSignals(True)
            self.cb_ps_preset.clear()
            preset_items = [str(a) for a in amads] + ['自定义']
            self.cb_ps_preset.addItems(preset_items)
            if len(amads) > 0:
                self.cb_ps_preset.setCurrentIndex(0)
                self._on_ps_preset_changed()
            self.cb_ps_preset.blockSignals(False)
        else:
            self.g_ps.setVisible(False)

        # 样本类型：仅列出当前核素+途径实际有非空数据的列
        all_sample_cols = [c for c in zdata.columns if c not in _META_COLS]
        sample_cols = [c for c in all_sample_cols if sub[c].notna().any()] if not sub.empty else all_sample_cols
        self._fill_cb(self.cb_sample, sample_cols)
        self.cb_sample.blockSignals(True)
        self.cb_sample.setCurrentIndex(0)
        self.cb_sample.blockSignals(False)

        self._on_material_changed()

    def _on_material_changed(self):
        self._refresh_z()

    def _on_ps_preset_changed(self):
        """粒径预设变化"""
        if self.cb_ps_preset.currentText() == '自定义':
            self.spin_ps_custom.setVisible(True)
        else:
            self.spin_ps_custom.setVisible(False)
        self._refresh_z()

    def _on_T_changed(self):
        is_routine = self.rb_routine.isChecked()
        val = self.spin_T.value()
        if is_routine:
            t = val / 2.0
            self.lbl_t_info.setText(f'计算用 t = T/2 = {t:.6f} 天')
        else:
            t = val
            self.lbl_t_info.setText(f'直接使用 t = {t:.6f} 天')
        self._refresh_z()

    # -----------------------------------------------------------------
    #   z(t) 刷新
    # -----------------------------------------------------------------
    def _refresh_z(self):
        """刷新 z(t) 显示"""
        z = self._compute_z()
        if z is not None:
            self.lbl_z.setText(f'{z:.3e} Sv/Bq')
        else:
            self.lbl_z.setText('— (无数据)')

    def _compute_z(self) -> float | None:
        nuc = self.cb_nuclide.currentText()
        route = self.cb_route.currentText()
        mat_text = self.cb_material.currentText()
        sample = self.cb_sample.currentText()

        if not all([nuc, route, mat_text, sample]):
            return None

        # material 列已是完整名称（如 "Soluble forms, fA=2E-2"），直接使用
        mat_name = mat_text

        # 时间 t（6位小数精度）
        is_routine = self.rb_routine.isChecked()
        if is_routine:
            t = self.spin_T.value() / 2.0
        else:
            t = self.spin_T.value()

        # 粒径
        ps_um = None
        if route == 'Inhalation':
            ps_text = self.cb_ps_preset.currentText()
            if ps_text == '自定义':
                ps_um = self.spin_ps_custom.value()
            else:
                try:
                    ps_um = float(ps_text)
                except ValueError:
                    ps_um = 5.0

        return lookup_z(nuc, route, mat_name, sample, t, ps_um)

    def _get_time(self) -> float:
        is_routine = self.rb_routine.isChecked()
        if is_routine:
            return self.spin_T.value() / 2.0
        else:
            return self.spin_T.value()

    def _get_ps_um(self) -> float | None:
        route = self.cb_route.currentText()
        if route != 'Inhalation':
            return None
        ps_text = self.cb_ps_preset.currentText()
        if ps_text == '自定义':
            return self.spin_ps_custom.value()
        try:
            return float(ps_text)
        except ValueError:
            return 5.0

    # -----------------------------------------------------------------
    #   计算 & 发射信号
    # -----------------------------------------------------------------
    def _on_calc(self):
        nuc = self.cb_nuclide.currentText()
        route = self.cb_route.currentText()
        mat_text = self.cb_material.currentText()
        sample = self.cb_sample.currentText()
        M = self.spin_M.value()

        if not all([nuc, route, mat_text, sample]):
            QMessageBox.warning(self, '参数不全', '请完成核素/途径/物质/样本类型选择')
            return

        mat_name = mat_text   # material 列已是完整名称，直接使用
        t = self._get_time()
        ps_um = self._get_ps_um()

        z = lookup_z(nuc, route, mat_name, sample, t, ps_um)
        if z is None or z == 0:
            QMessageBox.warning(self, '数据缺失',
                f'无法计算 z(t):\n核素={nuc} 途径={route} 物质={mat_name}\n'
                f'样本={sample} t={t:.6f}d')
            return

        E = calc_effective_dose(M, z)
        self._id_counter += 1

        item = {
            'id': self._id_counter,
            'nuclide': nuc,
            'route': route,
            'material': mat_text,
            'fA': '',  # material 列中已包含 fA 信息
            'ps_um': ps_um,
            'sample_type': sample,
            't': t,
            'z': z,
            'M': M,
            'E': E,
        }
        self._items.append(item)
        self.item_added.emit(item)


# =====================================================================
#   MainWindow
# =====================================================================
class MonitoringDoseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('监测法内照射剂量计算系统 v2.4 — GB/T 16148-2009 z(t) 函数法')
        self.resize(1310, 790)
        self.setMinimumSize(900, 600)
        self._setup_ui()
        self._connect_signals()
        self._init()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Horizontal)

        # ── 左侧: 参数面板 ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        scroll.setFrameShape(QFrame.NoFrame)

        self.param_panel = ParamPanel()
        scroll.setWidget(self.param_panel)
        splitter.addWidget(scroll)

        # ── 右侧: 表格 + 汇总 ──
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)

        self.result_table = ResultTable()
        rl.addWidget(self.result_table, 1)

        self.summary_panel = SummaryPanel(self.result_table)
        rl.addWidget(self.summary_panel)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([450, 800])

        main_layout.addWidget(splitter)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f'就绪 | 数据: z_data/ 懒加载模式 (GB/T 16148-2009) | 可用核素: {len(_nuc_index)}')

    def _connect_signals(self):
        self.param_panel.item_added.connect(self._on_item_added)
        self.summary_panel.export_requested.connect(self._export)

    def _init(self):
        scan_z_data()
        self.param_panel.init_data()

    def _on_item_added(self, item: dict):
        self.result_table.add_item(item)
        self.summary_panel._refresh_total()
        self.status_bar.showMessage(
            f'已添加: {item["nuclide"]} | z({item["t"]:.6f}d)={item["z"]:.3e} Sv/Bq | '
            f'E={item["E"]:.3e} Sv')

    def _export(self):
        items = self.result_table.get_items()
        if not items:
            QMessageBox.information(self, '提示', '结果表中无数据')
            return
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path, _ = QFileDialog.getSaveFileName(
            self, '导出报告', f'dose_report_{ts}.csv',
            'CSV (*.csv);;Excel (*.xlsx)')
        if not save_path:
            return
        try:
            df = pd.DataFrame(items)
            cols_order = ['id', 'nuclide', 'route', 'material', 'fA',
                          'ps_um', 'sample_type', 't', 'z', 'M', 'E']
            df = df[[c for c in cols_order if c in df.columns]]
            if save_path.endswith('.xlsx'):
                df.to_excel(save_path, index=False)
            else:
                df.to_csv(save_path, index=False, encoding='utf-8-sig')
            self.status_bar.showMessage(f'报告已导出: {save_path}')
        except Exception as e:
            QMessageBox.warning(self, '导出失败', str(e))


# =====================================================================
#   入口
# =====================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont('Microsoft YaHei', 9))
    win = MonitoringDoseApp()
    win.show()
    sys.exit(app.exec_())

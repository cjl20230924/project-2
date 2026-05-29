"""
监测法内照射剂量计算系统 v1.1 (桌面版)
基于 PyQt5 实现，依据 GB/T 16148 及 ICRP Publication 54/78/130

计算方法：
  E = I × e(AMAD, route)
  I = M / m(T)
  其中
    M  — 生物样品测量值（Bq 或 Bq/d）
    m  — 剂量当量系数（由 m_data/ 数据文件提供）
    e  — 剂量系数（Sv/Bq），按粒径双对数插值（由 processed_nuclide_files/ 提供）
    AMAD — 可手动输入或从下拉列表选择（μm）

操作步骤：
  ① 选核素  ② 设粒径  ③ 选监测参数（m 值唯一确定）  ④ 输入 M  ⑤ 计算 / 添加
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
#   路径辅助
# =====================================================================
def resource_path(rel):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, rel)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', rel)

DATA_DIR   = Path(resource_path('processed_nuclide_files'))
M_DATA_DIR = Path(resource_path('m_data'))

# =====================================================================
#   数据加载 —— m_data
# =====================================================================
_ROUTINE_DATA: pd.DataFrame = pd.DataFrame()
_SPECIAL_DATA: pd.DataFrame = pd.DataFrame()

_COL_ALIASES = {
    'radionuclide':      ['radionuclide', '核素', 'nuclide'],
    'aerosol_type':      ['aerosols type', 'aerosol_type', '气溶胶类型'],
    'monitoring_method': ['monitoring method', 'monitoring_method', '监测方法'],
    'time_days':         ['period (d)', 'time (d)', '周期', '时间', 'period', 'time'],
    'm_value':           ["m(t/2)", "m(t)", "m值", "m_value", "m"],
    'intake_route':      ['route of intake', 'intake_route', '摄入途径', 'route'],
    'fA':                ['fa'],
}

def _norm_col(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for std, aliases in _COL_ALIASES.items():
        for c in df.columns:
            if c.strip().lower() in aliases:
                rename[c] = std
                break
    df = df.rename(columns=rename)
    if 'aerosol_type' in df.columns:
        df['aerosol_type'] = df['aerosol_type'].replace('Gaseous', 'Unspecified')
    for num_col in ['time_days', 'm_value', 'fA']:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
    return df

def _parse_m_files(paths: list) -> pd.DataFrame:
    frames = []
    for f in paths:
        try:
            if f.suffix == '.parquet':
                df = pd.read_parquet(f)
            elif f.suffix in ('.xlsx', '.xls'):
                df = pd.read_excel(f)
            else:
                df = pd.read_csv(f)
            frames.append(_norm_col(df))
        except Exception as e:
            print(f'[警告] 跳过 {f.name}: {e}')
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def scan_monitoring():
    global _ROUTINE_DATA, _SPECIAL_DATA
    if not M_DATA_DIR.exists():
        return
    routine_files = [f for ext in ('.csv', '.xlsx', '.xls', '.parquet')
                     for f in M_DATA_DIR.glob(f'*{ext}') if 'routine' in f.stem.lower()]
    special_files = [f for ext in ('.csv', '.xlsx', '.xls', '.parquet')
                     for f in M_DATA_DIR.glob(f'*{ext}') if 'special' in f.stem.lower()]
    _ROUTINE_DATA = _parse_m_files(routine_files)
    _SPECIAL_DATA = _parse_m_files(special_files)

# =====================================================================
#   数据加载 —— processed_nuclide_files（剂量系数）
# =====================================================================
_nuclide_cache: dict = {}
_nuclide_files: dict = {}
_element_map:   dict = {}

def scan_nuclides():
    global _nuclide_files, _element_map
    if not DATA_DIR.exists():
        print(f'[警告] 剂量系数目录不存在: {DATA_DIR}')
        return
    for f in list(DATA_DIR.glob('*.parquet')) + list(DATA_DIR.glob('*.xlsx')):
        try:
            df = pd.read_parquet(f) if f.suffix == '.parquet' else pd.read_excel(f)
            cm = {}
            for c in df.columns:
                cl = c.lower().strip()
                if cl == 'element':              cm[c] = 'element'
                elif 'radionuclide' in cl:       cm[c] = 'radionuclide'
                elif 'route' in cl:              cm[c] = 'route_of_intake'
                elif 'aerosol' in cl:            cm[c] = 'aerosol_type'
                elif 'compound' in cl:           cm[c] = 'compound'
                elif 'particle' in cl:           cm[c] = 'particle_size'
                elif 'dose' in cl and 'coef' in cl: cm[c] = 'dose_coefficient'
                elif cl == 'fa':                 cm[c] = 'fA'
            df = df.rename(columns=cm)
            if 'particle_size' in df.columns:
                df['particle_size'] = pd.to_numeric(
                    df['particle_size'].astype(str)
                    .str.replace(r'\s*[µμ]m\s*|\s*micron\s*', '', regex=True)
                    .replace(['', 'nan', 'NaN', 'None'], np.nan), errors='coerce')
            if 'dose_coefficient' in df.columns:
                df['dose_coefficient'] = pd.to_numeric(df['dose_coefficient'], errors='coerce')
            if 'aerosol_type' in df.columns:
                df['aerosol_type'] = df['aerosol_type'].replace('Gaseous', 'Unspecified')
            if 'element' in df.columns and 'radionuclide' in df.columns:
                elem = str(df['element'].iloc[0])
                nuc  = str(df['radionuclide'].iloc[0]).replace('_', '-')
                _nuclide_files[nuc] = (f, df)
                _element_map.setdefault(elem, [])
                if nuc not in _element_map[elem]:
                    _element_map[elem].append(nuc)
        except Exception as e:
            print(f'[警告] 加载 {f.name} 失败: {e}')

def _norm_nuc(n):
    return n.strip().replace('_', '-')

def get_nuclide_df(nuclide: str, route: str = 'Inhalation') -> pd.DataFrame | None:
    """根据核素名和摄入途径获取剂量系数 DataFrame"""
    nk  = _norm_nuc(nuclide)
    key = (nk, route)
    if key in _nuclide_cache:
        return _nuclide_cache[key]
    if nk not in _nuclide_files:
        return None
    _, df = _nuclide_files[nk]
    if route and 'route_of_intake' in df.columns:
        sub = df[df['route_of_intake'] == route].copy()
        result = sub if not sub.empty else df.copy()   # fallback：途径无数据时用全部
    else:
        result = df.copy()
    _nuclide_cache[key] = result
    return result

def interp_dose_coeff(df: pd.DataFrame, aerosol: str, ps_um: float) -> float | None:
    """按粒径双对数插值剂量系数 e (Sv/Bq)"""
    if df is None or df.empty:
        return None
    if 'aerosol_type' not in df.columns:
        return None
    sub = df[df['aerosol_type'] == aerosol].copy()
    if sub.empty:
        return None
    sub = sub.dropna(subset=['particle_size', 'dose_coefficient']).sort_values('particle_size')
    if sub.empty:
        r0 = df[df['aerosol_type'] == aerosol]
        return float(r0['dose_coefficient'].iloc[0]) if not r0.empty else None
    ps = sub['particle_size'].values
    dc = sub['dose_coefficient'].values
    if len(ps) == 1:
        return float(dc[0])
    if ps_um <= ps.min():
        return float(dc[0])
    if ps_um >= ps.max():
        return float(dc[-1])
    lx = np.log(ps_um)
    lp = np.log(ps)
    for i in range(len(lp) - 1):
        if lp[i] <= lx <= lp[i + 1]:
            y1, y2 = dc[i], dc[i + 1]
            x1, x2 = lp[i], lp[i + 1]
            return float(y1 + (lx - x1) / (x2 - x1) * (y2 - y1)) if x2 != x1 else float(y1)
    return None

# =====================================================================
#   GB/T 16148 核心计算
# =====================================================================
def calc_intake(M: float, m: float) -> float:
    return M / m if m != 0 else float('nan')

def calc_dose(I: float, e: float) -> float:
    return I * e

# =====================================================================
#   ResultTable —— 带"删除"按钮的结果表格
# =====================================================================
class ResultTable(QTableWidget):
    row_deleted = pyqtSignal(int)

    HEADERS = ['ID', '核素', '摄入途径', '气溶胶类型', '粒径(μm)', '监测方法',
               '时间(d)', 'm值', '测量值M', '摄入量I(Bq)',
               '剂量系数e(Sv/Bq)', '有效剂量E(Sv)', '操作']

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
            item.get('intake_route', '—'),
            item.get('aerosol_type', '—'),
            f"{item['particle_size']:.3f}" if item.get('particle_size') is not None else '—',
            item.get('monitoring_method', '—'),
            str(item.get('time_days', '—')),
            f"{item['m_value']:.3e}",
            f"{item['M']:.3e}",
            f"{item['I']:.3e}",
            f"{item['e_value']:.3e}",
            f"{item['E']:.3e}",
        ]
        for col, v in enumerate(vals):
            cell = QTableWidgetItem(v)
            cell.setTextAlignment(Qt.AlignCenter)
            if col == 11:
                cell.setBackground(QColor('#d5f5e3'))
            self.setItem(row, col, cell)
        del_btn = QPushButton('🗑 删除')
        del_btn.setFixedHeight(24)
        del_btn.setStyleSheet('QPushButton{color:#c0392b; font-size:11px; border:none;}')
        del_btn.clicked.connect(lambda _, iid=item['id']: self._on_delete(iid))
        self.setCellWidget(row, len(self.HEADERS) - 1, del_btn)

    def _on_delete(self, iid: int):
        for r in range(self.rowCount()):
            cell = self.item(r, 0)
            if cell and cell.text() == str(iid):
                self.removeRow(r)
                self._items = [i for i in self._items if i['id'] != iid]
                self.row_deleted.emit(iid)
                return

    def clear_all(self):
        self.setRowCount(0)
        self._items.clear()

    def total_dose(self) -> float:
        return sum(i['E'] for i in self._items)

    def export_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'ID':             i['id'],
            '核素':            i['nuclide'],
            '摄入途径':         i.get('intake_route', ''),
            '气溶胶类型': i.get('aerosol_type', ''),
            '粒径(μm)':        i.get('particle_size', ''),
            '监测方法':         i.get('monitoring_method', ''),
            '时间(d)':         i.get('time_days', ''),
            'm值':             i['m_value'],
            '测量值M':          i['M'],
            '摄入量I(Bq)':      i['I'],
            '剂量系数e(Sv/Bq)': i['e_value'],
            '有效剂量E(Sv)':    i['E'],
        } for i in self._items])

# =====================================================================
#   ParamPanel —— 左侧参数面板（重构）
#   步骤：① 核素  ② 粒径 AMAD  ③ 监测参数→m值  ④ 剂量系数e  ⑤ 测量值M  ⑥ 计算
# =====================================================================
class ParamPanel(QWidget):
    item_ready = pyqtSignal(dict)

    def __init__(self, monitor_type: str = '常规监测', parent=None):
        super().__init__(parent)
        self._monitor_type = monitor_type
        self._next_id  = 1
        self._m_value  = None
        self._e_value  = None
        self._I_value  = None
        self._E_value  = None
        self._calc_result = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── ① 核素选择 ──────────────────────────────────
        g_nuc = QGroupBox('① 核素选择')
        fl = QFormLayout(g_nuc)
        fl.setLabelAlignment(Qt.AlignRight)
        self.cb_element = QComboBox()
        self.cb_nuclide = QComboBox()
        fl.addRow('元素：', self.cb_element)
        fl.addRow('核素：', self.cb_nuclide)
        root.addWidget(g_nuc)

        # ── ② 粒径 AMAD ──────────────────────────────────
        g_ps = QGroupBox('② 粒径 AMAD (μm)')
        fl_ps = QFormLayout(g_ps)
        fl_ps.setLabelAlignment(Qt.AlignRight)
        ps_row = QHBoxLayout()
        self.cb_ps_preset = QComboBox()
        PRESET_PS = ['0.001', '0.003', '0.01', '0.03', '0.1', '0.3',
                     '1.0', '3.0', '5.0', '10.0', '20.0', '自定义']
        self.cb_ps_preset.addItems(PRESET_PS)
        self.cb_ps_preset.setCurrentText('5.0')
        self.spin_ps_custom = QDoubleSpinBox()
        self.spin_ps_custom.setRange(0.001, 100.0)
        self.spin_ps_custom.setDecimals(3)
        self.spin_ps_custom.setValue(5.0)
        self.spin_ps_custom.setSuffix(' μm')
        self.spin_ps_custom.setVisible(False)
        self.spin_ps_custom.setMinimumWidth(110)
        ps_row.addWidget(self.cb_ps_preset)
        ps_row.addWidget(self.spin_ps_custom)
        ps_row.addStretch()
        fl_ps.addRow('粒径预设：', ps_row)
        root.addWidget(g_ps)

        # ── ③ 监测参数 → m 值 ────────────────────────────
        g_filt = QGroupBox('③ 监测参数（确定 m 值 / e 值气溶胶类型）')
        fl2 = QFormLayout(g_filt)
        fl2.setLabelAlignment(Qt.AlignRight)

        self.cb_intake  = QComboBox()   # 摄入途径（常规固定 Inhalation；应急三选一）
        self.cb_aerosol = QComboBox()   # 气溶胶类型
        self.cb_method  = QComboBox()   # 监测方法
        self.cb_time    = QComboBox()   # 时间/周期
        self.cb_fA      = QComboBox()   # fA（食入时）

        # 摄入途径行（常规监测时置灰并固定为 Inhalation）
        self.lbl_intake_row = QLabel('摄入途径：')
        fl2.addRow(self.lbl_intake_row, self.cb_intake)
        fl2.addRow('气溶胶类型：', self.cb_aerosol)
        fl2.addRow('监测方法：',   self.cb_method)
        fl2.addRow('时间 (d)：',   self.cb_time)
        fl2.addRow('fA：',         self.cb_fA)

        self.lbl_m = QLabel('—')
        self.lbl_m.setStyleSheet('color:#1a5276; font-weight:bold; font-size:12px;')
        fl2.addRow('m 值：', self.lbl_m)
        root.addWidget(g_filt)

        # ── ④ 剂量系数 e ─────────────────────────────────
        self.g_e = QGroupBox('④ 剂量系数 e（自动计算，气溶胶类型与③共用）')
        fl3 = QFormLayout(self.g_e)
        fl3.setLabelAlignment(Qt.AlignRight)

        self.lbl_e = QLabel('—')
        self.lbl_e.setStyleSheet('color:#1a5276; font-weight:bold; font-size:12px;')
        fl3.addRow('e (Sv/Bq)：', self.lbl_e)
        root.addWidget(self.g_e)

        # ── ⑤ 测量值 M ───────────────────────────────────
        g_M = QGroupBox('⑤ 测量值 M')
        fl_M = QFormLayout(g_M)
        fl_M.setLabelAlignment(Qt.AlignRight)
        self.spin_M = QDoubleSpinBox()
        self.spin_M.setRange(0, 1e18)
        self.spin_M.setDecimals(6)
        self.spin_M.setSingleStep(1.0)
        self.spin_M.setValue(1.0)
        self.spin_M.setSuffix('  Bq（或 Bq/d）')
        fl_M.addRow('M =', self.spin_M)
        root.addWidget(g_M)

        # ── ⑥ 计算结果 ───────────────────────────────────
        g_result = QGroupBox('⑥ 计算结果')
        fl_r = QFormLayout(g_result)
        fl_r.setLabelAlignment(Qt.AlignRight)
        self.lbl_I    = QLabel('I = —')
        self.lbl_I.setStyleSheet('color:#1a5276; font-size:12px;')
        self.lbl_dose = QLabel('E = —')
        self.lbl_dose.setStyleSheet('font-size:14px; font-weight:bold; color:#148f77;')
        fl_r.addRow('摄入量：', self.lbl_I)
        fl_r.addRow('有效剂量：', self.lbl_dose)
        root.addWidget(g_result)

        # ── 按钮 ─────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_calc = QPushButton('🔢 计算剂量')
        self.btn_calc.setStyleSheet(
            'background:#2980b9; color:white; font-weight:bold; padding:7px 20px; border-radius:4px;')
        self.btn_add = QPushButton('➕ 添加到列表')
        self.btn_add.setStyleSheet(
            'background:#27ae60; color:white; font-weight:bold; padding:7px 20px; border-radius:4px;')
        self.btn_add.setEnabled(False)
        btn_row.addWidget(self.btn_calc)
        btn_row.addWidget(self.btn_add)
        root.addLayout(btn_row)
        root.addStretch()

        # ── 信号连接 ─────────────────────────────────────
        self.cb_element.currentTextChanged.connect(self._on_element_changed)
        self.cb_nuclide.currentTextChanged.connect(self._on_nuclide_changed)
        self.cb_ps_preset.currentTextChanged.connect(self._on_ps_preset_changed)
        self.spin_ps_custom.valueChanged.connect(self._refresh_e)
        self.cb_intake.currentTextChanged.connect(self._on_intake_changed)
        self.cb_aerosol.currentTextChanged.connect(self._on_aerosol_changed)
        self.cb_method.currentTextChanged.connect(self._refresh_filter)
        self.cb_time.currentTextChanged.connect(self._refresh_filter)
        self.cb_fA.currentTextChanged.connect(self._refresh_filter)
        self.btn_calc.clicked.connect(self._on_calc)
        self.btn_add.clicked.connect(self._on_add)

    # ==================================================================
    #   公有接口
    # ==================================================================
    def populate(self):
        """扫描完成后由主窗口调用，填充元素列表"""
        elems = sorted(_element_map.keys())
        self.cb_element.blockSignals(True)
        self.cb_element.clear()
        self.cb_element.addItems(elems)
        self.cb_element.blockSignals(False)
        if elems:
            self.cb_element.setCurrentIndex(0)
            self._on_element_changed(elems[0])

    def set_monitor_type(self, mtype: str):
        self._monitor_type = mtype
        nuc = self.cb_nuclide.currentText()
        if nuc:
            self._on_nuclide_changed(nuc)

    # ==================================================================
    #   内部槽
    # ==================================================================
    def _on_element_changed(self, elem: str):
        self.cb_nuclide.blockSignals(True)
        self.cb_nuclide.clear()
        nucs = sorted(_element_map.get(elem, []))
        self.cb_nuclide.addItems(nucs)
        self.cb_nuclide.blockSignals(False)
        if nucs:
            self.cb_nuclide.setCurrentIndex(0)
            self._on_nuclide_changed(nucs[0])

    def _on_nuclide_changed(self, nuc: str):
        """核素变更：重新填充摄入途径下拉，触发后续过滤链"""
        if not nuc:
            return
        data = _ROUTINE_DATA if self._monitor_type == '常规监测' else _SPECIAL_DATA
        if data.empty:
            self._clear_filters()
            return

        df_n = data[data['radionuclide'].astype(str)
                    .str.replace('_', '-').str.strip() == nuc.strip()]

        # ── 摄入途径 ──
        if self._monitor_type == '应急监测' and 'intake_route' in df_n.columns:
            # 应急监测：从数据中读取实际存在的途径，并补全标准三途径顺序
            raw_routes = set(df_n['intake_route'].dropna().astype(str).str.strip().unique())
            preferred  = ['Inhalation', 'Injection', 'Ingestion']
            routes     = [r for r in preferred if r in raw_routes] + \
                         sorted(raw_routes - set(preferred))
            if not routes:
                routes = ['Inhalation']
            self._fill_cb(self.cb_intake, routes, block=True)
            self.cb_intake.setEnabled(True)
        else:
            # 常规监测：固定 Inhalation
            self._fill_cb(self.cb_intake, ['Inhalation'], block=True)
            self.cb_intake.setEnabled(False)

        # 触发监测参数刷新
        self._refresh_filter()
        # 触发 e 值刷新（气溶胶类型与③共用 cb_aerosol）
        self._refresh_e()

    def _on_intake_changed(self, route: str):
        """摄入途径变更：重刷监测参数过滤链 + e 值"""
        if not route:
            return
        self._refresh_filter()
        self._refresh_e()

    def _on_aerosol_changed(self, _aero: str):
        """气溶胶类型变更：刷新 m 值过滤 + 刷新 e 值插值（③④共用此下拉）"""
        self._refresh_filter()
        self._refresh_e()

    def _refresh_filter(self):
        """
        逐步筛选 m 值数据：
        途径 → 气溶胶 → 监测方法 → 时间 → fA → 读 m 值
        """
        nuc = self.cb_nuclide.currentText()
        if not nuc:
            return
        data = _ROUTINE_DATA if self._monitor_type == '常规监测' else _SPECIAL_DATA
        if data.empty:
            return

        df_n = data[data['radionuclide'].astype(str)
                    .str.replace('_', '-').str.strip() == nuc.strip()].copy()

        # 途径过滤
        route = self.cb_intake.currentText().strip()
        if route and 'intake_route' in df_n.columns:
            sub = df_n[df_n['intake_route'].astype(str).str.strip() == route]
            if not sub.empty:
                df_n = sub

        # 气溶胶类型
        if 'aerosol_type' in df_n.columns:
            aero_vals = sorted(df_n['aerosol_type'].dropna().astype(str).unique())
            self._fill_cb(self.cb_aerosol, aero_vals, block=True)
            sel_aero = self.cb_aerosol.currentText()
            if sel_aero:
                sub = df_n[df_n['aerosol_type'].astype(str) == sel_aero]
                if not sub.empty:
                    df_n = sub

        # 监测方法
        if 'monitoring_method' in df_n.columns:
            meth_vals = sorted(df_n['monitoring_method'].dropna().astype(str).unique())
            self._fill_cb(self.cb_method, meth_vals, block=True)
            sel_meth = self.cb_method.currentText()
            if sel_meth:
                sub = df_n[df_n['monitoring_method'].astype(str) == sel_meth]
                if not sub.empty:
                    df_n = sub

        # 时间
        if 'time_days' in df_n.columns:
            time_vals = sorted(df_n['time_days'].dropna().unique())
            self._fill_cb(self.cb_time, [str(t) for t in time_vals], block=True)
            sel_t = self.cb_time.currentText()
            try:
                sel_t_f = float(sel_t)
                sub = df_n[df_n['time_days'] == sel_t_f]
                if not sub.empty:
                    df_n = sub
            except Exception:
                pass

        # fA
        if 'fA' in df_n.columns:
            fa_vals = sorted(df_n['fA'].dropna().unique())
            if fa_vals:
                self._fill_cb(self.cb_fA, [str(v) for v in fa_vals], block=True)
                self.cb_fA.setEnabled(True)
                try:
                    sel_fa = float(self.cb_fA.currentText())
                    sub = df_n[df_n['fA'] == sel_fa]
                    if not sub.empty:
                        df_n = sub
                except Exception:
                    pass
            else:
                self._fill_cb(self.cb_fA, ['—'], block=True)
                self.cb_fA.setEnabled(False)
        else:
            self._fill_cb(self.cb_fA, ['—'], block=True)
            self.cb_fA.setEnabled(False)

        # 读 m 值
        if len(df_n) == 1 and 'm_value' in df_n.columns:
            m = df_n.iloc[0]['m_value']
            self.lbl_m.setText(f'{m:.3e}')
            self._m_value = float(m)
        elif len(df_n) > 1 and 'm_value' in df_n.columns:
            self.lbl_m.setText(f'匹配 {len(df_n)} 条，请细化条件')
            self._m_value = None
        else:
            self.lbl_m.setText('未找到')
            self._m_value = None

        self._update_I()

    def _on_ps_preset_changed(self, text: str):
        self.spin_ps_custom.setVisible(text == '自定义')
        self._refresh_e()

    def _refresh_e(self):
        nuc   = self.cb_nuclide.currentText()
        route = self.cb_intake.currentText() or 'Inhalation'
        if not nuc:
            self.lbl_e.setText('—')
            self._e_value = None
            return
        df_e = get_nuclide_df(nuc, route)
        if df_e is None or df_e.empty:
            self.lbl_e.setText('无数据')
            self._e_value = None
            self._update_I()
            return
        aero_e = self.cb_aerosol.currentText()
        ps = self._get_ps()
        if not aero_e or ps is None:
            self.lbl_e.setText('—')
            self._e_value = None
            self._update_I()
            return
        e = interp_dose_coeff(df_e, aero_e, ps)
        if e is not None:
            self.lbl_e.setText(f'{e:.4e} Sv/Bq  (插值 @ {ps:.1f}μm)')
            self._e_value = e
        else:
            self.lbl_e.setText('无数据')
            self._e_value = None
        self._update_I()

    def _update_I(self):
        m = self._m_value
        M = self.spin_M.value()
        if m and m > 0:
            I = calc_intake(M, m)
            self.lbl_I.setText(f'I = {I:.3e} Bq')
            self._I_value = I
        else:
            self.lbl_I.setText('I = —')
            self._I_value = None

    def _on_calc(self):
        self._update_I()
        self._refresh_e()
        I = self._I_value
        e = self._e_value
        m = self._m_value
        if I is None:
            QMessageBox.warning(self, '参数不完整',
                                'm 值未唯一确定，请调整监测参数筛选条件（当前仍匹配多行）')
            return
        if e is None:
            QMessageBox.warning(self, '参数不完整', '无法获取剂量系数 e，请检查核素数据或粒径/气溶胶类型设置')
            return
        E = calc_dose(I, e)
        self.lbl_dose.setText(f'E = {E:.4e} Sv')
        self._E_value = E
        self._calc_result = {
            'id':               self._next_id,
            'nuclide':          self.cb_nuclide.currentText(),
            'intake_route':     self.cb_intake.currentText(),
            'aerosol_type':     self.cb_aerosol.currentText(),
            'particle_size':    self._get_ps(),
            'monitoring_method': self.cb_method.currentText(),
            'time_days':        self._get_time(),
            'm_value':          m,
            'M':                self.spin_M.value(),
            'I':                I,
            'e_value':          e,
            'E':                E,
            'monitor_type':     self._monitor_type,
        }
        self.btn_add.setEnabled(True)

    def _on_add(self):
        if self._calc_result is None:
            return
        self.item_ready.emit(self._calc_result)
        self._next_id += 1
        self._calc_result = None
        self.btn_add.setEnabled(False)
        self.lbl_dose.setText('E = —')

    # ==================================================================
    #   工具方法
    # ==================================================================
    def _get_ps(self) -> float | None:
        txt = self.cb_ps_preset.currentText()
        if txt == '自定义':
            return float(self.spin_ps_custom.value())
        try:
            return float(txt)
        except Exception:
            return None

    def _get_time(self):
        try:
            return float(self.cb_time.currentText())
        except Exception:
            return self.cb_time.currentText()

    @staticmethod
    def _fill_cb(cb: QComboBox, items: list, block: bool = False):
        cur = cb.currentText()
        if block:
            cb.blockSignals(True)
        cb.clear()
        cb.addItems(items)
        idx = cb.findText(cur)
        cb.setCurrentIndex(idx if idx >= 0 else 0)
        if block:
            cb.blockSignals(False)

    def _clear_filters(self):
        for cb in [self.cb_intake, self.cb_aerosol, self.cb_method,
                   self.cb_time, self.cb_fA]:
            cb.blockSignals(True)
            cb.clear()
            cb.blockSignals(False)
        self.lbl_m.setText('—')
        self.lbl_e.setText('—')

# =====================================================================
#   SummaryPanel —— 右侧表格 + 汇总
# =====================================================================
class SummaryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setSpacing(6)

        self.table = ResultTable()
        self.table.row_deleted.connect(self._on_row_deleted)
        root.addWidget(self.table)

        sum_frame = QFrame()
        sum_frame.setFrameStyle(QFrame.StyledPanel)
        sum_frame.setStyleSheet('background:#d5f5e3; border-radius:6px;')
        sum_hl = QHBoxLayout(sum_frame)
        self.lbl_total = QLabel('总有效剂量：— Sv')
        self.lbl_total.setStyleSheet('font-size:15px; font-weight:bold; color:#148f77;')
        self.lbl_count = QLabel('共 0 项')
        sum_hl.addWidget(self.lbl_total)
        sum_hl.addStretch()
        sum_hl.addWidget(self.lbl_count)
        root.addWidget(sum_frame)

        btn_row = QHBoxLayout()
        self.btn_clear  = QPushButton('🗑 清空全部')
        self.btn_export = QPushButton('📥 导出 CSV')
        self.btn_clear.setStyleSheet('color:#c0392b; font-weight:bold;')
        self.btn_export.setStyleSheet(
            'background:#2980b9; color:white; padding:5px 14px; border-radius:4px;')
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_export)
        root.addLayout(btn_row)

        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_export.clicked.connect(self._on_export)

    def add_item(self, item: dict):
        self.table.add_item(item)
        self._refresh_total()

    def _on_row_deleted(self, _):
        self._refresh_total()

    def _on_clear(self):
        reply = QMessageBox.question(self, '确认清空', '确定清空所有监测项？',
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.table.clear_all()
            self._refresh_total()

    def _on_export(self):
        df = self.table.export_df()
        if df.empty:
            QMessageBox.information(self, '提示', '列表为空，无可导出数据')
            return
        path, _ = QFileDialog.getSaveFileName(self, '保存 CSV',
                                              'monitoring_dose_result.csv',
                                              'CSV Files (*.csv)')
        if path:
            df.to_csv(path, index=False, encoding='utf-8-sig')
            QMessageBox.information(self, '导出成功', f'已保存至:\n{path}')

    def _refresh_total(self):
        total = self.table.total_dose()
        n     = len(self.table._items)
        self.lbl_total.setText(f'总有效剂量：{total:.4e} Sv')
        self.lbl_count.setText(f'共 {n} 项')

# =====================================================================
#   主窗口
# =====================================================================
class MonitoringDoseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('监测法内照射剂量计算系统 v1.1  |  GB/T 16148')
        # 默认尺寸比 v1.0 增大 30px（1280→1310，760→790）
        self.resize(1310, 790)
        self.setMinimumSize(900, 600)
        self._build_ui()
        self._init_data()

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        v_main = QVBoxLayout(central)
        v_main.setSpacing(0)
        v_main.setContentsMargins(8, 8, 8, 6)

        # ── 标题 ──
        title = QLabel(
            '📋 监测法内照射有效剂量计算系统  |  依据 GB/T 16148 / ICRP Pub.54/78/130')
        title.setStyleSheet('font-size:16px; font-weight:bold; color:#1a5276; padding:4px 0;')
        v_main.addWidget(title)

        # ── 监测类型切换 ──
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel('监测类型：'))
        self.rb_routine = QRadioButton('常规监测')
        self.rb_special = QRadioButton('应急监测')
        self.rb_routine.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_routine)
        bg.addButton(self.rb_special)
        type_row.addWidget(self.rb_routine)
        type_row.addWidget(self.rb_special)
        type_row.addSpacing(20)
        # 提示标签
        self.lbl_route_hint = QLabel('（常规：仅吸入）')
        self.lbl_route_hint.setStyleSheet('color:#888; font-size:11px;')
        type_row.addWidget(self.lbl_route_hint)
        type_row.addStretch()
        self.lbl_data_status = QLabel('数据加载中…')
        self.lbl_data_status.setStyleSheet('color:#888;')
        type_row.addWidget(self.lbl_data_status)
        v_main.addLayout(type_row)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        v_main.addWidget(line)

        # ── 主体分割 ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)

        # 左：参数面板（可滚动，无最大宽度限制，完全可拖拽）
        self.param_panel = ParamPanel(monitor_type='常规监测')
        self.param_panel.setMinimumWidth(360)
        scroll = QScrollArea()
        scroll.setWidget(self.param_panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        splitter.addWidget(scroll)

        # 右：汇总面板
        self.summary_panel = SummaryPanel()
        self.summary_panel.setMinimumWidth(400)
        splitter.addWidget(self.summary_panel)

        # 初始左右比例 ~38:62（可拖动调整）
        splitter.setSizes([460, 850])
        v_main.addWidget(splitter, 1)

        # ── 状态栏 ──
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.status_lbl = QLabel('就绪')
        sb.addWidget(self.status_lbl)

        # ── 信号 ──
        bg.buttonClicked.connect(lambda btn: self._on_type_changed(btn.isChecked()))
        self.param_panel.item_ready.connect(self._on_item_ready)

    # ------------------------------------------------------------------
    def _init_data(self):
        self.status_lbl.setText('正在扫描数据…')
        QApplication.processEvents()
        scan_monitoring()
        scan_nuclides()

        msgs = []
        if not _ROUTINE_DATA.empty:
            msgs.append(f'常规监测 {len(_ROUTINE_DATA)} 条')
        if not _SPECIAL_DATA.empty:
            msgs.append(f'应急监测 {len(_SPECIAL_DATA)} 条')
        if _element_map:
            msgs.append(f'核素剂量系数 {sum(len(v) for v in _element_map.values())} 个')
        self.lbl_data_status.setText('  |  '.join(msgs) if msgs else '⚠️ 未找到数据')

        self.param_panel.populate()
        self.status_lbl.setText('就绪')

    # ------------------------------------------------------------------
    def _on_type_changed(self, checked):
        if checked:
            mtype = '常规监测' if self.rb_routine.isChecked() else '应急监测'
            self.param_panel.set_monitor_type(mtype)
            if mtype == '常规监测':
                self.lbl_route_hint.setText('（常规：仅吸入）')
            else:
                self.lbl_route_hint.setText('（应急：Inhalation / Injection / Ingestion）')

    def _on_item_ready(self, item: dict):
        self.summary_panel.add_item(item)
        self.status_lbl.setText(
            f"已添加: {item['nuclide']} [{item['intake_route']}]"
            f" | E = {item['E']:.3e} Sv  （{datetime.datetime.now().strftime('%H:%M:%S')}）")


# =====================================================================
#   入口
# =====================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont('Microsoft YaHei', 10)
    app.setFont(font)
    pal = app.palette()
    pal.setColor(pal.Window, QColor('#f4f6f7'))
    app.setPalette(pal)
    win = MonitoringDoseApp()
    win.show()
    sys.exit(app.exec_())

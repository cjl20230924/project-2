"""
空气采样法内照射剂量计算系统 v3.2 (桌面版)
基于 PyQt5 实现，支持四种计算方法：
  1. 国标单AMAD法（默认5μm）
  2. 单峰/双峰拟合法（多模态）
  3. 正态概率图法（直线拟合）
  4. 逐级独立法（每级视为独立均质气溶胶源）

v3.2 新增：
  - AIC/BIC 信息准则辅助方法选择
  - 数据质量自动检测（低活度/少级数 → 强制国标法）
  - 导出功能（Excel 完整报告 / CSV / 日志 / 含图一键导出）

运行：python dose_app.py
"""

from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QComboBox, QPushButton,
    QRadioButton, QButtonGroup, QTableWidget, QTableWidgetItem,
    QSplitter, QScrollArea, QFrame, QDoubleSpinBox, QSpinBox,
    QFileDialog, QMessageBox, QTabWidget, QHeaderView, QSizePolicy,
    QCheckBox, QTextEdit, QProgressBar, QStatusBar, QFormLayout
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import sys
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Qt5Agg')

# 中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei',
                                          'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # 数学符号用 STIX，兼容上下标


warnings.filterwarnings('ignore')

# ==================== 导入核心计算模块 ====================
try:
    from xf_core import (lognormal_pdf, stage_integral, fit_unimodal, fit_bimodal,
                         fit_linear_probit, judge_distribution, recommend_method,
                         calc_unimodal_stats, calc_bimodal_stats, calc_probit_stats,
                         data_quality_assessment,
                         stages_low, stages_high, cut_diameters, midpoints,
                         efficiency, apply_efficiency_correction)
except ImportError as e:
    print(f"[错误] 无法导入 xf_core 模块: {e}\n请确保 xf_core.py 在同目录下。")
    sys.exit(1)

# ==================== 全局常量 ====================
DATA_DIR = Path("./processed_nuclide_files")
STAGE_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Filter']
STAGE_RANGES = ['14.8–21.3 μm', '9.8–14.8 μm', '6.0–9.8 μm', '3.5–6.0 μm',
                '1.6–3.5 μm',  '0.9–1.6 μm',  '0.5–0.9 μm', '0.1–0.5 μm', '< 0.1 μm']
CONC_COLS = ['A_conc', 'B_conc', 'C_conc', 'D_conc', 'E_conc',
             'F_conc', 'G_conc', 'H_conc', 'Filter_conc']

MASK_PARAMS = {
    "无防护":       {"filtration_efficiency": 0.00, "leakage_rate": 1.00, "is_electret": False, "mpps_um": 0.3},
    "普通熔喷口罩": {"filtration_efficiency": 0.60, "leakage_rate": 0.15, "is_electret": False, "mpps_um": 0.3},
    "KN90":         {"filtration_efficiency": 0.90, "leakage_rate": 0.10, "is_electret": True,  "mpps_um": 0.3},
    "KN95":         {"filtration_efficiency": 0.95, "leakage_rate": 0.08, "is_electret": True,  "mpps_um": 0.3},
    "FFP2":         {"filtration_efficiency": 0.94, "leakage_rate": 0.08, "is_electret": True,  "mpps_um": 0.3},
    "N95":          {"filtration_efficiency": 0.95, "leakage_rate": 0.08, "is_electret": True,  "mpps_um": 0.3},
    "医用外科口罩": {"filtration_efficiency": 0.70, "leakage_rate": 0.20, "is_electret": True,  "mpps_um": 0.3},
}

# ==================== 核素数据缓存 ====================
_nuclide_cache = {}
_nuclide_files = {}
_element_nuclides_map = {}


def scan_nuclides():
    global _nuclide_files, _element_nuclides_map
    if not DATA_DIR.exists():
        return
    files = list(DATA_DIR.glob("*.parquet")) + list(DATA_DIR.glob("*.xlsx"))
    for f in files:
        try:
            df = pd.read_parquet(
                f) if f.suffix == '.parquet' else pd.read_excel(f)
            col_map = {}
            for c in df.columns:
                cl = c.lower().strip()
                if cl == 'element':
                    col_map[c] = 'element'
                elif 'radionuclide' in cl:
                    col_map[c] = 'radionuclide'
                elif 'route' in cl:
                    col_map[c] = 'route_of_intake'
                elif 'aerosol' in cl:
                    col_map[c] = 'aerosol_type'
                elif 'compound' in cl:
                    col_map[c] = 'compound'
                elif 'particle' in cl:
                    col_map[c] = 'particle_size'
                elif 'dose' in cl:
                    col_map[c] = 'dose_coefficient'
                elif cl == 'fa':
                    col_map[c] = 'fA'
            df = df.rename(columns=col_map)
            if 'particle_size' in df.columns:
                df['particle_size'] = pd.to_numeric(
                    df['particle_size'].astype(str)
                    .str.replace(r'\s*[µμ]m\s*|\s*micron\s*', '', regex=True)
                    .replace(['', 'nan', 'NaN', 'None'], np.nan),
                    errors='coerce')
            if 'dose_coefficient' in df.columns:
                df['dose_coefficient'] = pd.to_numeric(
                    df['dose_coefficient'], errors='coerce')
            if 'aerosol_type' in df.columns:
                df['aerosol_type'] = df['aerosol_type'].replace(
                    'Gaseous', 'Unspecified')
            if 'element' in df.columns and 'radionuclide' in df.columns:
                elem = str(df['element'].iloc[0])
                nuc = str(df['radionuclide'].iloc[0]).replace('_', '-')
                _nuclide_files[nuc] = (f, df)
                _element_nuclides_map.setdefault(elem, [])
                if nuc not in _element_nuclides_map[elem]:
                    _element_nuclides_map[elem].append(nuc)
        except Exception as e:
            print(f"[警告] 加载文件 {f.name} 失败: {e}")


def _normalize_nuclide(name):
    """归一化核素名：下划线→连字符，去空格"""
    return name.strip().replace('_', '-')


# ─ 化合物短名 → (parquet 全名, 对应气溶胶类型) ─
# aerosol_type 与 parquet 中实际绑定的类型完全一致，不得随意修改
_COMPOUND_SHORT_TO_FULL = {
    # ==================== Uranium (U) ====================
    # 根据 ICRP 68/72 分类
    'UO2':         ('Uranium octoxide, uranium dioxide',                                       'Intermediate Type M/S'),
    'U3O8':        ('Uranium octoxide, uranium dioxide',                                       'Intermediate Type M/S'),
    'UO3':         ('Uranyl nitrate, uranium peroxide hydrate, ammonium diuranate, uranium trioxide', 'Intermediate Type F/M'),
    '硝酸铀酰':    ('Uranyl nitrate, uranium peroxide hydrate, ammonium diuranate, uranium trioxide', 'Intermediate Type F/M'),
    'UNH':         ('Uranyl nitrate, uranium peroxide hydrate, ammonium diuranate, uranium trioxide', 'Intermediate Type F/M'),
    'ADU':         ('Uranyl nitrate, uranium peroxide hydrate, ammonium diuranate, uranium trioxide', 'Intermediate Type F/M'),
    'UF6':         ('Uranium hexafluoride, uranyl tributyl-phosphate',                         'Type F'),
    '铀酰三丁磷酸酯': ('Uranium hexafluoride, uranyl tributyl-phosphate',                         'Type F'),
    'U_metal':     ('Uranyl acetylacetonate; depleted uranium aerosols from use of kinetic energy penetrators; vaporised uranium metal; all unspecified forms', 'Type M'),
    'UAA':         ('Uranyl acetylacetonate; depleted uranium aerosols from use of kinetic energy penetrators; vaporised uranium metal; all unspecified forms', 'Type M'),
    '金属铀蒸气':  ('Uranyl acetylacetonate; depleted uranium aerosols from use of kinetic energy penetrators; vaporised uranium metal; all unspecified forms', 'Type M'),
    'DU':          ('Uranyl acetylacetonate; depleted uranium aerosols from use of kinetic energy penetrators; vaporised uranium metal; all unspecified forms', 'Type M'),
    '未知':        ('Uranyl acetylacetonate; depleted uranium aerosols from use of kinetic energy penetrators; vaporised uranium metal; all unspecified forms', 'Type M'),
    'U aluminide': ('Uranium aluminide',                                                       'Aerosols Uranium aluminide'),

    # ==================== Plutonium (Pu) ====================
    # 基础化合物（已有）
    'PuO2':        ('Plutonium-239 dioxide, plutonium in mixed oxide',                         'Unspecified'),
    'MOX':         ('Plutonium-239 dioxide, plutonium in mixed oxide',                         'Unspecified'),
    'Pu_nitrate':  ('Plutonium nitrate',                                                       'Unspecified'),
    '硝酸钚':      ('Plutonium nitrate',                                                       'Unspecified'),
    'Pu_citrate':  ('Plutonium citrate, plutonium tri-butyl-phosphate, plutonium chloride',    'Type M'),

    # 新增钚化合物（从 Pu-238 和 Pu-239 文件中提取）
    'Pu238O2_ceramic':      ('Plutonium-238 dioxide ceramic',                                 'Unspecified'),
    'Pu238O2_non_ceramic':  ('Plutonium-238 dioxide non-ceramic',                             'Unspecified'),
    'PuO2_nanoparticles':   ('Plutonium dioxide 1-nm nanoparticles',                          'Unspecified'),
    # 注意：Plutonium-239 dioxide, plutonium in mixed oxide 已由 'PuO2' 和 'MOX' 覆盖
    # Plutonium nitrate 已覆盖
    # Plutonium citrate... 已覆盖

    # ==================== Hydrogen / Tritium (H-3) ====================
    # 已有
    'HTO':         ('Gas or vapour Type V, Tritiated water',                                   'Unspecified'),
    '氚化水':      ('Gas or vapour Type V, Tritiated water',                                   'Unspecified'),
    'HT':          ('Gas or vapour Type V, Tritium gas',                                       'Unspecified'),
    '氚气':        ('Gas or vapour Type V, Tritium gas',                                       'Unspecified'),
    'OBT':         ('Biogenic organic compounds',                                              'Unspecified'),
    '有机氚':      ('Biogenic organic compounds',                                              'Unspecified'),

    # 新增氚化合物（从 H-3 文件中提取）
    'LaNiAl_tritide':   ('LaNiAl tritide',                                                     'Type F'),
    'TiZr_tritide':     ('All unspecified compounds, glass fragments, luminous paint, titanium tritide, zirconium tritide', 'Type M'),
    'C_Hf_tritide':     ('Carbon tritide, hafnium tritide',                                    'Type S'),
    'Tritiated_methane': ('Gas or vapour Type V, Tritiated methane',                            'Unspecified'),
}


def resolve_compound(short_name):
    """返回 (parquet 全名, 对应气溶胶类型)；未知短名则返回 (原值, None)"""
    r = _COMPOUND_SHORT_TO_FULL.get(short_name.strip())
    if r:
        return r  # (full_name, aerosol_type)
    return (short_name.strip(), None)


def get_nuclide_df(nuclide, inhalation_only=True):
    nuclide = _normalize_nuclide(nuclide)
    key = (nuclide, inhalation_only)
    if key in _nuclide_cache:
        return _nuclide_cache[key]
    if nuclide not in _nuclide_files:
        return None
    _, df = _nuclide_files[nuclide]
    if inhalation_only and 'route_of_intake' in df.columns:
        df = df[df['route_of_intake'] == 'Inhalation'].copy()
    _nuclide_cache[key] = df
    return df


def interp_dose_coeff(df, aerosol_type, particle_size_um):
    if df is None or df.empty:
        return None
    if 'aerosol_type' not in df.columns:
        return None
    sub = df[df['aerosol_type'] == aerosol_type].copy()
    if sub.empty:
        return None
    sub = sub.dropna(subset=['particle_size', 'dose_coefficient']).sort_values(
        'particle_size')
    if sub.empty:
        row0 = df[df['aerosol_type'] == aerosol_type]
        return row0['dose_coefficient'].iloc[0] if not row0.empty else None
    ps = sub['particle_size'].values
    dcs = sub['dose_coefficient'].values
    if len(ps) == 1:
        return dcs[0]
    if particle_size_um <= ps.min():
        return dcs[0]
    if particle_size_um >= ps.max():
        return dcs[-1]
    log_x = np.log(particle_size_um)
    log_ps = np.log(ps)
    for i in range(len(log_ps) - 1):
        if log_ps[i] <= log_x <= log_ps[i + 1]:
            y1, y2 = dcs[i], dcs[i + 1]
            x1, x2 = log_ps[i], log_ps[i + 1]
            return y1 + (log_x - x1) / (x2 - x1) * (y2 - y1) if x2 != x1 else y1
    return None


def calc_mask_pf(mask_type, particle_size_um=None):
    if mask_type == "无防护":
        return 1.0
    p = MASK_PARAMS.get(mask_type, MASK_PARAMS["无防护"])
    base_eff = p["filtration_efficiency"]
    leak = p["leakage_rate"]
    mpps = p["mpps_um"]
    is_elec = p["is_electret"]
    if particle_size_um is None:
        adj_eff = base_eff
    else:
        size = particle_size_um
        if not is_elec:
            adj_eff = base_eff * min(1.0, size / mpps) if size > 0.01 else 0.0
        else:
            if size <= 0.01:
                adj_eff = 0.1
            elif abs(size - mpps) < 1e-6:
                adj_eff = base_eff
            else:
                dist = abs(size - mpps)
                boost = min(dist / mpps * 0.15, 0.09)
                adj_eff = min(base_eff + boost, 0.99)
    return leak + (1 - adj_eff) * (1 - leak)


# ==================== 计算线程 ====================
class CalcThread(QThread):
    result_ready = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.result_ready.emit(result)
        except Exception as e:
            import traceback
            self.error_signal.emit(traceback.format_exc())


# ==================== 绘图 Canvas ====================
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=7):
        self.fig = Figure(figsize=(width, height), dpi=88)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def plot_fitting(self, raw_activities, corrected, amad_uni, gsd_uni, total_uni,
                     amad1, gsd1, frac1, amad2, gsd2, total_bi, total_corr,
                     D50_lin, GSD_lin, R2_lin, is_corrected=False):
        """
        绘制粒径分布拟合图
        Parameters
        ----------
        is_corrected : bool
            True  → corrected 数据已经过口罩防护校正（防护后）
            False → corrected 为效率校正后但未加口罩防护校正（防护前）
        """
        self.fig.clear()
        data_label = "防护后校正浓度" if is_corrected else "防护前浓度（效率校正）"
        plot_note = "（图中散点 = 口罩防护校正后数据）" if is_corrected else "（图中散点 = 防护前数据，效率校正）"

        x_plot = np.logspace(np.log10(0.1), np.log10(50), 300)
        stage_widths = np.log(stages_high / stages_low)
        data_density = corrected / stage_widths / \
            total_corr if total_corr > 0 else corrected

        y_uni = lognormal_pdf(x_plot, amad_uni, gsd_uni) * total_uni / total_corr \
            if total_corr > 0 else np.zeros_like(x_plot)

        # ── 2×2 布局 ──
        axes = self.fig.subplots(2, 2)

        # 标注防护状态的颜色
        scatter_color = '#e74c3c' if is_corrected else '#3498db'
        scatter_label = f'实测数据 {plot_note}'

        # ── 左上：单峰拟合 ──
        ax = axes[0, 0]
        ax.semilogx(x_plot, y_uni, 'b-', lw=2,
                    label=f'单峰 AMAD={amad_uni:.2f} $\mu m$')
        ax.scatter(midpoints, data_density, c=scatter_color, s=50, edgecolors='k', zorder=5,
                   label=data_label)
        ax.set_xlabel('空气动力学粒径 ($\mu m$)', fontsize=9)
        ax.set_ylabel('归一化密度', fontsize=9)
        ax.set_title(f'单峰对数正态拟合  [{data_label}]', fontsize=9)
        ax.grid(ls='--', alpha=0.5)
        ax.set_xlim(0.1, 50)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc='upper left')

        # ── 右上：双峰拟合 ──
        ax = axes[0, 1]
        if not np.isnan(amad2):
            y_coarse = frac1 * lognormal_pdf(x_plot, amad1, gsd1)
            y_fine = (1 - frac1) * lognormal_pdf(x_plot, amad2, gsd2)
            y_total = y_coarse + y_fine
            ax.semilogx(x_plot, y_total, 'k-', lw=2, label='总拟合')
            ax.semilogx(x_plot, y_coarse, 'b--', lw=1.5,
                        label=f'粗峰 {amad1:.1f} $\mu m$')
            ax.semilogx(x_plot, y_fine, 'r--', lw=1.5,
                        label=f'细峰 {amad2:.1f} $\mu m$')
        else:
            ax.semilogx(x_plot, y_uni, 'k--', lw=2, label='未检测到双峰')
        ax.scatter(midpoints, data_density, c=scatter_color, s=50, edgecolors='k', zorder=5,
                   label=data_label)
        ax.set_xlabel('空气动力学粒径 ($\mu m$)', fontsize=9)
        ax.set_title(f'双峰对数正态拟合  [{data_label}]', fontsize=9)
        ax.grid(ls='--', alpha=0.5)
        ax.set_xlim(0.1, 50)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc='upper left')

        # ── 左下：正态概率图（对齐 xf_core.py 原始逻辑）──
        ax = axes[1, 0]
        raw_acts = raw_activities[:8]
        filter_act = raw_activities[8]
        total_ra = np.sum(raw_acts) + filter_act

        if total_ra > 0:
            # 累积小于某粒径的活度百分比（ICRP 推荐算法）
            f = raw_acts / total_ra
            f_all = np.concatenate([f, [filter_act / total_ra]])
            cum_from_fine = np.cumsum(np.flip(f_all))[:-1]
            cum_less = np.flip(cum_from_fine) * 100
            valid = (cum_less > 1) & (cum_less < 99)

            if np.sum(valid) >= 3:
                x_v = np.log(cut_diameters[valid])
                y_v = norm.ppf(cum_less[valid] / 100)

                reg = LinearRegression().fit(x_v.reshape(-1, 1), y_v)
                slope, intercept_ = reg.coef_[0], reg.intercept_

                ln_AMAD = -intercept_ / slope
                AMAD_probit = np.exp(ln_AMAD)
                ln_GSD = 1 / slope
                GSD_probit = np.exp(ln_GSD)
                D84_1 = np.exp(ln_AMAD + ln_GSD)
                D15_9 = np.exp(ln_AMAD - ln_GSD)

                ax.set_xscale('log')
                ax.set_xlim(0.3, 30)

                percentiles = [1, 5, 10, 20, 30,
                               40, 50, 60, 70, 80, 90, 95, 99]
                probit_ticks = norm.ppf(np.array(percentiles) / 100)
                ax.set_yticks(probit_ticks)
                ax.set_yticklabels([str(p) for p in percentiles], fontsize=7)
                ax.set_ylim(norm.ppf(0.005), norm.ppf(0.995))

                ax.scatter(cut_diameters[valid], y_v, c='red', s=60,
                           label='实测数据', zorder=5)

                x_fit = np.logspace(np.log10(0.5), np.log10(21.3), 200)
                y_fit = slope * np.log(x_fit) + intercept_
                ax.plot(x_fit, y_fit, 'b-', lw=2, label='ICRP 回归线')

                ax.scatter(AMAD_probit, 0, c='green', s=90, marker='s',
                           label=f'$AMAD(D_{{50}})$={AMAD_probit:.2f} $\mu m$', zorder=6)
                ax.scatter(D84_1, norm.ppf(0.8413), c='orange', s=90, marker='s',
                           label=f'$D_{{84.1}}$={D84_1:.1f} $\mu m$', zorder=6)
                ax.scatter(D15_9, norm.ppf(0.1587), c='purple', s=90, marker='s',
                           label=f'$D_{{15.9}}$={D15_9:.1f} $\mu m$', zorder=6)

                ax.axhline(y=0, color='gray', ls=':', alpha=0.5)
                ax.axhline(y=norm.ppf(0.8413), color='gray', ls=':', alpha=0.5)
                ax.axhline(y=norm.ppf(0.1587), color='gray', ls=':', alpha=0.5)

                ss_res = np.sum((y_v - reg.predict(x_v.reshape(-1, 1))) ** 2)
                ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
                r2_calc = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                ax.text(0.05, 0.95,
                        f'$AMAD$ = {AMAD_probit:.2f} $\mu m$\n'
                        f'$GSD$  = {GSD_probit:.2f}\n'
                        f'$R^2$   = {r2_calc:.4f}',
                        transform=ax.transAxes, va='top', fontsize=8,
                        bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))
        ax.set_xlabel('空气动力学粒径 ($\mu m$)', fontsize=9)
        ax.set_ylabel('累积活度比例 (%)', fontsize=9)
        ax.set_title(f'正态概率图（ICRP方法）  [{data_label}]', fontsize=9)
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.legend(fontsize=6.5, loc='lower right')

        # ── 右下：拟合参数摘要 ──
        ax = axes[1, 1]
        ax.axis('off')
        lines = []
        lines.append('══════ 拟合参数摘要 ══════')
        lines.append(f'数据类型: {data_label}')
        lines.append('')
        lines.append('【单峰对数正态】')
        lines.append(f'  $AMAD$ = {amad_uni:.3f} $\\mu$m')
        lines.append(f'  $GSD$  = {gsd_uni:.3f}')
        lines.append('')
        if not np.isnan(amad2):
            lines.append('【双峰对数正态】')
            lines.append(f'  粗峰 $AMAD$ = {amad1:.3f} $\\mu$m')
            lines.append(f'  粗峰 $GSD$  = {gsd1:.3f}')
            lines.append(f'  细峰 $AMAD$ = {amad2:.3f} $\\mu$m')
            lines.append(f'  细峰 $GSD$  = {gsd2:.3f}')
            lines.append(f'  粗峰占比  = {frac1:.2%}')
        else:
            lines.append('【双峰对数正态】')
            lines.append('  (未检测到)')
        lines.append('')
        if not np.isnan(D50_lin) and not np.isnan(GSD_lin):
            lines.append('【正态概率图】')
            lines.append(f'  $AMAD$ = {D50_lin:.3f} $\\mu$m')
            lines.append(f'  $GSD$  = {GSD_lin:.3f}')
            lines.append(f'  $R^2$   = {R2_lin:.4f}')
        else:
            lines.append('【正态概率图】')
            lines.append('  (数据点不足)')
        text = '\n'.join(lines)
        ax.text(0.1, 0.95, text, transform=ax.transAxes, ha='left', va='top',
                fontsize=8.5,
                bbox=dict(boxstyle='round', fc='#f0f4ff', alpha=0.9, ec='#6688aa'))

        self.fig.tight_layout(pad=2.0)
        self.draw()


# ==================== 核素化合物配置组件（垂直三行式） ====================
class NuclideCompoundWidget(QFrame):
    """每个化合物配置为一个卡片（三行布局）
    行1: 化合物输入 | 活度占比 | 删除
    行2: 气溶胶类型（自动推断提示 + 手动覆盖下拉）
    行3: 核素丰度
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { border:1px solid #cce0ff; border-radius:5px; background:#f8fbff; }")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(3)

        # ── 第一行：化合物 | 活度占比 | 删除按钮 ──
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        lbl_comp = QLabel("化合物:")
        lbl_comp.setFixedWidth(52)
        self.compound_edit = QLineEdit("UO2")
        self.compound_edit.setPlaceholderText("如 UO2, U3O8, UO3, UF6…")
        self.compound_edit.setMinimumWidth(80)

        lbl_frac = QLabel("活度占比:")
        lbl_frac.setFixedWidth(62)
        self.act_frac_spin = QDoubleSpinBox()
        self.act_frac_spin.setRange(0.0, 1.0)
        self.act_frac_spin.setSingleStep(0.05)
        self.act_frac_spin.setValue(1.0)
        self.act_frac_spin.setDecimals(3)
        self.act_frac_spin.setMinimumWidth(80)

        self.del_btn = QPushButton("✕")
        self.del_btn.setFixedSize(26, 26)
        self.del_btn.setStyleSheet("color:red; font-weight:bold; border:none;")

        row1.addWidget(lbl_comp)
        row1.addWidget(self.compound_edit, 3)
        row1.addWidget(lbl_frac)
        row1.addWidget(self.act_frac_spin, 2)
        row1.addWidget(self.del_btn)
        outer.addLayout(row1)

        # ── 第二行：气溶胶类型（自动推断标签 + 手动覆盖下拉） ──
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        lbl_at = QLabel("气溶胶类型:")
        lbl_at.setFixedWidth(68)

        # 自动推断提示：根据化合物名自动更新
        self.aerosol_inferred_lbl = QLabel("（自动推断: —）")
        self.aerosol_inferred_lbl.setStyleSheet(
            "color:#1e8449; font-size:10px; font-style:italic;")
        self.aerosol_inferred_lbl.setMinimumWidth(180)

        lbl_override = QLabel("手动覆盖:")
        lbl_override.setFixedWidth(58)
        lbl_override.setStyleSheet("color:#888; font-size:10px;")
        self.aerosol_combo = QComboBox()
        self.aerosol_combo.setMinimumWidth(130)
        self.aerosol_combo.setToolTip(
            "默认由化合物名自动推断气溶胶类型。\n"
            "若需手动覆盖，请在此选择。\n"
            "已知映射：UO2/U3O8→Intermediate Type M/S，\n"
            "UO3/UNH/ADU→Intermediate Type F/M，UF6→Type F"
        )

        row2.addWidget(lbl_at)
        row2.addWidget(self.aerosol_inferred_lbl, 2)
        row2.addWidget(lbl_override)
        row2.addWidget(self.aerosol_combo, 2)
        outer.addLayout(row2)

        # ── 第三行：核素丰度 ──
        row3 = QHBoxLayout()
        row3.setSpacing(6)
        lbl_nuc = QLabel("核素丰度:")
        lbl_nuc.setFixedWidth(62)
        self.nuclides_edit = QLineEdit("U_238:0.993|U_235:0.007")
        self.nuclides_edit.setPlaceholderText(
            "格式：核素:比例|核素:比例  例如 U_238:0.993|U_235:0.007")
        row3.addWidget(lbl_nuc)
        row3.addWidget(self.nuclides_edit, 1)
        outer.addLayout(row3)

        # 化合物名变化时自动更新推断标签
        self.compound_edit.textChanged.connect(self._update_inferred_label)
        self._update_inferred_label(self.compound_edit.text())

    def _update_inferred_label(self, text):
        """根据化合物短名更新推断气溶胶类型标签"""
        name = text.strip()
        result = _COMPOUND_SHORT_TO_FULL.get(name)
        if result:
            _, aerosol = result
            self.aerosol_inferred_lbl.setText(f"✔ 自动推断: {aerosol}")
            self.aerosol_inferred_lbl.setStyleSheet(
                "color:#1e8449; font-size:10px; font-style:italic;")
        else:
            self.aerosol_inferred_lbl.setText("⚠ 未知化合物，将使用手动覆盖值")
            self.aerosol_inferred_lbl.setStyleSheet(
                "color:#c0392b; font-size:10px; font-style:italic;")

    def get_data(self):
        nuclide_str = self.nuclides_edit.text().strip()
        nuclides = {}
        for part in nuclide_str.split('|'):
            part = part.strip()
            if ':' in part:
                n, v = part.split(':', 1)
                try:
                    nuclides[n.strip()] = float(v.strip())
                except Exception:
                    pass
        compound = self.compound_edit.text().strip()
        # 优先使用自动推断；若化合物不在映射表则使用 combo 手动选择值
        result = _COMPOUND_SHORT_TO_FULL.get(compound)
        aerosol_type = result[1] if result else self.aerosol_combo.currentText(
        )
        return {
            'compound':          compound,
            'aerosol_type':      aerosol_type,
            'activity_fraction': self.act_frac_spin.value(),
            'nuclides':          nuclides,
        }

    def update_aerosol_options(self, options):
        cur = self.aerosol_combo.currentText()
        self.aerosol_combo.clear()
        self.aerosol_combo.addItems(options)
        idx = self.aerosol_combo.findText(cur)
        if idx >= 0:
            self.aerosol_combo.setCurrentIndex(idx)


# ==================== 主窗口 ====================
class DoseCalcApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("空气采样法内照射剂量计算系统 v3.2")
        self.setMinimumSize(1100, 700)
        self.resize(1440, 860)

        self._fit_result = None
        self._calc_thread = None
        self._file_df = None

        self._setup_ui()
        self._scan_data()

    # ─────────────────────────────────────────────────────────
    # UI 构建
    # ─────────────────────────────────────────────────────────
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)
        root_lay.setSpacing(4)
        root_lay.setContentsMargins(8, 4, 8, 4)

        # 顶部标题
        title = QLabel("空气采样法 · 内照射剂量计算系统  v3.2")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("微软雅黑", 14, QFont.Bold))
        title.setStyleSheet("color:#1a5276; padding:3px 0;")
        root_lay.addWidget(title)

        # 主分割器（左：参数输入  右：结果展示）
        self._splitter = QSplitter(Qt.Horizontal)
        root_lay.addWidget(self._splitter, 1)

        # ── 左侧滚动面板 ──
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(400)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        left_content = QWidget()
        left_lay = QVBoxLayout(left_content)
        left_lay.setSpacing(7)
        left_lay.setContentsMargins(6, 6, 6, 6)
        left_scroll.setWidget(left_content)
        self._splitter.addWidget(left_scroll)

        # ── 右侧结果面板 ──
        right_panel = QWidget()
        right_lay = QVBoxLayout(right_panel)
        right_lay.setSpacing(4)
        right_lay.setContentsMargins(4, 0, 4, 0)
        self._splitter.addWidget(right_panel)

        # 初始比例：左 600，右剩余；两侧均可拖拽但不可折叠
        self._splitter.setSizes([600, 990])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, False)
        self._splitter.setHandleWidth(6)        # 拖拽手柄加宽，更易抓取
        self._splitter.setStyleSheet(
            "QSplitter::handle { background:#c8d8ea; border-radius:3px; }"
            "QSplitter::handle:hover { background:#3498db; }"
        )

        # ══════════════════════════════════════════════════════
        # 左侧区域
        # ══════════════════════════════════════════════════════

        # ─ § 1. 采样数据输入 ─
        grp1 = QGroupBox("① 采样数据输入")
        grp1.setFont(QFont("微软雅黑", 10, QFont.Bold))
        g1 = QVBoxLayout(grp1)
        g1.setSpacing(5)

        # 数据来源选择
        src_row = QHBoxLayout()
        self.rb_manual = QRadioButton("手动输入各级浓度")
        self.rb_file = QRadioButton("从 CSV / Excel 读取")
        self.rb_manual.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_manual)
        bg.addButton(self.rb_file)
        src_row.addWidget(self.rb_manual)
        src_row.addWidget(self.rb_file)
        src_row.addStretch()
        g1.addLayout(src_row)

        # 文件读取行（初始隐藏）
        self.file_row = QWidget()
        fr = QGridLayout(self.file_row)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.setSpacing(4)
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("选择 CSV / Excel 文件…")
        btn_browse = QPushButton("浏览…")
        btn_browse.setFixedWidth(56)
        btn_browse.clicked.connect(self._browse_file)
        self.ws_combo = QComboBox()
        self.ws_combo.setMinimumWidth(90)
        self.sid_combo = QComboBox()
        self.sid_combo.setMinimumWidth(90)
        self.ws_combo.currentTextChanged.connect(self._on_ws_changed)
        self.sid_combo.currentTextChanged.connect(self._on_sid_changed)
        fr.addWidget(QLabel("文件:"),          0, 0)
        fr.addWidget(self.file_path_edit,       0, 1, 1, 3)
        fr.addWidget(btn_browse,                0, 4)
        fr.addWidget(QLabel("车间:"),           1, 0)
        fr.addWidget(self.ws_combo,             1, 1)
        fr.addWidget(QLabel("采样 ID:"),        1, 2)
        fr.addWidget(self.sid_combo,            1, 3, 1, 2)
        self.file_row.setVisible(False)
        g1.addWidget(self.file_row)

        # 9 级浓度表格（用 QTableWidget，可以更紧凑地展示）
        conc_lbl = QLabel("各级活度浓度 (Bq/m³)：")
        conc_lbl.setStyleSheet("color:#1a5276; font-weight:bold;")
        g1.addWidget(conc_lbl)

        self.conc_table = QTableWidget(9, 2)
        self.conc_table.setHorizontalHeaderLabels(
            ["级别 / 粒径范围", "活度浓度 (Bq/m³)"])
        self.conc_table.verticalHeader().setVisible(False)
        self.conc_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.conc_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.conc_table.setAlternatingRowColors(True)
        self.conc_table.setMinimumHeight(258)
        self.conc_table.setMaximumHeight(300)

        self.conc_spins = []
        for i, (name, rng) in enumerate(zip(STAGE_NAMES, STAGE_RANGES)):
            lbl_item = QTableWidgetItem(f"{name}  ({rng})")
            lbl_item.setFlags(Qt.ItemIsEnabled)
            lbl_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            self.conc_table.setItem(i, 0, lbl_item)
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1e9)
            sp.setDecimals(4)
            sp.setValue(0.0)
            sp.setButtonSymbols(QDoubleSpinBox.NoButtons)
            sp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.conc_spins.append(sp)
            self.conc_table.setCellWidget(i, 1, sp)
            self.conc_table.setRowHeight(i, 28)
        g1.addWidget(self.conc_table)

        self.rb_manual.toggled.connect(self._on_input_mode_changed)
        self.rb_file.toggled.connect(self._on_input_mode_changed)
        left_lay.addWidget(grp1)

        # ─ § 2. 防护 & 呼吸参数 ─
        grp4 = QGroupBox("② 防护 & 呼吸参数")
        grp4.setFont(QFont("微软雅黑", 10, QFont.Bold))
        g4 = QGridLayout(grp4)
        g4.setSpacing(6)
        g4.setColumnStretch(1, 1)
        g4.setColumnStretch(3, 1)

        g4.addWidget(QLabel("口罩型号:"), 0, 0)
        self.mask_combo = QComboBox()
        self.mask_combo.addItems(list(MASK_PARAMS.keys()))
        g4.addWidget(self.mask_combo, 0, 1)
        self.mask_info_lbl = QLabel("")
        self.mask_info_lbl.setStyleSheet("color:#555; font-size:10px;")
        g4.addWidget(self.mask_info_lbl, 0, 2, 1, 2)
        self.mask_combo.currentTextChanged.connect(self._update_mask_info)

        g4.addWidget(QLabel("呼吸速率 BR (m³/h):"), 1, 0)
        self.br_spin = QDoubleSpinBox()
        self.br_spin.setRange(0.1, 5.0)
        self.br_spin.setValue(1.2)
        self.br_spin.setSingleStep(0.1)
        self.br_spin.setDecimals(2)
        g4.addWidget(self.br_spin, 1, 1)

        g4.addWidget(QLabel("工作时长 T (h):"), 1, 2)
        self.work_spin = QDoubleSpinBox()
        self.work_spin.setRange(0.5, 24.0)
        self.work_spin.setValue(8.0)
        self.work_spin.setSingleStep(0.5)
        self.work_spin.setDecimals(1)
        g4.addWidget(self.work_spin, 1, 3)

        self._update_mask_info()
        left_lay.addWidget(grp4)

        # ─ 运行拟合按钮（在防护参数下方，计算方法选择上方）─
        fit_btn_row = QHBoxLayout()
        fit_btn_row.addStretch()
        self.fit_btn = QPushButton("🔍  运行拟合（查看粒径分布图）")
        self.fit_btn.setMinimumHeight(36)
        self.fit_btn.setFont(QFont("微软雅黑", 11, QFont.Bold))
        self.fit_btn.setStyleSheet(
            "QPushButton { background:#27ae60; color:white; border-radius:5px; padding:4px 20px; }"
            "QPushButton:hover { background:#2ecc71; }"
            "QPushButton:disabled { background:#aaa; }"
        )
        self.fit_btn.clicked.connect(self._on_fit)
        fit_btn_row.addWidget(self.fit_btn)
        fit_btn_hint = QLabel("  ↳ 先运行拟合可在右侧查看粒径分布图，再选择计算方法开始计算")
        fit_btn_hint.setStyleSheet("color:#888; font-size:10px;")
        left_lay.addLayout(fit_btn_row)
        left_lay.addWidget(fit_btn_hint)

        # ─ § 3. 计算方法 ─
        grp2 = QGroupBox("③ 计算方法选择")
        grp2.setFont(QFont("微软雅黑", 10, QFont.Bold))
        g2 = QVBoxLayout(grp2)
        g2.setSpacing(3)

        self.rb_std = QRadioButton("方法1 · 国标单AMAD（固定 5 μm）")
        self.rb_modal = QRadioButton("方法2 · 多模态拟合（手动选择峰型）")
        self.rb_probit = QRadioButton("方法3 · 正态概率图法（直线拟合）")
        self.rb_stage = QRadioButton("方法4 · 逐级独立法（每级视为均质源）")
        self.rb_std.setChecked(True)
        bg2 = QButtonGroup(self)
        _hints = {
            self.rb_std:    "  ↳ 使用固定 AMAD = 5 μm，符合 ICRP‑66 及国标。",
            self.rb_modal:  "  ↳ 对分级数据拟合对数正态分布，由您在下方选择单峰或双峰。",
            self.rb_probit: "  ↳ 用累积活度–正态概率图进行直线回归，求 AMAD 与 GSD。",
            self.rb_stage:  "  ↳ 每级视为独立气溶胶源，利用该级中值粒径直接查表后逐级累加。",
        }
        self._hint_lbls = {}

        # ── 方法2 峰型子选项（单峰/双峰，仅当选方法2时可用）──
        self._modal_peak_widget = QWidget()
        modal_peak_row = QHBoxLayout(self._modal_peak_widget)
        modal_peak_row.setContentsMargins(22, 2, 0, 2)
        modal_peak_row.setSpacing(14)
        peak_lbl = QLabel("峰型选择：")
        peak_lbl.setStyleSheet("color:#1a5276; font-size:10px;")
        self.rb_peak_uni = QRadioButton("单峰（Unimodal）")
        self.rb_peak_bi  = QRadioButton("双峰（Bimodal）")
        self.rb_peak_uni.setChecked(True)
        self.rb_peak_uni.setStyleSheet("font-size:10px;")
        self.rb_peak_bi.setStyleSheet("font-size:10px;")
        bg_peak = QButtonGroup(self)
        bg_peak.addButton(self.rb_peak_uni)
        bg_peak.addButton(self.rb_peak_bi)
        peak_ref_lbl = QLabel(
            "  （拟合图运行后可在右侧查看双峰结果，再决定选哪种）")
        peak_ref_lbl.setStyleSheet("color:#888; font-size:9px;")
        modal_peak_row.addWidget(peak_lbl)
        modal_peak_row.addWidget(self.rb_peak_uni)
        modal_peak_row.addWidget(self.rb_peak_bi)
        modal_peak_row.addWidget(peak_ref_lbl)
        modal_peak_row.addStretch()
        self._modal_peak_widget.setVisible(False)

        for rb, hint in _hints.items():
            bg2.addButton(rb)
            g2.addWidget(rb)
            hl = QLabel(hint)
            hl.setStyleSheet("color:#888; font-size:10px; margin-left:16px;")
            hl.setWordWrap(True)
            g2.addWidget(hl)
            self._hint_lbls[rb] = hl
            # 在方法2后插入峰型子选项
            if rb is self.rb_modal:
                g2.addWidget(self._modal_peak_widget)
            rb.toggled.connect(self._update_method_hints)
        self._update_method_hints()
        left_lay.addWidget(grp2)

        # ─ § 4. 核素 / 化合物配置 ─
        grp3 = QGroupBox("④ 核素 / 化合物配置")
        grp3.setFont(QFont("微软雅黑", 10, QFont.Bold))
        g3 = QVBoxLayout(grp3)
        g3.setSpacing(4)

        # 模式切换提示
        nuc_mode_row = QHBoxLayout()
        nuc_mode_lbl = QLabel("配置模式：")
        nuc_mode_lbl.setStyleSheet("font-weight:bold; color:#1a5276;")
        self.rb_nuc_auto = QRadioButton("自动从采样数据读取")
        self.rb_nuc_manual = QRadioButton("手动输入")
        self.rb_nuc_manual.setChecked(True)
        bg_nuc = QButtonGroup(self)
        bg_nuc.addButton(self.rb_nuc_auto)
        bg_nuc.addButton(self.rb_nuc_manual)
        self.rb_nuc_auto.toggled.connect(self._on_nuc_mode_changed)
        self.rb_nuc_manual.toggled.connect(self._on_nuc_mode_changed)
        nuc_mode_row.addWidget(nuc_mode_lbl)
        nuc_mode_row.addWidget(self.rb_nuc_auto)
        nuc_mode_row.addWidget(self.rb_nuc_manual)
        nuc_mode_row.addStretch()
        g3.addLayout(nuc_mode_row)

        # 自动模式提示标签
        self.nuc_auto_hint = QLabel(
            '  \u2714 已选择【自动】模式：切换左侧【车间/采样ID】后，化合物配置将自动填入。\n'
            '  若采样数据中无化合物信息，仍可在下方手动编辑。'
        )
        self.nuc_auto_hint.setStyleSheet(
            "background:#eafaf1; border:1px solid #a9dfbf; border-radius:4px; "
            "color:#1e8449; font-size:10px; padding:4px; margin:2px 0;")
        self.nuc_auto_hint.setWordWrap(True)
        self.nuc_auto_hint.setVisible(False)
        g3.addWidget(self.nuc_auto_hint)

        # 手动模式提示标签
        self.nuc_manual_hint = QLabel(
            '  \u270f 手动模式：请在下方直接填写化合物名称、核素丰度等参数。\n'
            '  气溶胶类型会根据化合物名自动推断（详见 _COMPOUND_SHORT_TO_FULL 映射表）。'
        )
        self.nuc_manual_hint.setStyleSheet(
            "background:#eaf2ff; border:1px solid #aac4e0; border-radius:4px; "
            "color:#1a5276; font-size:10px; padding:4px; margin:2px 0;")
        self.nuc_manual_hint.setWordWrap(True)
        self.nuc_manual_hint.setVisible(True)
        g3.addWidget(self.nuc_manual_hint)

        nuc_row = QHBoxLayout()
        nuc_row.addWidget(QLabel("元素:"))
        self.elem_combo = QComboBox()
        self.elem_combo.setMinimumWidth(80)
        self.elem_combo.currentTextChanged.connect(self._on_elem_changed)
        nuc_row.addWidget(self.elem_combo)
        nuc_row.addStretch()
        btn_add = QPushButton("＋ 添加化合物")
        btn_add.setFixedWidth(110)
        btn_add.clicked.connect(self._add_compound_row)
        nuc_row.addWidget(btn_add)
        g3.addLayout(nuc_row)

        # 化合物行滚动区
        comp_scroll = QScrollArea()
        comp_scroll.setWidgetResizable(True)
        comp_scroll.setFrameShape(QFrame.NoFrame)
        comp_scroll.setMinimumHeight(110)
        comp_scroll.setMaximumHeight(280)
        self._comp_container = QWidget()
        self._comp_lay = QVBoxLayout(self._comp_container)
        self._comp_lay.setSpacing(4)
        self._comp_lay.setContentsMargins(0, 0, 0, 0)
        self._comp_lay.addStretch()
        comp_scroll.setWidget(self._comp_container)
        g3.addWidget(comp_scroll)
        self._compound_widgets = []
        self._add_compound_row()
        left_lay.addWidget(grp3)

        # ─ § 5. 计算按钮 ─
        btn_row = QHBoxLayout()
        self.calc_btn = QPushButton("▶  开始计算")
        self.calc_btn.setMinimumHeight(40)
        self.calc_btn.setFont(QFont("微软雅黑", 12, QFont.Bold))
        self.calc_btn.setStyleSheet(
            "QPushButton { background:#1a5276; color:white; border-radius:6px; }"
            "QPushButton:hover { background:#2471a3; }"
            "QPushButton:disabled { background:#aaa; }"
        )
        self.calc_btn.clicked.connect(self._on_calc)
        btn_clear = QPushButton("清空结果")
        btn_clear.setMinimumHeight(40)
        btn_clear.setFixedWidth(90)
        btn_clear.clicked.connect(self._clear_results)
        btn_row.addWidget(self.calc_btn, 1)
        btn_row.addWidget(btn_clear)
        left_lay.addLayout(btn_row)
        left_lay.addStretch()

        # ══════════════════════════════════════════════════════
        # 右侧：结果展示 Tab
        # ══════════════════════════════════════════════════════
        result_tabs = QTabWidget()
        result_tabs.setDocumentMode(True)
        right_lay.addWidget(result_tabs, 1)

        # ── Tab 1: 拟合图 ──
        tab_fit = QWidget()
        tfl = QVBoxLayout(tab_fit)
        tfl.setContentsMargins(4, 4, 4, 4)
        # 内部垂直分割：上方 canvas，下方拟合信息标签（可拖拽）
        fit_splitter = QSplitter(Qt.Vertical)
        fit_splitter.setHandleWidth(6)
        fit_splitter.setStyleSheet(
            "QSplitter::handle { background:#c8d8ea; border-radius:3px; }"
            "QSplitter::handle:hover { background:#3498db; }"
        )
        self.plot_canvas = PlotCanvas(tab_fit, width=7, height=7)
        self.fit_info_lbl = QLabel("拟合参数将在计算后显示")
        self.fit_info_lbl.setStyleSheet(
            "background:#eaf2ff; border:1px solid #aac; padding:6px; border-radius:4px;")
        self.fit_info_lbl.setWordWrap(True)
        self.fit_info_lbl.setMinimumHeight(40)
        fit_splitter.addWidget(self.plot_canvas)
        fit_splitter.addWidget(self.fit_info_lbl)
        fit_splitter.setSizes([560, 80])
        fit_splitter.setCollapsible(0, False)
        fit_splitter.setCollapsible(1, False)
        tfl.addWidget(fit_splitter, 1)
        result_tabs.addTab(tab_fit, "📈 拟合图")

        # ── Tab 2: 剂量结果 ──
        tab_dose = QWidget()
        tdl = QVBoxLayout(tab_dose)
        tdl.setContentsMargins(4, 4, 4, 4)
        tdl.setSpacing(4)

        # 用垂直 splitter 分割"结果表区"与"累计+导出区"
        dose_splitter = QSplitter(Qt.Vertical)
        dose_splitter.setHandleWidth(6)
        dose_splitter.setStyleSheet(
            "QSplitter::handle { background:#c8d8ea; border-radius:3px; }"
            "QSplitter::handle:hover { background:#3498db; }"
        )

        # ── 上半：汇总标签 + 结果明细表 ──
        top_dose_widget = QWidget()
        top_dose_lay = QVBoxLayout(top_dose_widget)
        top_dose_lay.setContentsMargins(0, 0, 0, 0)
        top_dose_lay.setSpacing(4)
        self.summary_lbl = QLabel("剂量结果将在计算后显示")
        self.summary_lbl.setStyleSheet(
            "background:#eafaf1; border:1px solid #aad; padding:8px; font-size:13px; border-radius:4px;")
        self.summary_lbl.setWordWrap(True)
        top_dose_lay.addWidget(self.summary_lbl)

        self.result_table = QTableWidget(0, 7)
        self.result_table.setHorizontalHeaderLabels(
            ["化合物", "气溶胶类型", "核素", "AMAD/粒径", "e (Sv/Bq)", "活度贡献 (Bq/m³)", "剂量 (Sv)"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        top_dose_lay.addWidget(self.result_table, 1)
        dose_splitter.addWidget(top_dose_widget)

        # ── 下半：累计表 + 导出按钮 ──
        bot_dose_widget = QWidget()
        bot_dose_lay = QVBoxLayout(bot_dose_widget)
        bot_dose_lay.setContentsMargins(0, 0, 0, 0)
        bot_dose_lay.setSpacing(4)

        accum_lbl_hdr = QLabel("累计剂量列表（可多次计算叠加）")
        accum_lbl_hdr.setFont(QFont("微软雅黑", 10, QFont.Bold))
        bot_dose_lay.addWidget(accum_lbl_hdr)
        self.accum_total_lbl = QLabel("")
        self.accum_total_lbl.setStyleSheet("color:#1a5276; font-weight:bold;")
        bot_dose_lay.addWidget(self.accum_total_lbl)

        self.accum_table = QTableWidget(0, 5)
        self.accum_table.setHorizontalHeaderLabels(
            ["方法", "化合物/核素", "AMAD (μm)", "口罩防护因子", "剂量 (Sv)"])
        self.accum_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.accum_table.setAlternatingRowColors(True)
        self.accum_table.setEditTriggers(QTableWidget.NoEditTriggers)
        bot_dose_lay.addWidget(self.accum_table, 1)

        accum_btn_row = QHBoxLayout()
        btn_add_accum = QPushButton("➕ 将本次结果加入累计")
        btn_clear_accum = QPushButton("🗑 清空累计")
        btn_add_accum.clicked.connect(self._add_to_accum)
        btn_clear_accum.clicked.connect(self._clear_accum)
        accum_btn_row.addWidget(btn_add_accum)
        accum_btn_row.addWidget(btn_clear_accum)
        accum_btn_row.addStretch()
        bot_dose_lay.addLayout(accum_btn_row)

        # ─ ⑤ 导出结果 ─
        grp_export = QGroupBox("⑤ 导出结果")
        grp_export.setFont(QFont("微软雅黑", 10, QFont.Bold))
        exp_lay = QHBoxLayout(grp_export)
        exp_lay.setSpacing(8)

        self.btn_export_xlsx = QPushButton("📥 导出 Excel 完整报告")
        self.btn_export_xlsx.setMinimumHeight(34)
        self.btn_export_xlsx.setStyleSheet(
            "QPushButton { background:#2980b9; color:white; border-radius:4px; padding:3px 14px; }"
            "QPushButton:hover { background:#3498db; }")
        self.btn_export_xlsx.clicked.connect(self._export_xlsx)

        self.btn_export_csv = QPushButton("📄 导出 CSV 结果表")
        self.btn_export_csv.setMinimumHeight(34)
        self.btn_export_csv.setStyleSheet(
            "QPushButton { background:#27ae60; color:white; border-radius:4px; padding:3px 14px; }"
            "QPushButton:hover { background:#2ecc71; }")
        self.btn_export_csv.clicked.connect(self._export_csv)

        self.btn_export_log = QPushButton("📝 导出计算日志 (txt)")
        self.btn_export_log.setMinimumHeight(34)
        self.btn_export_log.setStyleSheet(
            "QPushButton { background:#8e44ad; color:white; border-radius:4px; padding:3px 14px; }"
            "QPushButton:hover { background:#9b59b6; }")
        self.btn_export_log.clicked.connect(self._export_log)

        self.btn_export_all = QPushButton("📦 导出全部（含图）")
        self.btn_export_all.setMinimumHeight(34)
        self.btn_export_all.setStyleSheet(
            "QPushButton { background:#c0392b; color:white; border-radius:4px; padding:3px 14px; }"
            "QPushButton:hover { background:#e74c3c; }")
        self.btn_export_all.clicked.connect(self._export_all)

        exp_lay.addWidget(self.btn_export_xlsx)
        exp_lay.addWidget(self.btn_export_csv)
        exp_lay.addWidget(self.btn_export_log)
        exp_lay.addWidget(self.btn_export_all)
        exp_lay.addStretch()
        bot_dose_lay.addWidget(grp_export)

        dose_splitter.addWidget(bot_dose_widget)
        dose_splitter.setSizes([320, 250])
        dose_splitter.setCollapsible(0, False)
        dose_splitter.setCollapsible(1, False)

        tdl.addWidget(dose_splitter, 1)
        result_tabs.addTab(tab_dose, "💊 剂量结果")

        # ── Tab 3: 计算日志 ──
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        result_tabs.addTab(self.log_edit, "📋 计算日志")

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 — 请在左侧填写参数后点击【开始计算】")

    # ─────────────────────────────────────────────────────────
    # 数据扫描 & 初始化
    # ─────────────────────────────────────────────────────────
    def _scan_data(self):
        self.status_bar.showMessage("正在扫描核素数据文件…")
        scan_nuclides()
        elems = sorted(_element_nuclides_map.keys())
        self.elem_combo.clear()
        self.elem_combo.addItems(elems)
        if elems:
            self._on_elem_changed(elems[0])
        n_nuclides = sum(len(v) for v in _element_nuclides_map.values())
        self.status_bar.showMessage(
            f"已加载 {len(elems)} 个元素 / {n_nuclides} 种核素  |  就绪")

    # ─────────────────────────────────────────────────────────
    # 交互回调
    # ─────────────────────────────────────────────────────────
    def _on_nuc_mode_changed(self):
        is_auto = self.rb_nuc_auto.isChecked()
        self.nuc_auto_hint.setVisible(is_auto)
        self.nuc_manual_hint.setVisible(not is_auto)
        # 切换到自动模式时，若已有采样数据则立即刷新
        if is_auto and self._file_df is not None:
            self._on_sid_changed(self.sid_combo.currentText())

    def _on_input_mode_changed(self):
        manual = self.rb_manual.isChecked()
        self.file_row.setVisible(not manual)
        for sp in self.conc_spins:
            sp.setEnabled(manual)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "数据文件 (*.csv *.xlsx *.xls)")
        if not path:
            return
        self.file_path_edit.setText(path)
        try:
            self._file_df = pd.read_csv(path) if path.endswith(
                '.csv') else pd.read_excel(path)
            ws_list = sorted(self._file_df['Workshop'].unique())
            self.ws_combo.blockSignals(True)
            self.ws_combo.clear()
            self.ws_combo.addItems([str(w) for w in ws_list])
            self.ws_combo.blockSignals(False)
            if ws_list:
                self._on_ws_changed(str(ws_list[0]))
        except Exception as e:
            QMessageBox.warning(self, "读取失败", f"文件读取失败：{e}")

    def _on_ws_changed(self, ws):
        if self._file_df is None:
            return
        df_w = self._file_df[self._file_df['Workshop'].astype(str) == ws]
        sids = sorted(df_w['SamplingID'].unique())
        self.sid_combo.blockSignals(True)
        self.sid_combo.clear()
        self.sid_combo.addItems([str(s) for s in sids])
        self.sid_combo.blockSignals(False)
        if sids:
            self._on_sid_changed(str(sids[0]))

    def _on_sid_changed(self, sid):
        if self._file_df is None:
            return
        ws = self.ws_combo.currentText()
        mask = ((self._file_df['Workshop'].astype(str) == ws) &
                (self._file_df['SamplingID'].astype(str) == sid))
        rows = self._file_df[mask]
        if rows.empty:
            return
        row = rows.iloc[0]
        for i, col in enumerate(CONC_COLS):
            if col in row.index:
                try:
                    self.conc_spins[i].setValue(float(row[col]))
                except Exception:
                    self.conc_spins[i].setValue(0.0)
        # 仅在"自动从采样数据读取"模式下才自动填充化合物配置
        if self.rb_nuc_auto.isChecked():
            if 'nuclides' in row.index and pd.notna(row.get('nuclides', '')):
                self._fill_compounds_from_row(row)

    def _fill_compounds_from_row(self, row):
        try:
            comps = str(row.get('compounds', '')).split(';')
            fracs = str(row.get('activity_fractions', '')).split(';')
            abunds = str(row.get('nuclide_abundances', '')).split(';')
            # 直接清空，绕过"至少保留一个"保护（此处是程序内部触发，非用户手动删除）
            for w in list(self._compound_widgets):
                self._comp_lay.removeWidget(w)
                w.deleteLater()
            self._compound_widgets.clear()
            for i in range(len(comps)):
                self._add_compound_row()
                w = self._compound_widgets[-1]
                w.compound_edit.setText(comps[i].strip())
                try:
                    w.act_frac_spin.setValue(float(fracs[i]))
                except Exception:
                    pass
                if i < len(abunds):
                    w.nuclides_edit.setText(abunds[i].strip())
        except Exception as e:
            self._log(f"[警告] 自动填充化合物失败: {e}")

    def _on_elem_changed(self, elem):
        nucs = _element_nuclides_map.get(elem, [])
        opts = set()
        for nuc in nucs:
            df = get_nuclide_df(nuc)
            if df is not None and 'aerosol_type' in df.columns:
                opts.update(df['aerosol_type'].dropna().unique())
        opts_sorted = sorted(opts)
        for w in self._compound_widgets:
            w.update_aerosol_options(opts_sorted)
        self._current_elem = elem

    def _add_compound_row(self):
        w = NuclideCompoundWidget()
        w.del_btn.clicked.connect(lambda: self._remove_compound_row(w))
        elem = self.elem_combo.currentText()
        nucs = _element_nuclides_map.get(elem, [])
        opts = set()
        for nuc in nucs:
            df = get_nuclide_df(nuc)
            if df is not None and 'aerosol_type' in df.columns:
                opts.update(df['aerosol_type'].dropna().unique())
        w.update_aerosol_options(sorted(opts))
        self._compound_widgets.append(w)
        # 插入在 stretch 之前
        insert_pos = self._comp_lay.count() - 1
        self._comp_lay.insertWidget(insert_pos, w)

    def _remove_compound_row(self, w):
        if len(self._compound_widgets) <= 1:
            QMessageBox.information(self, "提示", "至少保留一个化合物行")
            return
        self._compound_widgets.remove(w)
        self._comp_lay.removeWidget(w)
        w.deleteLater()

    def _update_mask_info(self):
        mtype = self.mask_combo.currentText()
        p = MASK_PARAMS.get(mtype, {})
        eff = p.get('filtration_efficiency', 0)
        leak = p.get('leakage_rate', 1)
        self.mask_info_lbl.setText(f"过滤效率 {eff:.0%}  泄漏率 {leak:.0%}")

    def _update_method_hints(self):
        for rb, hl in self._hint_lbls.items():
            if rb.isChecked():
                hl.setStyleSheet(
                    "color:#1a5276; font-size:10px; font-weight:bold; margin-left:16px;")
            else:
                hl.setStyleSheet(
                    "color:#999; font-size:10px; margin-left:16px;")
        # 峰型子选项：仅多模态法激活时显示
        self._modal_peak_widget.setVisible(self.rb_modal.isChecked())

    # ─────────────────────────────────────────────────────────
    # 手动拟合（选好数据后点击「运行拟合」按钮触发）
    # ─────────────────────────────────────────────────────────
    def _on_fit(self):
        """手动触发：运行三种拟合并更新图表"""
        raw_concs = np.array([sp.value()
                             for sp in self.conc_spins], dtype=float)
        if np.sum(raw_concs) == 0:
            return

        try:
            eff_corr = apply_efficiency_correction(raw_concs)
            total_corr = float(np.sum(eff_corr))

            # 数据质量评估
            quality = data_quality_assessment(raw_concs)

            amad_uni, gsd_uni, total_uni = fit_unimodal(eff_corr)
            amad1, gsd1, frac1, amad2, gsd2, total_bi = fit_bimodal(eff_corr)
            D50_lin, GSD_lin, D84_lin, D16_lin, R2_lin = fit_linear_probit(
                raw_concs)

            # AIC/BIC 计算
            aic_uni, bic_uni = calc_unimodal_stats(
                amad_uni, gsd_uni, total_uni, eff_corr)
            aic_bi, bic_bi = (np.nan, np.nan)
            if not np.isnan(amad2):
                aic_bi, bic_bi = calc_bimodal_stats(
                    amad1, gsd1, frac1, amad2, gsd2, total_bi, eff_corr)
            # probit 的 AIC/BIC（独立于上面的 fit_linear_probit，额外算 stats）
            D50_pb, GSD_pb, R2_pb, aic_pb, bic_pb, n_pb = calc_probit_stats(
                raw_concs)

            ratio = amad1 / \
                amad2 if not np.isnan(amad2) and amad2 > 0 else np.nan
            recommended = recommend_method(
                amad2, ratio, frac1, R2_lin,
                aic_uni=aic_uni, bic_uni=bic_uni,
                aic_bi=aic_bi, bic_bi=bic_bi,
                quality_flag=quality['quality_flag'],
                n_nonzero=quality['n_nonzero'],
            )

            # 存储临时拟合结果
            self._fit_auto = {
                'raw_concs': raw_concs,
                'eff_corr': eff_corr,
                'total_corr': total_corr,
                'amad_uni': amad_uni, 'gsd_uni': gsd_uni, 'total_uni': total_uni,
                'amad1': amad1, 'gsd1': gsd1, 'frac1': frac1,
                'amad2': amad2, 'gsd2': gsd2,
                'D50_lin': D50_lin, 'GSD_lin': GSD_lin, 'R2_lin': R2_lin,
                'recommended': recommended,
                'aic_uni': aic_uni, 'bic_uni': bic_uni,
                'aic_bi': aic_bi, 'bic_bi': bic_bi,
                'aic_probit': aic_pb, 'bic_probit': bic_pb, 'n_probit': n_pb,
                'quality': quality,
            }

            # 画图（此处为防护前数据，未加口罩校正）
            self.plot_canvas.plot_fitting(
                raw_concs, eff_corr,
                amad_uni, gsd_uni, total_uni,
                amad1, gsd1, frac1, amad2, gsd2, total_bi, total_corr,
                D50_lin, GSD_lin, R2_lin,
                is_corrected=False
            )

            # 更新参数信息标签（含 AIC/BIC）
            parts = [f"【推荐方法】{recommended}"]
            parts.append(f"单峰 AMAD={amad_uni:.3f} \u03bcm  GSD={gsd_uni:.3f}"
                         f"  AIC={aic_uni:.1f}  BIC={bic_uni:.1f}")
            if not np.isnan(amad2):
                parts.append(f"双峰：粗峰={amad1:.3f} \u03bcm  细峰={amad2:.3f} \u03bcm  粗峰={frac1:.2%}"
                             f"  AIC={aic_bi:.1f}  BIC={bic_bi:.1f}")
            if not np.isnan(D50_lin):
                parts.append(f"正态概率图 AMAD={D50_lin:.3f} \u03bcm  GSD={GSD_lin:.3f}  R\u00b2={R2_lin:.4f}"
                             f"  AIC={aic_pb:.1f}")
            parts.append(
                f"数据质量: {quality['quality_flag']}（非零级数={quality['n_nonzero']}/9）")
            self.fit_info_lbl.setText("  \n".join(parts))

            self.log_edit.append(f"[拟合] 单峰 AMAD={amad_uni:.3f}  AIC={aic_uni:.1f} | "
                                 f"双峰粗峰={amad1:.3f}  AIC={aic_bi:.1f} | "
                                 f"正态概率图 D50={D50_lin:.3f}  R²={R2_lin:.4f} | "
                                 f"数据质量={quality['quality_flag']}")
            self.status_bar.showMessage(f"✓ 拟合完成 | 推荐: {recommended}", 5000)

        except Exception as e:
            import traceback
            self.log_edit.append(f"[拟合失败] {traceback.format_exc()}")

    # ─────────────────────────────────────────────────────────
    # 核心计算入口
    # ─────────────────────────────────────────────────────────
    def _on_calc(self):
        raw_concs = np.array([sp.value()
                             for sp in self.conc_spins], dtype=float)
        if np.sum(raw_concs) == 0:
            QMessageBox.warning(self, "数据为空", "所有级别浓度均为零，请先输入数据。")
            return

        self.calc_btn.setEnabled(False)
        self.status_bar.showMessage("计算中，请稍候…")
        self.log_edit.clear()

        mask_type = self.mask_combo.currentText()
        br = self.br_spin.value()
        work_hours = self.work_spin.value()
        elem = self.elem_combo.currentText()
        nucs_elem = _element_nuclides_map.get(elem, [])

        compounds_data = [w.get_data() for w in self._compound_widgets]
        total_frac = sum(c['activity_fraction'] for c in compounds_data)
        if total_frac > 0:
            for c in compounds_data:
                c['activity_fraction'] /= total_frac

        method = ('std' if self.rb_std.isChecked() else
                  'modal' if self.rb_modal.isChecked() else
                  'probit' if self.rb_probit.isChecked() else 'stage')
        # 多模态法峰型：由用户手动选择（单峰/双峰）
        user_peak = 'bimodal' if self.rb_peak_bi.isChecked() else 'unimodal'

        self._calc_thread = CalcThread(
            self._do_calc,
            raw_concs, mask_type, br, work_hours,
            compounds_data, method, nucs_elem, user_peak
        )
        self._calc_thread.result_ready.connect(self._on_result)
        self._calc_thread.error_signal.connect(self._on_error)
        self._calc_thread.start()

    def _do_calc(self, raw_concs, mask_type, br, work_hours, compounds_data, method, nucs_elem, user_peak='unimodal'):
        log_lines = []
        def log(s): log_lines.append(s)

        log("===== 计算开始 =====")
        log(f"方法: {method}  口罩: {mask_type}  BR={br} m³/h  T={work_hours} h")

        # 口罩校正（逐级）
        corrected_concs = np.array([
            raw_concs[i] * calc_mask_pf(mask_type, midpoints[i])
            for i in range(len(raw_concs))
        ], dtype=float)
        total_raw = float(np.sum(raw_concs))
        total_masked = float(np.sum(corrected_concs))
        mask_pf_overall = total_masked / total_raw if total_raw > 0 else 1.0
        log(f"原始总浓度: {total_raw:.4e} Bq/m³  校正后: {total_masked:.4e} Bq/m³  防护因子: {mask_pf_overall:.4f}")

        # AMAD 拟合
        eff_corr = apply_efficiency_correction(corrected_concs)
        total_corr = float(np.sum(eff_corr))

        if method == 'std':
            fit_res = {
                'method': 'std', 'amad': 5.0, 'gsd': 1.5,
                'amad_uni': 5.0, 'gsd_uni': 1.5, 'total_uni': total_corr,
                'amad1': 5.0, 'gsd1': 1.5, 'frac1': 1.0,
                'amad2': np.nan, 'gsd2': np.nan,
                'D50_lin': np.nan, 'GSD_lin': np.nan, 'R2_lin': np.nan,
                'recommended': '国标单AMAD (5 μm)',
            }
            log("使用国标法：AMAD = 5 μm（固定）")

        elif method == 'modal':
            # 数据质量评估
            quality = data_quality_assessment(corrected_concs)

            log("拟合单峰…")
            amad_uni, gsd_uni, total_uni = fit_unimodal(eff_corr)
            log(f"  单峰: AMAD={amad_uni:.3f} μm  GSD={gsd_uni:.3f}")
            log("拟合双峰…")
            amad1, gsd1, frac1, amad2, gsd2, total_bi = fit_bimodal(eff_corr)
            log(f"  双峰: 粗峰={amad1:.3f} μm  细峰={amad2:.3f} μm  粗峰占比={frac1:.3f}")
            log("正态概率图拟合…")
            D50_lin, GSD_lin, D84_lin, D16_lin, R2_lin = fit_linear_probit(
                corrected_concs)
            log(f"  正态概率图: AMAD={D50_lin:.3f} μm  GSD={GSD_lin:.3f}  R²={R2_lin:.4f}")

            # AIC/BIC 计算
            aic_uni, bic_uni = calc_unimodal_stats(
                amad_uni, gsd_uni, total_uni, eff_corr)
            aic_bi, bic_bi = (np.nan, np.nan)
            if not np.isnan(amad2):
                aic_bi, bic_bi = calc_bimodal_stats(
                    amad1, gsd1, frac1, amad2, gsd2, total_bi, eff_corr)
            log(f"  AIC: 单峰={aic_uni:.1f}  双峰={aic_bi:.1f}  |  BIC: 单峰={bic_uni:.1f}  双峰={bic_bi:.1f}")
            log(f"  数据质量: {quality['quality_flag']}（非零级数={quality['n_nonzero']}/9）")

            ratio = amad1 / \
                amad2 if not np.isnan(amad2) and amad2 > 0 else np.nan
            recommended = recommend_method(
                amad2, ratio, frac1, R2_lin,
                aic_uni=aic_uni, bic_uni=bic_uni,
                aic_bi=aic_bi, bic_bi=bic_bi,
                quality_flag=quality['quality_flag'],
                n_nonzero=quality['n_nonzero'],
            )
            log(f"参考推荐方法（仅供参考，实际使用用户选定峰型）: {recommended}")
            log(f"用户选定峰型: {'双峰' if user_peak == 'bimodal' else '单峰'}")

            D50_pb, GSD_pb, R2_pb, aic_pb, bic_pb, n_pb = calc_probit_stats(
                corrected_concs)

            fit_res = {
                'method': 'modal', 'recommended': recommended,
                'user_peak': user_peak,
                'amad_uni': amad_uni, 'gsd_uni': gsd_uni, 'total_uni': total_uni,
                'amad1': amad1, 'gsd1': gsd1, 'frac1': frac1,
                'amad2': amad2, 'gsd2': gsd2,
                'D50_lin': D50_lin, 'GSD_lin': GSD_lin, 'R2_lin': R2_lin,
                'aic_uni': aic_uni, 'bic_uni': bic_uni,
                'aic_bi': aic_bi, 'bic_bi': bic_bi,
                'aic_probit': aic_pb, 'bic_probit': bic_pb, 'n_probit': n_pb,
                'quality': quality,
            }
            # 用用户选定的峰型决定 amad/gsd，不自动判断
            if user_peak == 'bimodal' and not np.isnan(amad2):
                fit_res['amad'] = amad1
                fit_res['gsd'] = gsd1
                log(f"  [双峰模式] 使用粗峰 AMAD={amad1:.3f} μm 作为代表粒径")
            elif user_peak == 'bimodal' and np.isnan(amad2):
                log("  [警告] 用户选择双峰但拟合未检测到双峰，回退使用单峰 AMAD")
                fit_res['amad'] = amad_uni
                fit_res['gsd'] = gsd_uni
            else:
                # 单峰模式
                fit_res['amad'] = amad_uni
                fit_res['gsd'] = gsd_uni
                log(f"  [单峰模式] 使用单峰 AMAD={amad_uni:.3f} μm")

        elif method == 'probit':
            D50_lin, GSD_lin, D84_lin, D16_lin, R2_lin = fit_linear_probit(
                corrected_concs)
            log(f"正态概率图: AMAD={D50_lin:.3f} μm  GSD={GSD_lin:.3f}  R²={R2_lin:.4f}")
            if np.isnan(D50_lin):
                raise ValueError("正态概率图拟合失败（有效数据点 < 3），请检查浓度数据。")
            amad_uni, gsd_uni, total_uni = fit_unimodal(eff_corr)
            fit_res = {
                'method': 'probit', 'amad': D50_lin, 'gsd': GSD_lin,
                'amad_uni': amad_uni, 'gsd_uni': gsd_uni, 'total_uni': total_uni,
                'amad1': D50_lin, 'gsd1': GSD_lin, 'frac1': 1.0,
                'amad2': np.nan, 'gsd2': np.nan,
                'D50_lin': D50_lin, 'GSD_lin': GSD_lin, 'R2_lin': R2_lin,
                'recommended': f'正态概率法 (R²={R2_lin:.4f})',
            }

        else:  # stage
            amad_uni, gsd_uni, total_uni = fit_unimodal(eff_corr)
            fit_res = {
                'method': 'stage',
                'amad_uni': amad_uni, 'gsd_uni': gsd_uni, 'total_uni': total_uni,
                'amad1': amad_uni, 'gsd1': gsd_uni, 'frac1': 1.0,
                'amad2': np.nan, 'gsd2': np.nan,
                'D50_lin': np.nan, 'GSD_lin': np.nan, 'R2_lin': np.nan,
                'recommended': '逐级独立法',
            }
            log("逐级独立法：各级独立使用中值粒径查表")

        # ─ 剂量计算 ─
        detail_rows = []
        total_dose = 0.0

        for comp_data in compounds_data:
            compound = comp_data['compound']
            aerosol_type = comp_data['aerosol_type']
            act_frac = comp_data['activity_fraction']
            nuclides_abu = comp_data['nuclides']
            log(f"\n--- 化合物: {compound}  气溶胶: {aerosol_type}  活度比例: {act_frac:.3f} ---")

            if not nuclides_abu:
                log("  [警告] 未配置核素丰度，跳过")
                continue

            for nuc, abundance in nuclides_abu.items():
                nuc_clean = nuc.strip()
                log(f"  核素: {nuc_clean}  丰度: {abundance:.4f}")
                df_nuc = get_nuclide_df(nuc_clean)
                if df_nuc is None or df_nuc.empty:
                    log(f"  [警告] 找不到核素 {nuc_clean} 数据，跳过")
                    continue

                # 1) 按化合物全名筛选；同时获取 parquet 内绑定的气溶胶类型
                full_compound, inferred_aerosol = resolve_compound(compound)
                # UI 下拉选的类型优先，但如果映射表有明确绑定值则覆盖
                effective_aerosol = inferred_aerosol if inferred_aerosol else aerosol_type

                sub = df_nuc.copy()
                if 'compound' in df_nuc.columns:
                    sub_cp = df_nuc[df_nuc['compound'] == full_compound]
                    if not sub_cp.empty:
                        sub = sub_cp
                        log(f"  化合物: {compound} → {full_compound} (气溶胶: {effective_aerosol})")
                    else:
                        log(f"  [提示] 未在 parquet 中找到化合物 '{full_compound}'，使用全核素数据")
                # 2) 按气溶胶类型筛选
                if 'aerosol_type' in sub.columns:
                    sub_at = sub[sub['aerosol_type'] == effective_aerosol]
                    if sub_at.empty:
                        log(f"  [提示] {nuc_clean} ({effective_aerosol}) 无匹配行，回退 Unspecified")
                        sub_at = sub[sub['aerosol_type'] == 'Unspecified']
                    if sub_at.empty:
                        log(f"  [警告] {nuc_clean} 无可用剂量系数，跳过")
                        continue
                    sub = sub_at

                if method == 'stage':
                    dose_nuc = 0.0
                    for si in range(len(STAGE_NAMES)):
                        conc_si = corrected_concs[si] * act_frac * abundance
                        dp_mid = midpoints[si]
                        e_val = interp_dose_coeff(
                            sub, effective_aerosol, dp_mid)
                        if e_val is None:
                            e_val = interp_dose_coeff(
                                sub, 'Unspecified', dp_mid)
                        if e_val is None:
                            continue
                        d_si = e_val * conc_si * br * work_hours
                        dose_nuc += d_si
                        log(
                            f"    级{STAGE_NAMES[si]}({dp_mid:.2f}μm) e={e_val:.2e} C={conc_si:.3e} D={d_si:.3e}")
                    log(f"  逐级合计: {dose_nuc:.3e} Sv")
                    detail_rows.append({
                        'compound': compound, 'aerosol_type': effective_aerosol,
                        'nuclide': nuc_clean, 'amad': '逐级',
                        'e_val': np.nan,
                        'act_conc': total_masked * act_frac * abundance,
                        'dose': dose_nuc,
                    })
                    total_dose += dose_nuc

                elif method == 'modal' and fit_res.get('user_peak') == 'bimodal' and not np.isnan(fit_res.get('amad2', np.nan)):
                    amad1 = fit_res['amad1']
                    gsd1 = fit_res['gsd1']
                    frac1 = fit_res['frac1']
                    amad2 = fit_res['amad2']
                    frac2 = 1 - frac1

                    pk1_integ = np.array([
                        stage_integral(
                            amad1, gsd1, stages_low[i], stages_high[i])
                        for i in range(len(STAGE_NAMES))
                    ])
                    pk2_integ = np.array([
                        stage_integral(
                            amad2, fit_res['gsd2'], stages_low[i], stages_high[i])
                        for i in range(len(STAGE_NAMES))
                    ])
                    s1 = float(np.sum(pk1_integ))
                    s2 = float(np.sum(pk2_integ))
                    sh1 = pk1_integ / \
                        s1 if s1 > 0 else np.ones(
                            len(STAGE_NAMES)) / len(STAGE_NAMES)
                    sh2 = pk2_integ / \
                        s2 if s2 > 0 else np.ones(
                            len(STAGE_NAMES)) / len(STAGE_NAMES)

                    c1 = float(np.sum(corrected_concs * sh1)) * \
                        frac1 * act_frac * abundance
                    c2 = float(np.sum(corrected_concs * sh2)) * \
                        frac2 * act_frac * abundance

                    e1 = interp_dose_coeff(sub, effective_aerosol, amad1) or interp_dose_coeff(
                        sub, 'Unspecified', amad1)
                    e2 = interp_dose_coeff(sub, effective_aerosol, amad2) or interp_dose_coeff(
                        sub, 'Unspecified', amad2)

                    d1 = (e1 * c1 * br * work_hours) if e1 else 0.0
                    d2 = (e2 * c2 * br * work_hours) if e2 else 0.0
                    dose_nuc = d1 + d2
                    e1s = f"{e1:.2e}" if e1 else "N/A"
                    e2s = f"{e2:.2e}" if e2 else "N/A"
                    log(f"  双峰峰1(AMAD={amad1:.2f}μm) C={c1:.3e} e={e1s} D={d1:.3e} Sv")
                    log(f"  双峰峰2(AMAD={amad2:.2f}μm) C={c2:.3e} e={e2s} D={d2:.3e} Sv")
                    log(f"  合计: {dose_nuc:.3e} Sv")
                    detail_rows.append({
                        'compound': compound, 'aerosol_type': effective_aerosol,
                        'nuclide': nuc_clean,
                        'amad': f"峰1:{amad1:.2f} / 峰2:{amad2:.2f}",
                        'e_val': np.nan, 'act_conc': c1 + c2, 'dose': dose_nuc,
                    })
                    total_dose += dose_nuc

                else:
                    amad_use = fit_res.get('amad', 5.0)
                    gsd_use = fit_res.get('gsd', 1.5)
                    pk_integ = np.array([
                        stage_integral(amad_use, gsd_use,
                                       stages_low[i], stages_high[i])
                        for i in range(len(STAGE_NAMES))
                    ])
                    s_integ = float(np.sum(pk_integ))
                    pk_share = pk_integ / \
                        s_integ if s_integ > 0 else np.ones(
                            len(STAGE_NAMES)) / len(STAGE_NAMES)
                    c_equiv = float(
                        np.sum(corrected_concs * pk_share)) * act_frac * abundance

                    e_val = interp_dose_coeff(sub, effective_aerosol, amad_use)
                    if e_val is None:
                        e_val = interp_dose_coeff(sub, 'Unspecified', amad_use)
                    if e_val is None:
                        log(f"  [警告] 无法获取剂量系数，跳过")
                        continue

                    dose_nuc = e_val * c_equiv * br * work_hours
                    log(f"  AMAD={amad_use:.2f}μm  e={e_val:.3e}  C={c_equiv:.3e}  D={dose_nuc:.3e} Sv")
                    detail_rows.append({
                        'compound': compound, 'aerosol_type': effective_aerosol,
                        'nuclide': nuc_clean, 'amad': f"{amad_use:.2f}",
                        'e_val': e_val, 'act_conc': c_equiv, 'dose': dose_nuc,
                    })
                    total_dose += dose_nuc

        log(f"\n===== 总有效剂量 = {total_dose:.4e} Sv =====")
        log(f"BR = {br} m³/h  T = {work_hours} h")

        return {
            'fit_res': fit_res,
            'raw_concs': raw_concs,
            'corrected_concs': corrected_concs,
            'eff_corr': eff_corr,
            'total_corr': total_corr,
            'total_raw': total_raw,
            'total_masked': total_masked,
            'mask_pf_overall': mask_pf_overall,
            'method': method,
            'detail_rows': detail_rows,
            'total_dose': total_dose,
            'log': '\n'.join(log_lines),
            'br': br, 'work_hours': work_hours,
            'mask_type': mask_type,
        }

    def _on_result(self, res):
        self._fit_result = res
        fit = res['fit_res']

        # 日志
        self.log_edit.setPlainText(res['log'])

        # 拟合图（计算结果使用防护后校正数据绘制）
        try:
            self.plot_canvas.plot_fitting(
                res['raw_concs'], res['eff_corr'],
                fit.get('amad_uni', 5.0), fit.get(
                    'gsd_uni', 1.5), fit.get('total_uni', 1.0),
                fit.get('amad1', 5.0), fit.get(
                    'gsd1', 1.5), fit.get('frac1', 1.0),
                fit.get('amad2', np.nan), fit.get('gsd2', np.nan),
                fit.get('total_uni', 1.0), res['total_corr'],
                fit.get('D50_lin', np.nan), fit.get(
                    'GSD_lin', np.nan), fit.get('R2_lin', np.nan),
                is_corrected=True
            )
        except Exception as e:
            self._log(f"[图] 绘图失败: {e}")

        # 拟合参数信息
        rec = fit.get('recommended', '')
        user_peak = fit.get('user_peak', '')
        peak_label = ''
        if res['method'] == 'modal':
            if user_peak == 'bimodal':
                peak_label = '【双峰模式（用户选定）】'
            else:
                peak_label = '【单峰模式（用户选定）】'
        parts = [f"【推荐/参考方法】{rec}  {peak_label}"]
        amad_uni = fit.get('amad_uni', 5)
        parts.append(f"单峰 AMAD={amad_uni:.3f} \u03bcm  GSD={fit.get('gsd_uni', 1.5):.3f}"
                     f"  AIC={fit.get('aic_uni', np.nan):.1f}  BIC={fit.get('bic_uni', np.nan):.1f}")
        if not np.isnan(fit.get('amad2', np.nan)):
            parts.append(f"双峰：粗峰={fit.get('amad1', 0):.3f} \u03bcm  细峰={fit.get('amad2', 0):.3f} \u03bcm  "
                         f"粗峰占比={fit.get('frac1', 0):.2%}"
                         f"  AIC={fit.get('aic_bi', np.nan):.1f}  BIC={fit.get('bic_bi', np.nan):.1f}")
        D50_lin = fit.get('D50_lin', np.nan)
        if not np.isnan(D50_lin):
            parts.append(f"正态概率图 AMAD={D50_lin:.3f} \u03bcm  GSD={fit.get('GSD_lin', 0):.3f}  "
                         f"R\u00b2={fit.get('R2_lin', 0):.4f}"
                         f"  AIC={fit.get('aic_probit', np.nan):.1f}")
        q = fit.get('quality', {})
        if q:
            parts.append(f"数据质量: {q.get('quality_flag', '?')}（非零级数={q.get('n_nonzero', '?')}/9"
                         f"  总活度={q.get('total_activity', 0):.3e} Bq/m³）")
        self.fit_info_lbl.setText("  \n".join(parts))

        # 剂量结果表
        self.result_table.setRowCount(0)
        for row in res['detail_rows']:
            r = self.result_table.rowCount()
            self.result_table.insertRow(r)
            e_s = f"{row['e_val']:.3e}" if not np.isnan(
                row.get('e_val', np.nan)) else "逐级"
            vals = [row['compound'], row['aerosol_type'], row['nuclide'],
                    str(row['amad']), e_s,
                    f"{row['act_conc']:.4e}", f"{row['dose']:.4e}"]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                self.result_table.setItem(r, c, item)

        # 汇总标签
        td = res['total_dose']
        pf = res['mask_pf_overall']
        mn_base = {'std': '国标单AMAD法', 'modal': '多模态拟合法',
                   'probit': '正态概率图法', 'stage': '逐级独立法'}.get(res['method'], res['method'])
        if res['method'] == 'modal':
            peak_suffix = '（双峰）' if fit.get('user_peak') == 'bimodal' else '（单峰）'
            mn = mn_base + peak_suffix
        else:
            mn = mn_base
        self.summary_lbl.setText(
            f"【{mn}】  总有效剂量 = {td:.4e} Sv"
            f"  |  口罩防护因子 = {pf:.4f}"
            f"  |  BR = {res['br']:.1f} m³/h  T = {res['work_hours']:.1f} h"
        )

        self.calc_btn.setEnabled(True)
        self.status_bar.showMessage(f"✓ 计算完成  总剂量 = {td:.4e} Sv")

    def _on_error(self, msg):
        QMessageBox.critical(self, "计算错误", f"计算过程中发生错误：\n\n{msg}")
        self.calc_btn.setEnabled(True)
        self.status_bar.showMessage("计算失败，请检查参数")

    def _clear_results(self):
        self.result_table.setRowCount(0)
        self.summary_lbl.setText("剂量结果将在计算后显示")
        self.fit_info_lbl.setText("拟合参数将在计算后显示")
        self.log_edit.clear()
        self.plot_canvas.fig.clear()
        self.plot_canvas.draw()

    def _add_to_accum(self):
        if self._fit_result is None:
            QMessageBox.information(self, "提示", "请先完成一次计算")
            return
        res = self._fit_result
        mn = {'std': '国标法', 'modal': '多模态', 'probit': '正态概率',
              'stage': '逐级'}.get(res['method'], res['method'])
        comps = ','.join(sorted({r['compound'] for r in res['detail_rows']}))
        nucs = ','.join(sorted({r['nuclide'] for r in res['detail_rows']}))
        amad_s = str(res['fit_res'].get('amad', '逐级'))
        pf_s = f"{res['mask_pf_overall']:.4f}"
        r = self.accum_table.rowCount()
        self.accum_table.insertRow(r)
        for c, v in enumerate([mn, f"{comps}/{nucs}", amad_s, pf_s, f"{res['total_dose']:.4e}"]):
            item = QTableWidgetItem(v)
            item.setTextAlignment(Qt.AlignCenter)
            self.accum_table.setItem(r, c, item)
        total_accum = sum(
            float(self.accum_table.item(i, 4).text())
            for i in range(self.accum_table.rowCount())
        )
        self.accum_total_lbl.setText(f"累计总剂量 = {total_accum:.4e} Sv")

    def _clear_accum(self):
        self.accum_table.setRowCount(0)
        self.accum_total_lbl.setText("")

    # ─────────────────────────────────────────────────────────
    # 导出功能
    # ─────────────────────────────────────────────────────────
    def _check_result(self):
        if self._fit_result is None:
            QMessageBox.information(self, "提示", "请先完成一次计算再导出。")
            return False
        return True

    def _build_export_data(self):
        """构建包含所有计算信息的字典，供各导出格式共用"""
        res = self._fit_result
        fit = res['fit_res']
        q = fit.get('quality', {})

        # ─ 输入参数 ─
        raw_concs = res['raw_concs']
        input_table = []
        for i, (name, rng) in enumerate(zip(STAGE_NAMES, STAGE_RANGES)):
            input_table.append({
                '级别': name, '粒径范围': rng,
                '原始浓度 (Bq/m³)': f"{raw_concs[i]:.6e}",
                '校正后浓度 (Bq/m³)': f"{res['corrected_concs'][i]:.6e}",
            })

        # ─ 拟合结果 ─
        fitting_table = []
        fitting_table.append({'方法': '单峰对数正态', 'AMAD (μm)': f"{fit.get('amad_uni', 0):.3f}",
                              'GSD': f"{fit.get('gsd_uni', 0):.3f}",
                              'AIC': f"{fit.get('aic_uni', np.nan):.1f}",
                              'BIC': f"{fit.get('bic_uni', np.nan):.1f}"})
        if not np.isnan(fit.get('amad2', np.nan)):
            fitting_table.append({'方法': '双峰-粗峰', 'AMAD (μm)': f"{fit.get('amad1', 0):.3f}",
                                  'GSD': f"{fit.get('gsd1', 0):.3f}",
                                  '粗峰占比': f"{fit.get('frac1', 0):.2%}",
                                  'AIC': f"{fit.get('aic_bi', np.nan):.1f}",
                                  'BIC': f"{fit.get('bic_bi', np.nan):.1f}"})
            fitting_table.append({'方法': '双峰-细峰', 'AMAD (μm)': f"{fit.get('amad2', 0):.3f}",
                                  'GSD': f"{fit.get('gsd2', 0):.3f}",
                                  'AIC': f"{fit.get('aic_bi', np.nan):.1f}",
                                  'BIC': f"{fit.get('bic_bi', np.nan):.1f}"})
        if not np.isnan(fit.get('D50_lin', np.nan)):
            fitting_table.append({'方法': '正态概率图', 'AMAD (μm)': f"{fit.get('D50_lin', 0):.3f}",
                                  'GSD': f"{fit.get('GSD_lin', 0):.3f}",
                                  'R²': f"{fit.get('R2_lin', 0):.4f}",
                                  'AIC': f"{fit.get('aic_probit', np.nan):.1f}",
                                  'BIC': f"{fit.get('bic_probit', np.nan):.1f}"})

        # ─ 剂量明细 ─
        dose_rows = res['detail_rows']

        # ─ 汇总 ─
        mn_map = {'std': '国标单AMAD法', 'modal': '多模态拟合法',
                  'probit': '正态概率图法', 'stage': '逐级独立法'}
        summary = {
            '计算方法': mn_map.get(res['method'], res['method']),
            '推荐方法': fit.get('recommended', ''),
            '口罩型号': res['mask_type'],
            '总防护因子': f"{res['mask_pf_overall']:.6f}",
            '原始总浓度 (Bq/m³)': f"{res['total_raw']:.4e}",
            '校正后总浓度 (Bq/m³)': f"{res['total_masked']:.4e}",
            '总有效剂量 (Sv)': f"{res['total_dose']:.4e}",
            '呼吸速率 (m³/h)': f"{res['br']:.2f}",
            '工作时长 (h)': f"{res['work_hours']:.1f}",
            '数据质量': q.get('quality_flag', '?'),
            '非零级数': f"{q.get('n_nonzero', '?')}/9",
        }

        return {
            'input': input_table,
            'fitting': fitting_table,
            'dose_rows': dose_rows,
            'summary': summary,
            'log': res['log'],
        }

    def _export_xlsx(self):
        if not self._check_result():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 Excel 报告", "dose_report.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
            from openpyxl.utils import get_column_letter
        except ImportError:
            QMessageBox.critical(
                self, "缺少依赖", "请安装 openpyxl: pip install openpyxl")
            return

        data = self._build_export_data()
        wb = Workbook()

        thin_border = Border(
            left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin'))
        header_fill = PatternFill(
            start_color='1A5276', end_color='1A5276', fill_type='solid')
        header_font = Font(name='微软雅黑', bold=True, color='FFFFFF', size=11)
        title_font = Font(name='微软雅黑', bold=True, size=14, color='1A5276')
        cell_font = Font(name='Consolas', size=10)
        warn_font = Font(name='微软雅黑', bold=True, size=11, color='C0392B')

        def write_table(ws, headers, rows, start_row=1, col_widths=None):
            for ci, h in enumerate(headers, 1):
                c = ws.cell(row=start_row, column=ci, value=h)
                c.font, c.fill, c.alignment, c.border = header_font, header_fill, Alignment(
                    horizontal='center'), thin_border
            for ri, row in enumerate(rows):
                for ci, val in enumerate(row.values() if isinstance(row, dict) else row, 1):
                    c = ws.cell(row=start_row + 1 + ri, column=ci, value=val)
                    c.font, c.alignment, c.border = cell_font, Alignment(
                        horizontal='center'), thin_border
            if col_widths:
                for ci, w in enumerate(col_widths, 1):
                    ws.column_dimensions[get_column_letter(ci)].width = w

        # ── Sheet 1: 汇总 ──
        ws1 = wb.active
        ws1.title = "汇总"
        ws1.cell(row=1, column=1, value="空气采样法内照射剂量计算报告").font = title_font
        ws1.merge_cells('A1:B1')
        ws1.cell(row=2, column=1, value=f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}").font = Font(
            name='微软雅黑', size=10, color='666666')
        for ri, (k, v) in enumerate(data['summary'].items(), 4):
            ws1.cell(row=ri, column=1, value=k).font = Font(
                name='微软雅黑', bold=True, size=10)
            ws1.cell(row=ri, column=2, value=v).font = cell_font
            if k == '数据质量' and 'poor' in str(v).lower():
                ws1.cell(row=ri, column=2).font = warn_font
        ws1.column_dimensions['A'].width = 24
        ws1.column_dimensions['B'].width = 32

        # ── Sheet 2: 拟合参数 ──
        ws2 = wb.create_sheet("拟合参数")
        ws2.cell(row=1, column=1, value="拟合参数与信息准则").font = title_font
        write_table(ws2, list(data['fitting'][0].keys()), data['fitting'], start_row=3,
                    col_widths=[18, 16, 14, 12, 12, 12])

        # ── Sheet 3: 剂量明细 ──
        ws3 = wb.create_sheet("剂量明细")
        ws3.cell(row=1, column=1, value="逐核素剂量明细").font = title_font
        dose_headers = ['化合物', '气溶胶类型', '核素', 'AMAD/粒径',
                        'e (Sv/Bq)', '活度贡献 (Bq/m³)', '剂量 (Sv)']
        dose_rows = [[
            r['compound'], r['aerosol_type'], r['nuclide'],
            str(r['amad']),
            f"{r['e_val']:.3e}" if not np.isnan(
                r.get('e_val', np.nan)) else "逐级",
            f"{r['act_conc']:.4e}", f"{r['dose']:.4e}",
        ] for r in data['dose_rows']]
        write_table(ws3, dose_headers, dose_rows, start_row=3,
                    col_widths=[14, 22, 12, 20, 14, 20, 16])

        # ── Sheet 4: 输入数据 ──
        ws4 = wb.create_sheet("输入数据")
        ws4.cell(row=1, column=1, value="输入采样数据与校正").font = title_font
        in_headers = list(data['input'][0].keys())
        write_table(ws4, in_headers, data['input'], start_row=3,
                    col_widths=[10, 18, 22, 22])

        # ── Sheet 5: 计算日志 ──
        ws5 = wb.create_sheet("计算日志")
        ws5.cell(row=1, column=1, value="计算详细日志").font = title_font
        for ri, line in enumerate(data['log'].split('\n'), 3):
            ws5.cell(row=ri, column=1, value=line).font = Font(
                name='Consolas', size=10)
        ws5.column_dimensions['A'].width = 100

        wb.save(path)
        QMessageBox.information(self, "导出成功", f"Excel 报告已保存至：\n{path}")
        self.status_bar.showMessage(
            f"✓ 已导出 Excel → {os.path.basename(path)}", 5000)

    def _export_csv(self):
        if not self._check_result():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV 结果表", "dose_results.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            data = self._build_export_data()
            rows = [['化合物', '气溶胶类型', '核素', 'AMAD/粒径',
                     'e (Sv/Bq)', '活度贡献 (Bq/m³)', '剂量 (Sv)']]
            for r in data['dose_rows']:
                rows.append([
                    r['compound'], r['aerosol_type'], r['nuclide'],
                    str(r['amad']),
                    f"{r['e_val']:.3e}" if not np.isnan(
                        r.get('e_val', np.nan)) else "逐级",
                    f"{r['act_conc']:.4e}", f"{r['dose']:.4e}",
                ])
            # 追加汇总行
            rows.append([])
            for k, v in data['summary'].items():
                rows.append([k, v])
            import csv
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            QMessageBox.information(self, "导出成功", f"CSV 已保存至：\n{path}")
            self.status_bar.showMessage(
                f"✓ 已导出 CSV → {os.path.basename(path)}", 5000)
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"导出 CSV 时出错：{e}")

    def _export_log(self):
        if not self._check_result():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出计算日志", "dose_calc_log.txt", "文本文件 (*.txt)")
        if not path:
            return
        try:
            data = self._build_export_data()
            with open(path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("  空气采样法内照射剂量计算系统 v3.1 — 计算日志\n")
                f.write(
                    f"  生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.write("[ 汇总参数 ]\n")
                for k, v in data['summary'].items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n[ 拟合参数 ]\n")
                for ft in data['fitting']:
                    f.write(
                        "  " + " | ".join(f"{k}={v}" for k, v in ft.items()) + "\n")
                f.write("\n" + "-" * 60 + "\n")
                f.write("[ 详细计算日志 ]\n\n")
                f.write(data['log'])
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("  报告结束\n")
                f.write("=" * 60 + "\n")
            QMessageBox.information(self, "导出成功", f"日志已保存至：\n{path}")
            self.status_bar.showMessage(
                f"✓ 已导出日志 → {os.path.basename(path)}", 5000)
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"导出日志时出错：{e}")

    def _export_all(self):
        if not self._check_result():
            return
        import datetime
        dir_path = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if not dir_path:
            return
        try:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            base = os.path.join(dir_path, f"dose_report_{ts}")

            # Excel
            xlsx_path = base + ".xlsx"
            self._do_export_xlsx(xlsx_path)

            # CSV
            csv_path = base + ".csv"
            self._do_export_csv(csv_path)

            # 日志
            log_path = base + "_log.txt"
            self._do_export_log(log_path)

            # 图表
            fig = self.plot_canvas.fig
            if fig and len(fig.axes) > 0:
                png_path = base + ".png"
                fig.savefig(png_path, dpi=150, bbox_inches='tight')

            files = [xlsx_path, csv_path, log_path]
            if os.path.exists(base + ".png"):
                files.append(base + ".png")

            QMessageBox.information(self, "导出成功",
                                    f"已导出 {len(files)} 个文件至：\n{dir_path}")
            self.status_bar.showMessage(f"✓ 导出完成 → {dir_path}", 8000)

        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"导出全部时出错：{e}")

    def _do_export_xlsx(self, path):
        """内部调用，不弹对话框"""
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        from openpyxl.utils import get_column_letter
        data = self._build_export_data()
        wb = Workbook()
        thin_border = Border(left=Side(style='thin'), right=Side(style='thin'),
                             top=Side(style='thin'), bottom=Side(style='thin'))
        header_fill = PatternFill(
            start_color='1A5276', end_color='1A5276', fill_type='solid')
        header_font = Font(name='微软雅黑', bold=True, color='FFFFFF', size=11)
        title_font = Font(name='微软雅黑', bold=True, size=14, color='1A5276')
        cell_font = Font(name='Consolas', size=10)

        def write_table(ws, headers, rows, start_row=1, col_widths=None):
            for ci, h in enumerate(headers, 1):
                c = ws.cell(row=start_row, column=ci, value=h)
                c.font, c.fill, c.alignment, c.border = header_font, header_fill, Alignment(
                    horizontal='center'), thin_border
            for ri, row in enumerate(rows):
                for ci, val in enumerate(row.values() if isinstance(row, dict) else row, 1):
                    c = ws.cell(row=start_row + 1 + ri, column=ci, value=val)
                    c.font, c.alignment, c.border = cell_font, Alignment(
                        horizontal='center'), thin_border
            if col_widths:
                for ci, w in enumerate(col_widths, 1):
                    ws.column_dimensions[get_column_letter(ci)].width = w

        ws1 = wb.active
        ws1.title = "汇总"
        ws1.cell(row=1, column=1, value="空气采样法内照射剂量计算报告").font = title_font
        ws1.merge_cells('A1:B1')
        for ri, (k, v) in enumerate(data['summary'].items(), 4):
            ws1.cell(row=ri, column=1, value=k).font = Font(
                name='微软雅黑', bold=True, size=10)
            ws1.cell(row=ri, column=2, value=v).font = cell_font
        ws1.column_dimensions['A'].width = 24
        ws1.column_dimensions['B'].width = 32

        ws2 = wb.create_sheet("拟合参数")
        ws2.cell(row=1, column=1, value="拟合参数与信息准则").font = title_font
        write_table(ws2, list(data['fitting'][0].keys()), data['fitting'], start_row=3,
                    col_widths=[18, 16, 14, 12, 12, 12])

        ws3 = wb.create_sheet("剂量明细")
        dose_headers = ['化合物', '气溶胶类型', '核素', 'AMAD/粒径',
                        'e (Sv/Bq)', '活度贡献 (Bq/m³)', '剂量 (Sv)']
        dose_rows = [[r['compound'], r['aerosol_type'], r['nuclide'], str(r['amad']),
                      f"{r['e_val']:.3e}" if not np.isnan(
                          r.get('e_val', np.nan)) else "逐级",
                      f"{r['act_conc']:.4e}", f"{r['dose']:.4e}"] for r in data['dose_rows']]
        write_table(ws3, dose_headers, dose_rows, start_row=3,
                    col_widths=[14, 22, 12, 20, 14, 20, 16])

        ws4 = wb.create_sheet("输入数据")
        in_headers = list(data['input'][0].keys())
        write_table(ws4, in_headers, data['input'],
                    start_row=3, col_widths=[10, 18, 22, 22])

        ws5 = wb.create_sheet("计算日志")
        for ri, line in enumerate(data['log'].split('\n'), 3):
            ws5.cell(row=ri, column=1, value=line).font = Font(
                name='Consolas', size=10)
        ws5.column_dimensions['A'].width = 100
        wb.save(path)

    def _do_export_csv(self, path):
        import csv
        data = self._build_export_data()
        rows = [['化合物', '气溶胶类型', '核素', 'AMAD/粒径',
                 'e (Sv/Bq)', '活度贡献 (Bq/m³)', '剂量 (Sv)']]
        for r in data['dose_rows']:
            rows.append([r['compound'], r['aerosol_type'], r['nuclide'], str(r['amad']),
                         f"{r['e_val']:.3e}" if not np.isnan(
                             r.get('e_val', np.nan)) else "逐级",
                         f"{r['act_conc']:.4e}", f"{r['dose']:.4e}"])
        rows.append([])
        for k, v in data['summary'].items():
            rows.append([k, v])
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerows(rows)

    def _do_export_log(self, path):
        data = self._build_export_data()
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("  空气采样法内照射剂量计算系统 v3.1 — 计算日志\n")
            f.write(
                f"  生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write("[ 汇总参数 ]\n")
            for k, v in data['summary'].items():
                f.write(f"  {k}: {v}\n")
            f.write("\n[ 拟合参数 ]\n")
            for ft in data['fitting']:
                f.write(
                    "  " + " | ".join(f"{k}={v}" for k, v in ft.items()) + "\n")
            f.write("\n" + "-" * 60 + "\n")
            f.write("[ 详细计算日志 ]\n\n")
            f.write(data['log'])
            f.write("\n\n" + "=" * 60 + "\n  报告结束\n" + "=" * 60 + "\n")

    def _log(self, s):
        self.log_edit.append(s)


# ==================== 程序入口 ====================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("微软雅黑", 10))
    pal = QPalette()
    pal.setColor(QPalette.Window,        QColor("#f5f6fa"))
    pal.setColor(QPalette.Base,          QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase, QColor("#eaf2ff"))
    app.setPalette(pal)
    win = DoseCalcApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

"""集成化剂量计算系统 v2.4 - 优化版 | streamlit run integrated_dose_full.py

架构说明：
├── 监测法（独立计算，输出总剂量）
└── 空气采样法（内嵌口罩防护 + 三种AMAD模式，支持文件上传/手动输入）
    ├── 国标单AMAD（默认5μm）
    ├── 多模态法（自动判断单峰/双峰，正确加权计算）
    └── Probit直线拟合法
└── 每个模块下方直接显示该模块的总剂量，无需单独汇总页面
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# 尝试导入 xf_core，如果失败则给出提示
try:
    from xf_core import (lognormal_pdf, stage_integral, fit_unimodal, fit_bimodal,
                         fit_linear_probit, judge_distribution, recommend_method,
                         stages_low, stages_high, cut_diameters, midpoints,
                         efficiency, apply_efficiency_correction)
except ImportError as e:
    st.error(f"❌ 无法导入 xf_core 模块：{e}\n请确保 xf_core.py 文件存在于当前目录，并包含所需函数。")
    st.stop()

warnings.filterwarnings('ignore')
st.set_page_config(page_title="集成化剂量计算系统 v2.4", page_icon="🧪", layout="wide")

# ==================== 全局常量 ====================
DATA_DIR = Path("./processed_nuclide_files")
M_DATA_DIR = Path("./m_data")

# 口罩参数（国标口罩型号）
MASK_PARAMS = {
    "无防护": {
        "filtration_efficiency": 0.0,
        "leakage_rate": 1.0,
        "description": "无口罩防护（基准对照）",
        "is_electret": False,
        "mpps_um": 0.3,
    },
    "普通熔喷口罩": {
        "filtration_efficiency": 0.60,
        "leakage_rate": 0.15,
        "description": "普通熔喷布（无驻极静电，仅机械过滤）",
        "is_electret": False,
        "mpps_um": 0.3,
    },
    "KN90": {
        "filtration_efficiency": 0.90,
        "leakage_rate": 0.10,
        "description": "KN90（GB 2626-2019，驻极体熔喷）",
        "is_electret": True,
        "mpps_um": 0.3,
    },
    "KN95": {
        "filtration_efficiency": 0.95,
        "leakage_rate": 0.08,
        "description": "KN95（GB 2626-2019，驻极体熔喷）",
        "is_electret": True,
        "mpps_um": 0.3,
    },
    "FFP2": {
        "filtration_efficiency": 0.94,
        "leakage_rate": 0.08,
        "description": "FFP2（EN 149，驻极体熔喷）",
        "is_electret": True,
        "mpps_um": 0.3,
    },
    "N95": {
        "filtration_efficiency": 0.95,
        "leakage_rate": 0.08,
        "description": "N95（NIOSH，驻极体熔喷）",
        "is_electret": True,
        "mpps_um": 0.3,
    },
    "医用外科口罩": {
        "filtration_efficiency": 0.70,
        "leakage_rate": 0.20,
        "description": "医用外科（YY 0469，弱驻极）",
        "is_electret": True,
        "mpps_um": 0.3,
    },
}

# ==================== 状态初始化 ====================


def init_state():
    keys_defaults = {
        "monitor_items": [], "next_monitor_id": 1,
        "air_items": [], "next_air_id": 1,
        "element_nuclides_map": {}, "available_elements": [],
        "nuclide_files": {}, "nuclide_data_cache": {}, "full_nuclide_data_cache": {},
        "routine_data": None, "special_data": None, "auto_scanned": False,
        "air_fitted": False,
        "air_fit_result": None,
        "air_fig": None,
        "air_current_row": None,
        "air_row_hash": None,
    }
    for k, v in keys_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ==================== 数据扫描函数 ====================


def scan_nuclides():
    try:
        if not DATA_DIR.exists():
            st.error(f"目录不存在: {DATA_DIR}")
            return False
        files = [f for ext in ['.parquet', '.xlsx', '.xls', '.csv']
                 for f in DATA_DIR.glob(f"*{ext}") if not f.name.startswith('.')]
        elem_map, nuc_files = {}, {}
        for f in files:
            try:
                name = f.stem.replace("Processed_", "")
                df = pd.read_parquet(f) if f.suffix == '.parquet' else (
                    pd.read_excel(f, sheet_name='Processed Data') if f.suffix in ['.xlsx', '.xls']
                    else pd.read_csv(f))
                if 'element' in df.columns and 'radionuclide' in df.columns:
                    elem, nuc = df['element'].iloc[0], df['radionuclide'].iloc[0]
                    if elem not in elem_map:
                        elem_map[elem] = []
                    if nuc not in elem_map[elem]:
                        elem_map[elem].append(nuc)
                    nuc_files[nuc] = f
            except:
                pass
        st.session_state.element_nuclides_map = elem_map
        st.session_state.available_elements = sorted(elem_map.keys())
        st.session_state.nuclide_files = nuc_files
        return True
    except Exception as e:
        st.error(f"扫描出错: {e}")
        return False


def load_nucl(nuclide, full=False):
    cache = 'full_nuclide_data_cache' if full else 'nuclide_data_cache'
    if nuclide in st.session_state[cache]:
        return st.session_state[cache][nuclide]
    try:
        if 'nuclide_files' not in st.session_state or nuclide not in st.session_state.nuclide_files:
            return None
        f = st.session_state.nuclide_files[nuclide]
        df = pd.read_parquet(f) if f.suffix == '.parquet' else (
            pd.read_excel(f, sheet_name='Processed Data') if f.suffix in ['.xlsx', '.xls']
            else pd.read_csv(f))
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if 'route' in cl and 'intake' in cl:
                col_map[c] = 'route_of_intake'
            elif 'aerosol' in cl and 'type' in cl:
                col_map[c] = 'aerosol_type'
            elif 'particle' in cl and 'size' in cl:
                col_map[c] = 'particle_size'
            elif 'dose' in cl and 'coefficient' in cl:
                col_map[c] = 'dose_coefficient'
            elif 'radionuclide' in cl:
                col_map[c] = 'radionuclide'
            elif cl == 'element':
                col_map[c] = 'element'
            elif cl == 'fa':
                col_map[c] = 'fA'
        df = df.rename(columns=col_map)
        if 'aerosol_type' in df.columns:
            df['aerosol_type'] = df['aerosol_type'].replace(
                'Gaseous', 'Unspecified')
        if 'particle_size' in df.columns:
            df['particle_size'] = pd.to_numeric(
                df['particle_size'].astype(str).str.replace(
                    r'\s*µm\s*|\s*micron\s*', '', regex=True)
                .replace(['', 'nan', 'NaN', 'None'], np.nan), errors='coerce')
        if not full and 'route_of_intake' in df.columns:
            df = df[df['route_of_intake'] == 'Inhalation']
        for c in ['dose_coefficient', 'fA']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        st.session_state[cache][nuclide] = df
        return df
    except Exception as e:
        st.warning(f"加载核素 {nuclide} 失败: {e}")
        return None


def interp_dc(df, aerosol_type, particle_size):
    if df is None or df.empty:
        return None
    ad = df[df['aerosol_type'] == aerosol_type].copy()
    if ad.empty:
        return None
    ad = ad.sort_values('particle_size')
    ps, dcs = ad['particle_size'].values, ad['dose_coefficient'].values
    if len(ps) == 1:
        return dcs[0]
    if particle_size in ps:
        return dcs[np.where(ps == particle_size)[0][0]]
    if particle_size <= ps.min():
        return dcs[0]
    if particle_size >= ps.max():
        return dcs[-1]
    log_ps, log_x = np.log(ps), np.log(particle_size)
    for i in range(len(log_ps) - 1):
        if log_ps[i] <= log_x <= log_ps[i + 1]:
            x1, x2, y1, y2 = log_ps[i], log_ps[i + 1], dcs[i], dcs[i + 1]
            return y1 if x2 == x1 else y1 + (log_x - x1) / (x2 - x1) * (y2 - y1)
    return None


def scan_monitoring():
    if not M_DATA_DIR.exists():
        return None, None

    def parse(files):
        records = []
        for f in files:
            try:
                df = pd.read_parquet(f) if f.suffix == '.parquet' else (
                    pd.read_excel(f) if f.suffix in ['.xlsx', '.xls'] else pd.read_csv(f))
                cm = {}
                for c in df.columns:
                    cl = c.lower()
                    if 'radionuclide' in cl or '核素' in cl:
                        cm[c] = 'radionuclide'
                    elif 'aerosols type' in cl or '气溶胶类型' in cl:
                        cm[c] = 'aerosol_type'
                    elif 'monitoring method' in cl or '监测方法' in cl:
                        cm[c] = 'monitoring_method'
                    elif 'period' in cl or 'time' in cl or '周期' in cl or '时间' in cl:
                        cm[c] = 'time_days'
                    elif 'm(t/2)' in cl or 'm(t)' in cl or 'm值' in cl or cl == 'm':
                        cm[c] = 'm_value'
                    elif 'route of intake' in cl or '摄入途径' in cl:
                        cm[c] = 'intake_route'
                df = df.rename(columns=cm)
                if 'aerosol_type' in df.columns:
                    df['aerosol_type'] = df['aerosol_type'].replace(
                        'Gaseous', 'Unspecified')
                if 'radionuclide' in df.columns and 'm_value' in df.columns:
                    if 'time_days' in df.columns:
                        df['time_days'] = pd.to_numeric(
                            df['time_days'], errors='coerce')
                    records.append(df)
            except:
                pass
        return pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    r = [f for ext in ['.csv', '.xlsx', '.xls', '.parquet']
         for f in M_DATA_DIR.glob(f"*{ext}") if 'routine' in f.stem.lower()]
    s = [f for ext in ['.csv', '.xlsx', '.xls', '.parquet']
         for f in M_DATA_DIR.glob(f"*{ext}") if 'special' in f.stem.lower()]
    return parse(r), parse(s)


# ==================== 自动扫描数据 ====================
if not st.session_state.get("auto_scanned", False):
    with st.spinner("正在自动扫描数据..."):
        scan_nuclides()
        r, s = scan_monitoring()
        st.session_state.routine_data, st.session_state.special_data = r, s
        st.session_state.auto_scanned = True

# ==================== 口罩防护计算 ====================


def calc_mask_protection_factor(mask_type, particle_size_um=None):
    if mask_type not in MASK_PARAMS:
        return 1.0, 0.0, 1.0, ["未知型号，使用无防护基准"]

    p = MASK_PARAMS[mask_type]
    base_eff = p["filtration_efficiency"]
    leak = p["leakage_rate"]
    mpps = p.get("mpps_um", 0.3)
    is_electret = p.get("is_electret", False)
    notes = [p["description"]]

    if particle_size_um is None:
        pf = leak + (1 - base_eff) * (1 - leak)
        notes.append(f"无粒径输入，使用标称效率：{base_eff:.0%}")
        notes.append(f"防护因子 PF = {pf:.4f}")
        return pf, base_eff, leak, notes

    size = particle_size_um
    adj_eff = base_eff

    if not is_electret:
        if size <= 0.01:
            adj_eff = 0.0
            notes.append("⚠️ 极小粒径，普通熔喷布完全无法防护")
        else:
            adj_eff = base_eff * min(1.0, size / mpps)
            adj_eff = max(0.0, min(adj_eff, 0.99))
            notes.append(f"普通熔喷：粒径{size}μm → 效率={adj_eff:.1%}")
    else:
        if size <= 0.01:
            adj_eff = 0.1
            notes.append("⚠️ 超微纳米颗粒，驻极体效率大幅下降")
        elif abs(size - mpps) < 1e-6:
            adj_eff = base_eff
            notes.append(f"✅ MPPS={mpps}μm，防护效率最低（标称值）")
        else:
            dist = abs(size - mpps)
            boost = min(dist / mpps * 0.15, 0.09)
            adj_eff = min(base_eff + boost, 0.99)
            notes.append(f"驻极体：偏离MPPS，效率提升至 {adj_eff:.1%}")

    final_pf = leak + (1 - adj_eff) * (1 - leak)
    notes.append(f"最终 PF = {final_pf:.4f}（越小越好）")
    return final_pf, adj_eff, leak, notes


def get_stage_pf(mask_type, particle_size_um):
    pf, _, _, _ = calc_mask_protection_factor(mask_type, particle_size_um)
    return pf


def apply_mask_to_stages(raw_activities, mask_type, stage_midpoints):
    if mask_type == "无防护":
        return raw_activities.copy()
    corrected = []
    for act, d_mid in zip(raw_activities, stage_midpoints):
        pf = get_stage_pf(mask_type, d_mid)
        corrected.append(act * pf)
    return np.array(corrected)

# ==================== 绘图函数 ====================


def plot_fitting_results(workshop, sampling_id, raw_activities, corrected_activities,
                         amad_uni, gsd_uni, total_uni,
                         amad1, gsd1, frac1, amad2, gsd2, total_bi,
                         total_activity_corr, return_fig=False, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    x_plot = np.logspace(np.log10(0.1), np.log10(50), 200)
    stage_widths = np.log(stages_high / stages_low)
    data_density = corrected_activities / stage_widths / total_activity_corr

    y_unimodal = lognormal_pdf(
        x_plot, amad_uni, gsd_uni) * total_uni / total_activity_corr
    if not np.isnan(amad2):
        def total_pdf(dp, a1, g1, f1, a2, g2):
            return f1 * lognormal_pdf(dp, a1, g1) + (1-f1) * lognormal_pdf(dp, a2, g2)
        y_bimodal = total_pdf(x_plot, amad1, gsd1, frac1, amad2, gsd2)
    else:
        y_bimodal = y_unimodal

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 单峰拟合图
    ax = axes[0]
    ax.semilogx(x_plot, y_unimodal, 'b-', linewidth=2,
                label=f'Unimodal AMAD={amad_uni:.1f} μm')
    ax.scatter(midpoints, data_density, color='red',
               s=60, edgecolor='black', zorder=5)
    ax.set_xlabel('Aerodynamic Diameter (μm)')
    ax.set_ylabel('dA/dln(dp) normalized')
    ax.set_title(f'Unimodal Fit | {workshop} {sampling_id}')
    ax.grid(ls='--', alpha=0.6)
    ax.set_xlim(0.1, 50)
    ax.set_ylim(0)
    ax.legend()

    # 双峰拟合图
    ax = axes[1]
    ax.semilogx(x_plot, y_bimodal, 'k-', linewidth=2, label='Total fit')
    if not np.isnan(amad2):
        ax.semilogx(x_plot, frac1 * lognormal_pdf(x_plot, amad1,
                    gsd1), 'b--', label=f'Coarse {amad1:.1f} μm')
        ax.semilogx(x_plot, (1 - frac1) * lognormal_pdf(x_plot,
                    amad2, gsd2), 'r--', label=f'Fine {amad2:.1f} μm')
    ax.scatter(midpoints, data_density, color='red',
               s=60, edgecolor='black', zorder=5)
    ax.set_xlabel('Aerodynamic Diameter (μm)')
    ax.set_title(f'Bimodal Fit | {workshop} {sampling_id}')
    ax.grid(ls='--', alpha=0.6)
    ax.set_xlim(0.1, 50)
    ax.set_ylim(0)
    ax.legend()

    # 概率图法
    ax = axes[2]
    raw_acts = raw_activities[:8]
    filter_act = raw_activities[8]
    total = np.sum(raw_acts) + filter_act
    f = raw_acts / total
    f_filt = filter_act / total
    f_all = np.concatenate([f, [f_filt]])
    cum_from_fine = np.cumsum(np.flip(f_all))[:-1]
    cum_less = np.flip(cum_from_fine) * 100
    valid = (cum_less > 1) & (cum_less < 99)

    if np.sum(valid) >= 3:
        x = np.log(cut_diameters[valid])
        y = norm.ppf(cum_less[valid] / 100)
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        ln_AMAD = -intercept / slope
        AMAD = np.exp(ln_AMAD)
        D50 = AMAD
        ln_GSD = 1 / slope
        GSD = np.exp(ln_GSD)
        D84_1 = np.exp(ln_AMAD + ln_GSD)
        D15_9 = np.exp(ln_AMAD - ln_GSD)
        ax.set_xscale('log')
        ax.set_xlim(0.3, 30)
        ax.set_xlabel('Aerodynamic Diameter (μm)', fontsize=12)
        percentiles = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 40,
                       50, 60, 70, 80, 90, 95, 98, 99, 99.5, 99.9]
        probit_ticks = norm.ppf(np.array(percentiles) / 100)
        ax.set_yticks(probit_ticks)
        ax.set_yticklabels([f'{p}' for p in percentiles])
        ax.set_ylim(norm.ppf(0.005), norm.ppf(0.995))
        ax.set_ylabel('Cumulative activity (%)', fontsize=12)
        ax.scatter(cut_diameters[valid], y, color='red',
                   s=80, label='Measured data', zorder=5)
        x_fit = np.logspace(np.log10(0.5), np.log10(21.3), 200)
        y_fit = slope * np.log(x_fit) + intercept
        ax.plot(x_fit, y_fit, 'b-', linewidth=2, label='ICRP regression line')
        ax.scatter(D50, 0, color='green', s=110, marker='s',
                   label=f'AMAD (D50) = {D50:.1f} μm')
        ax.scatter(D84_1, norm.ppf(0.8413), color='orange', s=110,
                   marker='s', label=f'D84.1 = {D84_1:.1f} μm')
        ax.scatter(D15_9, norm.ppf(0.1587), color='purple', s=110,
                   marker='s', label=f'D15.9 = {D15_9:.1f} μm')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=norm.ppf(0.8413), color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=norm.ppf(0.1587), color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f'Probit Plot | {workshop} {sampling_id}')
        ax.legend()
        y_pred = reg.predict(x.reshape(-1, 1))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'Insufficient data for probit plot',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    if return_fig:
        return fig
    else:
        plt.savefig(os.path.join(
            output_dir, f'{workshop}_{sampling_id}.png'), dpi=150)
        plt.close()
        return None


# ==================== 页面布局 ====================
st.title("🧪 集成化剂量计算系统 v2.4")
st.markdown(
    "**监测法 + 空气采样法（内嵌口罩防护）** | 各模块独立计算并显示总剂量 | **已优化：添加呼吸参数、修正双峰逻辑、支持多化合物类型**")

# 侧边栏
st.sidebar.header("📂 数据管理")
if st.sidebar.button("🔄 重新扫描数据"):
    with st.spinner("扫描中..."):
        scan_nuclides()
        r, s = scan_monitoring()
        st.session_state.routine_data, st.session_state.special_data = r, s
        st.rerun()

if st.session_state.available_elements:
    st.sidebar.success(f"✅ 已加载 {len(st.session_state.available_elements)} 个元素")
    with st.sidebar.expander("📊 完整数据概览", expanded=True):
        st.subheader("🧪 元素与核素")
        st.write(f"✅ 元素总数：{len(st.session_state.available_elements)}")
        total_nuclides = sum(len(v)
                             for v in st.session_state.element_nuclides_map.values())
        st.write(f"✅ 核素总数：{total_nuclides}")
        st.divider()
        st.subheader("📅 常规监测数据")
        routine = st.session_state.routine_data
        if not routine.empty:
            st.write(f"✅ 常规记录数：{len(routine)} 条")
        else:
            st.write("❌ 无常规监测数据")
        st.divider()
        st.subheader("⚠️ 专项监测数据")
        special = st.session_state.special_data
        if not special.empty:
            st.write(f"✅ 专项记录数：{len(special)} 条")
        else:
            st.write("❌ 无专项监测数据")
        st.divider()
        st.subheader("📊 总数据量")
        total_rows = len(routine) + len(special)
        st.write(f"✅ 所有监测数据总计：{total_rows} 条")
else:
    st.sidebar.warning("⚠️ 未扫描到数据")

# ==================== 主界面分页 ====================
tab1, tab2 = st.tabs(["📋 监测法", "💨 空气采样法"])

# ==================== 监测法 ====================
with tab1:
    st.header("📋 监测法")
    st.info("基于监测数据，计算内照射有效剂量（独立计算）")

    if 'monitor_items' not in st.session_state:
        st.session_state.monitor_items = []
    if 'next_monitor_id' not in st.session_state:
        st.session_state.next_monitor_id = 1
    if 'prev_monitor_type' not in st.session_state:
        st.session_state.prev_monitor_type = None

    monitor_type = st.radio(
        "选择监测类型", ["常规监测", "应急监测"], horizontal=True, key="monitor_type_radio")
    if monitor_type != st.session_state.prev_monitor_type:
        st.session_state.monitor_items = []
        st.session_state.next_monitor_id = 1
        st.session_state.prev_monitor_type = monitor_type
        st.rerun()

    if monitor_type == "常规监测":
        data = st.session_state.routine_data
        data_name = "常规"
    else:
        data = st.session_state.special_data
        data_name = "应急"

    if data is None or data.empty:
        st.info(f"请先在左侧边栏扫描并加载{data_name}监测数据")
    else:
        col_left, col_right = st.columns([1.2, 1.6])

        with col_left:
            st.subheader("➕ 添加监测项")
            nuclides = sorted(data['radionuclide'].unique())
            if len(nuclides) == 0:
                st.error("数据中无核素信息")
                st.stop()
            selected_nuclide = st.selectbox("核素", nuclides, key="add_nuclide")
            df_nuc = data[data['radionuclide'] == selected_nuclide]

            selected_intake = None
            if monitor_type == "应急监测":
                if 'intake_route' in df_nuc.columns:
                    intake_routes = df_nuc['intake_route'].dropna().unique()
                    if len(intake_routes) > 0:
                        selected_intake = st.selectbox(
                            "摄入途径", sorted(intake_routes), key="add_intake")
                        df_nuc = df_nuc[df_nuc['intake_route']
                                        == selected_intake]
                    else:
                        st.caption("摄入途径: 无数据，跳过")
                else:
                    st.caption("摄入途径: 列不存在，跳过")
            else:
                selected_intake = "Inhalation"
                st.caption("常规监测默认摄入途径: Inhalation")

            selected_aerosol = None
            if selected_intake == "Inhalation" and 'aerosol_type' in df_nuc.columns:
                aerosol_types = df_nuc['aerosol_type'].dropna().unique()
                if len(aerosol_types) > 0:
                    selected_aerosol = st.selectbox(
                        "气溶胶类型", sorted(aerosol_types), key="add_aerosol")
                    df_nuc = df_nuc[df_nuc['aerosol_type'] == selected_aerosol]
                else:
                    st.caption("气溶胶类型: 无数据，跳过")
            elif selected_intake != "Inhalation":
                st.caption("气溶胶类型: 仅吸入途径需要")
            else:
                st.caption("气溶胶类型: 列不存在，跳过")

            selected_method = None
            if 'monitoring_method' in df_nuc.columns:
                methods = df_nuc['monitoring_method'].dropna().unique()
                if len(methods) > 0:
                    selected_method = st.selectbox(
                        "监测方法", sorted(methods), key="add_method")
                    df_nuc = df_nuc[df_nuc['monitoring_method']
                                    == selected_method]
                else:
                    st.caption("监测方法: 无数据，跳过")
            else:
                st.caption("监测方法: 列不存在，跳过")

            selected_time = None
            if 'time_days' in df_nuc.columns:
                times = df_nuc['time_days'].dropna().unique()
                if len(times) > 0:
                    selected_time = st.selectbox(
                        "时间 (d)", sorted(times), key="add_time")
                    df_nuc = df_nuc[df_nuc['time_days'] == selected_time]
                else:
                    st.warning("该组合下没有时间数据")

            selected_fA = None
            if selected_intake == "Ingestion" and 'fA' in df_nuc.columns:
                fA_values = df_nuc['fA'].dropna().unique()
                if len(fA_values) == 0:
                    st.warning("该核素食入途径下无fA数据")
                elif len(fA_values) == 1:
                    selected_fA = fA_values[0]
                    df_nuc = df_nuc[df_nuc['fA'] == selected_fA]
                    st.caption(f"fA = {selected_fA}")
                else:
                    selected_fA = st.selectbox(
                        "选择fA值", sorted(fA_values), key="add_fA_m")
                    df_nuc = df_nuc[df_nuc['fA'] == selected_fA]
                    st.caption(f"fA = {selected_fA}")

            if len(df_nuc) == 1:
                m_value = df_nuc.iloc[0]['m_value']
                st.info(f"**m 值** = {m_value:.2e}")
            else:
                st.warning("未找到唯一 m 值，请调整选择条件")
                m_value = None

            if m_value is not None:
                M = st.number_input("测量值 M (单位需与 m 匹配):", min_value=0.0,
                                    value=1.0, step=0.1, format="%.2e", key="M_input")
                I = M / m_value
                st.caption(f"摄入量 I = M / m = {I:.2e} Bq")

                e_value = None
                selected_size = None
                df_e = load_nucl(selected_nuclide, full=True)
                if df_e is not None and not df_e.empty:
                    if selected_intake is not None and 'route_of_intake' in df_e.columns:
                        df_e = df_e[df_e['route_of_intake'] == selected_intake]

                    if selected_intake == "Inhalation":
                        if selected_aerosol is not None and 'aerosol_type' in df_e.columns:
                            df_e = df_e[df_e['aerosol_type']
                                        == selected_aerosol]
                        if df_e.empty:
                            st.warning(
                                f"空气采样法数据中无核素 {selected_nuclide} 的吸入途径数据")
                        else:
                            has_particle_sizes = df_e['particle_size'].notna(
                            ).any()
                            if not has_particle_sizes:
                                if len(df_e) == 1:
                                    e_value = df_e.iloc[0]['dose_coefficient']
                                    st.info(f"剂量系数 e = {e_value:.2e} Sv/Bq")
                                else:
                                    st.warning("该气溶胶类型下有多个记录但无粒径数据，无法确定")
                            else:
                                available_sizes = sorted(
                                    df_e['particle_size'].dropna().unique())
                                if len(available_sizes) == 0:
                                    st.warning("该气溶胶类型下没有粒径数据")
                                else:
                                    predefined_sizes = [
                                        0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0, 20.0]
                                    size_options = [
                                        f"{s}µm" for s in predefined_sizes if s in available_sizes] + ["自定义"]
                                    if len(size_options) == 0:
                                        size_options = ["自定义"]
                                    selected_size_str = st.selectbox(
                                        "选择粒径 (µm):", size_options, key="add_size_select")
                                    if selected_size_str == "自定义":
                                        particle_size = st.number_input("输入自定义粒径 (µm):", min_value=0.001, max_value=100.0,
                                                                        value=1.0, step=0.001, format="%.3f", key="add_custom_size")
                                    else:
                                        particle_size = float(
                                            selected_size_str.replace("µm", ""))
                                    e_value = interp_dc(
                                        df_e, selected_aerosol, particle_size)
                                    if e_value is not None:
                                        st.info(
                                            f"剂量系数 e = {e_value:.2e} Sv/Bq")
                                        selected_size = particle_size
                    elif selected_intake == "Ingestion":
                        if 'fA' in df_e.columns and selected_fA is not None:
                            df_e = df_e[df_e['fA'] == selected_fA]
                        if not df_e.empty:
                            e_value = df_e.iloc[0]['dose_coefficient']
                            st.info(f"剂量系数 e = {e_value:.2e} Sv/Bq")
                    elif selected_intake == "Injection":
                        if not df_e.empty:
                            e_value = df_e.iloc[0]['dose_coefficient']
                            st.info(f"剂量系数 e = {e_value:.2e} Sv/Bq")
                    else:
                        if not df_e.empty:
                            e_value = df_e.iloc[0]['dose_coefficient']
                            st.info(f"剂量系数 e = {e_value:.2e} Sv/Bq")

                if e_value is not None:
                    E = I * e_value
                    st.success(f"**该监测项有效剂量** = {E:.2e} Sv")
                    if st.button("➕ 添加此监测项", type="primary", key="add_monitor_item"):
                        new_item = {
                            "id": st.session_state.next_monitor_id,
                            "monitor_type": monitor_type,
                            "nuclide": selected_nuclide,
                            "intake_route": selected_intake,
                            "aerosol_type": selected_aerosol,
                            "monitoring_method": selected_method,
                            "time_days": selected_time,
                            "fA": selected_fA,
                            "m_value": m_value,
                            "M": M,
                            "I": I,
                            "e_value": e_value,
                            "E": E,
                            "particle_size": selected_size
                        }
                        st.session_state.monitor_items.append(new_item)
                        st.session_state.next_monitor_id += 1
                        st.rerun()
                else:
                    st.info("无法计算有效剂量，请检查参数")
            else:
                st.info("请完成监测参数选择")

        with col_right:
            st.subheader("📋 当前监测项列表")
            if not st.session_state.monitor_items:
                st.info("暂无监测项，请左侧添加")
            else:
                for item in st.session_state.monitor_items:
                    with st.expander(f"监测项 {item['id']}: {item['nuclide']} - 测量值 {item['M']:.2e}", expanded=False):
                        st.write(f"**核素**: {item['nuclide']}")
                        st.write(
                            f"**摄入途径**: {item['intake_route'] if item['intake_route'] else '默认吸入'}")
                        if item['aerosol_type']:
                            st.write(f"**气溶胶类型**: {item['aerosol_type']}")
                        if item['monitoring_method']:
                            st.write(f"**监测方法**: {item['monitoring_method']}")
                        if item['time_days']:
                            st.write(f"**时间 (d)**: {item['time_days']}")
                        if item['fA']:
                            st.write(f"**fA值**: {item['fA']}")
                        if item['particle_size']:
                            st.write(
                                f"**粒径 (µm)**: {item['particle_size']:.3f}")
                        st.write(f"**m值**: {item['m_value']:.2e}")
                        st.write(f"**测量值 M**: {item['M']:.2e}")
                        st.write(f"**摄入量 I**: {item['I']:.2e} Bq")
                        st.write(f"**剂量系数 e**: {item['e_value']:.2e} Sv/Bq")
                        st.write(f"**有效剂量 E**: {item['E']:.2e} Sv")
                        if st.button(f"🗑️ 删除此项", key=f"del_{item['id']}"):
                            st.session_state.monitor_items = [
                                i for i in st.session_state.monitor_items if i["id"] != item['id']]
                            st.rerun()
                total_eff = sum(item['E']
                                for item in st.session_state.monitor_items)
                st.markdown("---")
                st.success(f"### **监测法总有效剂量**  \n# {total_eff:.2e} Sv")
                st.caption(f"共 {len(st.session_state.monitor_items)} 个监测项")
                if st.button("🗑️ 清空所有监测项", type="secondary"):
                    st.session_state.monitor_items = []
                    st.session_state.next_monitor_id = 1
                    st.rerun()
                if st.button("📥 导出监测结果", key="export_monitor_items"):
                    export_df = pd.DataFrame(st.session_state.monitor_items)
                    export_cols = ['id', 'monitor_type', 'nuclide', 'intake_route', 'aerosol_type',
                                   'monitoring_method', 'time_days', 'fA', 'm_value', 'M', 'I',
                                   'e_value', 'E', 'particle_size']
                    export_cols = [
                        c for c in export_cols if c in export_df.columns]
                    export_df = export_df[export_cols].copy()
                    export_df['计算时间'] = pd.Timestamp.now().strftime(
                        "%Y-%m-%d %H:%M:%S")
                    csv = export_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 下载结果", data=csv, file_name="monitor_items.csv", mime="text/csv")

# ==================== 空气采样法 ====================
with tab2:
    st.header("💨 空气采样法")
    st.info("基于分级测量数据计算AMAD/GSD，内嵌口罩防护修正，输出带防护的总剂量（独立计算）")
    st.warning("⚠️ **重要优化**：已添加呼吸速率和工作时长参数，修正双峰处理逻辑，支持多化合物类型叠加")

    # 口罩选择（放在数据录入之前）
    st.subheader("😷 口罩防护（内嵌）")
    c_mask = st.columns([2, 1, 1])
    with c_mask[0]:
        sel_mask = st.selectbox("选择口罩型号", list(
            MASK_PARAMS.keys()), key="air_mask")
    p_mask = MASK_PARAMS[sel_mask]
    c_mask[1].metric("过滤效率", f"{p_mask['filtration_efficiency']:.0%}")
    c_mask[2].metric("泄露率", f"{p_mask['leakage_rate']:.0%}")
    if sel_mask != "无防护":
        st.caption(f"ℹ️ {p_mask['description']}")
    st.caption("⚠️ 更改口罩后，请重新点击「计算AMAD并绘图」以更新结果")

    # 第一步：选择方法
    method_choice = st.radio(
        "选择AMAD确定方法",
        ["🌐 国标单AMAD（默认5μm）", "📊 录入分级测量数据（拟合AMAD）"],
        horizontal=True,
        key="air_method_choice"
    )

    if method_choice == "🌐 国标单AMAD（默认5μm）":
        st.success("✅ 使用固定AMAD=5μm（符合ICRP-66及国标规范）")
        total_conc_input = st.number_input(
            "原始总活度浓度 (Bq/m³)", value=1.0, format="%.2f", key="std_total_conc")
        pf_overall = get_stage_pf(sel_mask, 5.0)
        total_conc_corrected = total_conc_input * pf_overall
        st.info(
            f"口罩防护后总活度浓度 = {total_conc_corrected:.2e} Bq/m³ (防护因子={pf_overall:.4f})")

        # 添加呼吸参数
        st.subheader("🫁 呼吸参数")
        col_breath = st.columns(2)
        with col_breath[0]:
            br_std = st.number_input("呼吸速率 BR (m³/h)", min_value=0.5,
                                     max_value=3.0, value=1.2, step=0.1, key="breathing_rate_std")
        with col_breath[1]:
            work_hours_std = st.number_input(
                "工作时长 T (h)", min_value=0.5, max_value=24.0, value=8.0, step=0.5, key="work_hours_std")

        st.session_state.air_fitted = True
        st.session_state.air_fit_result = {
            'method': 'Std_AMAD', 'amad': 5.0, 'gsd': 1.5,
            'total_raw': total_conc_input,
            'total_masked': total_conc_corrected,
            'mask_pf_overall': pf_overall,
            'breathing_rate': br_std,
            'work_hours': work_hours_std
        }
        st.session_state.air_current_row = None
    else:
        # 录入分级测量数据
        data_source = st.radio(
            "数据来源", ["从文件上传 (CSV/Excel)", "手动输入分级浓度"], horizontal=True, key="air_data_source")
        current_row = None

        if data_source == "从文件上传 (CSV/Excel)":
            uploaded_file = st.file_uploader(
                "上传分级测量数据文件", type=["csv", "xlsx", "xls"], key="air_file")
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_upload = pd.read_csv(uploaded_file)
                    else:
                        df_upload = pd.read_excel(uploaded_file)
                    required_cols = ['Workshop', 'SamplingID', 'A_conc', 'B_conc', 'C_conc', 'D_conc',
                                     'E_conc', 'F_conc', 'G_conc', 'H_conc', 'Filter_conc']
                    if all(col in df_upload.columns for col in required_cols):
                        ws_list = sorted(df_upload['Workshop'].unique())
                        col_ws, col_id = st.columns(2)
                        with col_ws:
                            sel_w = st.selectbox(
                                "车间", ws_list, key="air_ws_upload")
                        df_w = df_upload[df_upload['Workshop'] == sel_w]
                        with col_id:
                            sids = sorted(df_w['SamplingID'].unique())
                            sel_s = st.selectbox(
                                "采样ID", sids, key="air_sid_upload")
                        current_row = df_w[df_w['SamplingID'] == sel_s].iloc[0]
                        with st.expander("📊 分级测量数据"):
                            cols = ['A_conc', 'B_conc', 'C_conc', 'D_conc', 'E_conc',
                                    'F_conc', 'G_conc', 'H_conc', 'Filter_conc']
                            data_show = pd.DataFrame({
                                '分级': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Filter'],
                                '粒径范围(µm)': ['14.8-21.3', '9.8-14.8', '6.0-9.8', '3.5-6.0', '1.6-3.5', '0.9-1.6', '0.5-0.9', '0.1-0.5', '<0.1'],
                                '浓度(Bq/m³)': [current_row[c] for c in cols]
                            })
                            st.dataframe(data_show, use_container_width=True)
                    else:
                        st.error(f"文件缺少必要列，需要：{required_cols}")
                except Exception as e:
                    st.error(f"读取文件失败: {e}")
            else:
                st.info("请上传CSV或Excel文件")
        else:  # 手动输入
            st.subheader("📝 手动输入各分级活度浓度 (Bq/m³)")
            cols_input = st.columns(3)
            conc_values = {}
            stages = ['A_conc', 'B_conc', 'C_conc', 'D_conc', 'E_conc',
                      'F_conc', 'G_conc', 'H_conc', 'Filter_conc']
            stage_names = ['A (14.8-21.3µm)', 'B (9.8-14.8µm)', 'C (6.0-9.8µm)',
                           'D (3.5-6.0µm)', 'E (1.6-3.5µm)', 'F (0.9-1.6µm)',
                           'G (0.5-0.9µm)', 'H (0.1-0.5µm)', 'Filter (<0.1µm)']
            for i, (stage, name) in enumerate(zip(stages, stage_names)):
                with cols_input[i % 3]:
                    conc_values[stage] = st.number_input(
                        name, value=0.0, format="%.2f", key=f"manual_{stage}")
            current_row = pd.Series(conc_values)
            current_row['Workshop'] = "手动输入"
            current_row['SamplingID'] = "manual"
            with st.expander("📊 输入数据汇总"):
                data_show = pd.DataFrame({
                    '分级': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Filter'],
                    '粒径范围(µm)': ['14.8-21.3', '9.8-14.8', '6.0-9.8', '3.5-6.0', '1.6-3.5', '0.9-1.6', '0.5-0.9', '0.1-0.5', '<0.1'],
                    '浓度(Bq/m³)': [conc_values[stage] for stage in stages]
                })
                st.dataframe(data_show, use_container_width=True)

        # 如果当前有数据行且与上次存储的不同，则重置拟合状态（哈希包含口罩型号）
        if current_row is not None:
            new_row_hash = hash((
                tuple(current_row[['A_conc', 'B_conc', 'C_conc', 'D_conc',
                      'E_conc', 'F_conc', 'G_conc', 'H_conc', 'Filter_conc']].values),
                sel_mask
            ))
            old_row_hash = st.session_state.get('air_row_hash', None)
            if new_row_hash != old_row_hash:
                st.session_state.air_fitted = False
                st.session_state.air_fit_result = None
                if st.session_state.air_fig is not None:
                    plt.close(st.session_state.air_fig)
                st.session_state.air_fig = None
                st.session_state.air_row_hash = new_row_hash
                st.session_state.air_current_row = current_row
            else:
                st.session_state.air_current_row = current_row

        # 计算按钮
        if method_choice != "🌐 国标单AMAD（默认5μm）" and st.session_state.air_current_row is not None:
            if st.button("🔢 计算AMAD并绘图", key="calc_amad_button"):
                with st.spinner("拟合计算中..."):
                    current_row = st.session_state.air_current_row
                    conc_cols = ['A_conc', 'B_conc', 'C_conc', 'D_conc', 'E_conc',
                                 'F_conc', 'G_conc', 'H_conc', 'Filter_conc']
                    raw_activities = np.array(
                        [current_row[c] for c in conc_cols], dtype=float)
                    if np.sum(raw_activities) == 0:
                        st.error("数据全为零，无法计算")
                    else:
                        # 口罩校正
                        stage_midpoints = np.array(midpoints)
                        mask_corrected_activities = apply_mask_to_stages(
                            raw_activities, sel_mask, stage_midpoints)
                        total_raw = np.sum(raw_activities)
                        total_masked = np.sum(mask_corrected_activities)
                        st.info(
                            f"口罩校正: 原始总活度 {total_raw:.2e} Bq/m³ → 校正后总活度 {total_masked:.2e} Bq/m³ (防护因子={total_masked/total_raw:.4f})")

                        # 效率校正（用于拟合）
                        corrected = apply_efficiency_correction(
                            mask_corrected_activities)
                        total_corr = np.sum(corrected)

                        # 拟合
                        amad_uni, gsd_uni, total_uni = fit_unimodal(corrected)
                        amad1, gsd1, frac1, amad2, gsd2, total_bi = fit_bimodal(
                            corrected)
                        D50_lin, GSD_lin, D84_lin, D16_lin, R2_lin = fit_linear_probit(
                            mask_corrected_activities)

                        ratio_amads = amad1 / \
                            amad2 if (not np.isnan(amad2)
                                      and amad2 > 0) else np.nan
                        recommended = recommend_method(
                            amad2, ratio_amads, frac1, R2_lin)

                        # 保存拟合结果
                        st.session_state.air_fit_result = {
                            'method_details': {
                                'amad_uni': amad_uni, 'gsd_uni': gsd_uni, 'total_uni': total_uni,
                                'amad1': amad1, 'gsd1': gsd1, 'frac1': frac1,
                                'amad2': amad2, 'gsd2': gsd2, 'total_bi': total_bi,
                                'D50_lin': D50_lin, 'GSD_lin': GSD_lin, 'R2_lin': R2_lin,
                                'recommended': recommended
                            },
                            'raw_activities': raw_activities,
                            'mask_corrected_activities': mask_corrected_activities,
                            'corrected': corrected,
                            'total_corr': total_corr,
                            'total_raw': total_raw,
                            'total_masked': total_masked,
                            'mask_pf_overall': total_masked / total_raw if total_raw > 0 else 1.0
                        }
                        # 生成图像
                        if st.session_state.air_fig is not None:
                            plt.close(st.session_state.air_fig)
                        fig = plot_fitting_results(
                            workshop=current_row.get('Workshop', 'manual'),
                            sampling_id=current_row.get(
                                'SamplingID', 'manual'),
                            raw_activities=mask_corrected_activities,
                            corrected_activities=corrected,
                            amad_uni=amad_uni, gsd_uni=gsd_uni, total_uni=total_uni,
                            amad1=amad1, gsd1=gsd1, frac1=frac1,
                            amad2=amad2, gsd2=gsd2, total_bi=total_bi,
                            total_activity_corr=total_corr,
                            return_fig=True
                        )
                        st.session_state.air_fig = fig
                        st.session_state.air_fitted = True
                        st.rerun()

    # 如果已经拟合过，显示拟合结果和后续界面
    if st.session_state.air_fitted and st.session_state.air_fit_result is not None:
        fit_res = st.session_state.air_fit_result
        method_details = fit_res.get('method_details', None)
        recommended = method_details.get(
            'recommended', '') if method_details else ''
        amad_uni = method_details.get('amad_uni', 0) if method_details else 0
        gsd_uni = method_details.get('gsd_uni', 0) if method_details else 0
        amad1 = method_details.get('amad1', 0) if method_details else 0
        gsd1 = method_details.get('gsd1', 0) if method_details else 0
        frac1 = method_details.get('frac1', 0) if method_details else 0
        amad2 = method_details.get('amad2', 0) if method_details else 0
        gsd2 = method_details.get('gsd2', 0) if method_details else 0
        D50_lin = method_details.get(
            'D50_lin', np.nan) if method_details else np.nan
        GSD_lin = method_details.get(
            'GSD_lin', np.nan) if method_details else np.nan
        R2_lin = method_details.get(
            'R2_lin', np.nan) if method_details else np.nan

        total_raw = fit_res.get('total_raw', 1.0)
        total_masked = fit_res.get('total_masked', 1.0)

        if method_choice == "🌐 国标单AMAD（默认5μm）":
            amad_result = {'method': 'Std_AMAD', 'amad': 5.0, 'gsd': 1.5}
            br = fit_res.get('breathing_rate', 1.2)
            work_hours = fit_res.get('work_hours', 8.0)
        else:
            # 显示所有拟合结果，并让用户选择
            st.subheader("📐 拟合结果汇总")
            col_res = st.columns(3)
            with col_res[0]:
                st.metric(
                    "单峰拟合", f"AMAD={amad_uni:.2f} µm", f"GSD={gsd_uni:.2f}")
            with col_res[1]:
                if not np.isnan(amad2):
                    st.metric(
                        "双峰拟合（粗）", f"AMAD={amad1:.2f} µm", f"GSD={gsd1:.2f}, 占比={frac1 * 100:.1f}%")
                    st.metric(
                        "双峰拟合（细）", f"AMAD={amad2:.2f} µm", f"GSD={gsd2:.2f}")
                else:
                    st.info("双峰拟合未检测到显著双峰")
            with col_res[2]:
                if not np.isnan(D50_lin):
                    st.metric(
                        "Probit拟合", f"AMAD={D50_lin:.2f} µm", f"GSD={GSD_lin:.2f}, R²={R2_lin:.4f}")
                else:
                    st.info("Probit拟合失败")

            # 给出建议
            st.markdown("**建议：**")
            if "Bimodal" in recommended:
                st.success("✅ 推荐使用双峰拟合（数据呈现明显双峰）")
            elif "Linear" in recommended:
                st.success(f"✅ 推荐使用Probit拟合（R²={R2_lin:.4f} > 0.85）")
            else:
                st.info("推荐使用单峰拟合")

            # 用户选择
            method_choice_user = st.radio(
                "请选择用于剂量计算的方法",
                ["单峰拟合", "双峰拟合", "Probit拟合"],
                index=0,
                key="user_fit_choice"
            )
            if method_choice_user == "单峰拟合":
                amad_result = {'method': 'Unimodal',
                               'amad': amad_uni, 'gsd': gsd_uni}
            elif method_choice_user == "双峰拟合" and not np.isnan(amad2):
                amad_result = {
                    'method': 'Bimodal',
                    'amad1': amad1, 'amad2': amad2,
                    'frac1': frac1, 'frac2': 1 - frac1,
                    'gsd1': gsd1, 'gsd2': gsd2
                }
            else:
                if not np.isnan(D50_lin):
                    amad_result = {'method': 'Linear_Probit',
                                   'amad': D50_lin, 'gsd': GSD_lin}
                else:
                    amad_result = {'method': 'Unimodal',
                                   'amad': amad_uni, 'gsd': gsd_uni}
                    st.warning("双峰拟合无效或Probit失败，已回退到单峰拟合")

        # 显示拟合图
        if st.session_state.air_fig is not None:
            st.subheader("📈 拟合结果图")
            st.pyplot(st.session_state.air_fig)
            buf = BytesIO()
            st.session_state.air_fig.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            st.download_button("📥 下载图像", data=buf,
                               file_name=f"fitting_{st.session_state.air_current_row.get('Workshop', 'manual')}_{st.session_state.air_current_row.get('SamplingID', 'manual')}.png",
                               mime="image/png")

        # 核素选择与剂量计算
        if amad_result is not None:
            st.markdown("---")
            st.subheader("🧪 选择核素与呼吸参数")

            # 核素选择
            c_nuc_sel = st.columns([1, 1, 1])
            with c_nuc_sel[0]:
                if st.session_state.available_elements:
                    sel_el = st.selectbox(
                        "元素", st.session_state.available_elements, key="air_el")
                else:
                    sel_el = None
            with c_nuc_sel[1]:
                if sel_el and sel_el in st.session_state.element_nuclides_map:
                    sel_nuc = st.selectbox(
                        "核素", st.session_state.element_nuclides_map[sel_el], key="air_nuc")
                else:
                    sel_nuc = None

            # 气溶胶类型选择（多选）
            with c_nuc_sel[2]:
                if sel_nuc is not None:
                    df_n = load_nucl(sel_nuc)
                    if df_n is not None and 'aerosol_type' in df_n.columns:
                        ats = sorted(df_n['aerosol_type'].dropna().unique())
                        sel_ats = st.multiselect("气溶胶类型（可多选）", ats, default=[
                                                 ats[0]] if ats else [], key="air_ats")
                    else:
                        sel_ats = ["Unspecified"]
                else:
                    sel_ats = ["Unspecified"]

            # 呼吸参数
            st.subheader("🫁 呼吸参数")
            col_breath = st.columns(2)
            with col_breath[0]:
                br = st.number_input("呼吸速率 BR (m³/h)", min_value=0.5,
                                     max_value=3.0, value=1.2, step=0.1, key="breathing_rate")
            with col_breath[1]:
                work_hours = st.number_input(
                    "工作时长 T (h)", min_value=0.5, max_value=24.0, value=8.0, step=0.5, key="work_hours")

            # 浓度选择
            st.subheader("💧 活度浓度")
            use_corrected = st.checkbox(
                "使用口罩校正后的活度浓度", value=True, key="use_masked_conc")
            if use_corrected:
                default_conc = total_masked
                conc_label = "活度浓度(Bq/m³) (已口罩校正)"
            else:
                default_conc = total_raw
                conc_label = "活度浓度(Bq/m³) (原始)"
            conc = st.number_input(conc_label, value=float(
                default_conc), format="%.2f", key="air_conc")

            # 两个按钮：计算并显示剂量、添加到列表
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                calc_clicked = st.button("🔢 计算剂量", key="calc_dose_btn")
            with col_btn2:
                add_clicked = st.button("➕ 添加到空气采样法列表", key="add_to_list_btn")

            if 'last_dose_result' not in st.session_state:
                st.session_state.last_dose_result = None

            if calc_clicked and sel_nuc is not None and sel_ats:
                df_e = load_nucl(sel_nuc, full=True)
                if df_e is not None:
                    total_dose = 0
                    dose_details = []

                    # 对每个气溶胶类型分别计算剂量
                    for sel_at in sel_ats:
                        if amad_result['method'] == 'Bimodal':
                            dc1 = interp_dc(df_e, sel_at, amad_result['amad1'])
                            dc2 = interp_dc(df_e, sel_at, amad_result['amad2'])

                            if dc1 is not None and dc2 is not None:
                                # 分别计算两个峰的剂量，然后相加
                                dose1 = dc1 * conc * \
                                    amad_result['frac1'] * br * work_hours
                                dose2 = dc2 * conc * \
                                    amad_result['frac2'] * br * work_hours
                                dose_compound = dose1 + dose2
                                detail = f"{sel_at}: 粗峰({amad_result['amad1']:.1f}μm)贡献{dose1:.2e}Sv + 细峰({amad_result['amad2']:.1f}μm)贡献{dose2:.2e}Sv"
                            else:
                                # 如果某个峰的剂量系数无法计算，使用加权AMAD
                                ref_amad = amad_result['amad1'] * amad_result['frac1'] + \
                                    amad_result['amad2'] * amad_result['frac2']
                                dc = interp_dc(df_e, sel_at, ref_amad)
                                dose_compound = dc * conc * br * work_hours
                                detail = f"{sel_at}: 使用加权AMAD={ref_amad:.2f}μm计算"

                        else:  # 单峰或国标模式
                            ref_amad = amad_result['amad']
                            dc = interp_dc(df_e, sel_at, ref_amad)
                            dose_compound = dc * conc * br * work_hours
                            detail = f"{sel_at}: AMAD={ref_amad:.2f}μm"

                        total_dose += dose_compound
                        dose_details.append(detail)

                    # 同时计算防护前后的剂量
                    if use_corrected:
                        dose_raw = total_dose * \
                            (total_raw / total_masked) if total_masked > 0 else total_dose
                        dose_masked = total_dose
                    else:
                        dose_raw = total_dose
                        dose_masked = total_dose * \
                            (total_masked / total_raw) if total_raw > 0 else total_dose

                    st.session_state.last_dose_result = {
                        'total_dose': total_dose,
                        'dose_details': dose_details,
                        'dose_raw': dose_raw,
                        'dose_masked': dose_masked,
                        'br': br,
                        'work_hours': work_hours,
                        'conc': conc,
                        'sel_nuc': sel_nuc,
                        'sel_ats': sel_ats,
                        'amad_result': amad_result,
                        'use_corrected': use_corrected
                    }
                else:
                    st.session_state.last_dose_result = None
                    st.error("无法加载核素数据")

            # 显示上次计算结果
            if st.session_state.last_dose_result is not None:
                res = st.session_state.last_dose_result
                st.markdown("---")
                st.subheader("📊 计算结果")

                # 显示详细分解
                with st.expander("📐 剂量分解", expanded=True):
                    for detail in res['dose_details']:
                        st.write(f"- {detail}")

                # 显示总剂量
                col_d = st.columns(2)
                with col_d[0]:
                    st.metric("本次选用浓度下的总剂量", f"{res['total_dose']:.2e} Sv")
                with col_d[1]:
                    if res['use_corrected']:
                        st.metric("若使用原始浓度（无防护）", f"{res['dose_raw']:.2e} Sv")
                    else:
                        st.metric("若使用口罩校正浓度", f"{res['dose_masked']:.2e} Sv")

                # 显示计算参数
                with st.expander("📋 计算参数"):
                    st.markdown(f"""
                    | 参数 | 值 |
                    |------|-----|
                    | 呼吸速率 BR | {res['br']:.1f} m³/h |
                    | 工作时长 T | {res['work_hours']:.1f} h |
                    | 活度浓度 | {res['conc']:.2f} Bq/m³ |
                    | 核素 | {res['sel_nuc']} |
                    | 气溶胶类型 | {', '.join(res['sel_ats'])} |
                    | AMAD方法 | {res['amad_result']['method']} |
                    """)

            # 添加到列表
            if add_clicked and st.session_state.last_dose_result is not None:
                res = st.session_state.last_dose_result
                item = {
                    'id': st.session_state.next_air_id,
                    'nuclide': res['sel_nuc'],
                    'aerosol_types': ', '.join(res['sel_ats']),
                    'amad': res['amad_result']['amad'] if 'amad' in res['amad_result'] else f"{res['amad_result'].get('amad1', 0):.1f}/{res['amad_result'].get('amad2', 0):.1f}",
                    'method': res['amad_result']['method'],
                    'conc': res['conc'],
                    'br': res['br'],
                    'work_hours': res['work_hours'],
                    'dc': res['total_dose'] / res['conc'] / res['br'] / res['work_hours'] if res['conc'] > 0 and res['br'] > 0 and res['work_hours'] > 0 else 0,
                    'dose': res['total_dose'],
                    'dose_raw': res['dose_raw'],
                    'dose_masked': res['dose_masked']
                }
                st.session_state.air_items.append(item)
                st.session_state.next_air_id += 1
                st.success("已添加到列表！")
                st.rerun()
            elif add_clicked and st.session_state.last_dose_result is None:
                st.warning("请先点击「计算剂量」按钮")

    # 空气采样法列表（独立累计，显示防护前后剂量）
    st.markdown("---")
    st.subheader("📋 空气采样法列表（独立累计）")
    if st.session_state.air_items:
        # 显示表头
        col_titles = st.columns([2, 1, 1, 1, 1, 1, 1])
        col_titles[0].write("**核素/气溶胶**")
        col_titles[1].write("**AMAD(µm)**")
        col_titles[2].write("**呼吸参数**")
        col_titles[3].write("**所选浓度剂量(Sv)**")
        col_titles[4].write("**原始浓度剂量(Sv)**")
        col_titles[5].write("**口罩校正浓度剂量(Sv)**")
        col_titles[6].write("**操作**")
        for item in st.session_state.air_items:
            cols_item = st.columns([2, 1, 1, 1, 1, 1, 1])
            cols_item[0].write(
                f"**{item['nuclide']}**/{item['aerosol_types']}")
            cols_item[1].write(f"{item['amad']}")
            cols_item[2].write(
                f"BR={item['br']:.1f}, T={item['work_hours']:.1f}h")
            cols_item[3].metric("", f"{item['dose']:.2e}")
            cols_item[4].write(f"{item['dose_raw']:.2e}")
            cols_item[5].write(f"{item['dose_masked']:.2e}")
            if cols_item[6].button("🗑️", key=f"del_air_{item['id']}"):
                st.session_state.air_items = [
                    i for i in st.session_state.air_items if i['id'] != item['id']]
                st.rerun()
        total_dose_selected = sum(i['dose']
                                  for i in st.session_state.air_items)
        total_dose_raw = sum(i['dose_raw'] for i in st.session_state.air_items)
        total_dose_masked = sum(i['dose_masked']
                                for i in st.session_state.air_items)
        st.markdown("---")
        col_sum = st.columns(3)
        col_sum[0].metric("💉 所选浓度总剂量", f"{total_dose_selected:.2e} Sv")
        col_sum[1].metric("💉 原始浓度总剂量（无口罩）", f"{total_dose_raw:.2e} Sv")
        col_sum[2].metric("💉 口罩校正总剂量", f"{total_dose_masked:.2e} Sv")
        if st.button("🗑️ 清空空气采样列表"):
            st.session_state.air_items = []
            st.session_state.next_air_id = 1
            st.rerun()
    else:
        st.info("暂无数据，点击上方添加")

# 底部说明
st.markdown("---")
st.caption("集成化剂量计算系统 v2.4 | 监测法 + 空气采样法（内嵌口罩防护）| 各模块独立显示总剂量")
st.caption("✅ 已优化：添加呼吸参数(BR×T)、修正双峰逻辑（分别计算再相加）、支持多化合物类型叠加")

if __name__ == "__main__":
    print("✅ 请使用: streamlit run integrated_dose_full.py")

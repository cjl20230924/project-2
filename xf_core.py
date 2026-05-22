"""
xf.py 核心函数模块
整合自 xf.py，用于多峰对数正态分布拟合和正态概率图直线拟合
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares
from scipy.integrate import quad
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# ==================== 1. 粒径边界（全局）====================
stages_low = np.array([21.3, 14.8, 9.8, 6.0, 3.5, 1.6, 0.9, 0.5, 0.1])
stages_high = np.array([50.0, 21.3, 14.8, 9.8, 6.0, 3.5, 1.6, 0.9, 0.5])
cut_diameters = np.array([21.3, 14.8, 9.8, 6.0, 3.5, 1.6, 0.9, 0.5])
stage_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Filter']
midpoints = np.sqrt(stages_low * stages_high)

# 撞击效率（A~H + Filter），文献值
efficiency = np.array([0.52, 0.61, 0.78, 0.89, 0.95, 0.96, 0.97, 0.99, 1.0])


# ==================== 2. 对数正态函数 ====================

def lognormal_pdf(dp, amad, gsd):
    """对数正态分布概率密度函数"""
    sigma = np.log(np.maximum(gsd, 1e-10))
    mu = np.log(amad)
    return np.where((dp > 0) & (gsd > 0),
                    1.0 / (np.sqrt(2.0 * np.pi) * sigma) *
                    np.exp(- (np.log(dp) - mu)**2 / (2 * sigma**2)),
                    0.0)


def stage_integral(amad, gsd, low, high):
    """计算给定粒径区间内对数正态分布的积分"""
    if low >= high:
        return 0.0

    def integrand(ln_dp):
        return lognormal_pdf(np.exp(ln_dp), amad, gsd)
    integral, _ = quad(integrand, np.log(low), np.log(high))
    return integral


# ==================== 3. 单峰模型 ====================

def unimodal_activities(amad, gsd, total_act, low_bounds, high_bounds):
    """计算单峰模型下各级的活度"""
    acts = []
    for low, high in zip(low_bounds, high_bounds):
        frac = stage_integral(amad, gsd, low, high)
        acts.append(total_act * frac)
    return np.array(acts)


def residuals_unimodal(params, low_bounds, high_bounds, measured):
    """单峰模型残差"""
    amad, gsd, total_act = params
    pred = unimodal_activities(amad, gsd, total_act, low_bounds, high_bounds)
    return pred - measured


# ==================== 4. 双峰模型 ====================

def bimodal_activities(amad1, gsd1, frac1, amad2, gsd2, total_act, low_bounds, high_bounds):
    """计算双峰模型下各级的活度"""
    acts = []
    for low, high in zip(low_bounds, high_bounds):
        f1 = stage_integral(amad1, gsd1, low, high)
        f2 = stage_integral(amad2, gsd2, low, high)
        total_frac = frac1 * f1 + (1 - frac1) * f2
        acts.append(total_act * total_frac)
    return np.array(acts)


def residuals_bimodal(params, low_bounds, high_bounds, measured):
    """双峰模型残差"""
    amad1, gsd1, frac1, amad2, gsd2, total_act = params
    pred = bimodal_activities(
        amad1, gsd1, frac1, amad2, gsd2, total_act, low_bounds, high_bounds)
    return pred - measured


# ==================== 5. 效率校正 ====================

def apply_efficiency_correction(activities):
    """应用撞击效率校正"""
    return activities / efficiency


# ==================== 6. 单峰拟合（自动）====================

def fit_unimodal(activities_corrected):
    """拟合单峰对数正态分布"""
    total_act = np.sum(activities_corrected)
    bounds_uni = [(0.5, 50), (1.01, 5), (0.1*total_act, 10*total_act)]

    def obj(params):
        return np.sum(residuals_unimodal(params, stages_low, stages_high, activities_corrected)**2)
    result = differential_evolution(
        obj, bounds_uni, seed=42, maxiter=1500, popsize=15, disp=False)
    res = least_squares(residuals_unimodal, result.x,
                        bounds=([b[0] for b in bounds_uni], [b[1]
                                for b in bounds_uni]),
                        args=(stages_low, stages_high, activities_corrected),
                        method='trf')
    return res.x[0], res.x[1], res.x[2]


# ==================== 7. 双峰拟合（改进边界 + 后处理）====================

def fit_bimodal(activities_corrected):
    """拟合双峰对数正态分布"""
    total_act = np.sum(activities_corrected)
    coarse_low, coarse_high = 5.0, 30.0
    fine_low, fine_high = 0.5, 10.0
    bounds_bi = [(coarse_low, coarse_high), (1.01, 5), (0.1, 0.99),
                 (fine_low, fine_high), (1.01, 5), (0.1*total_act, 10*total_act)]

    def obj(params):
        return np.sum(residuals_bimodal(params, stages_low, stages_high, activities_corrected)**2)
    result = differential_evolution(
        obj, bounds_bi, seed=42, maxiter=2000, popsize=20, disp=False)
    res = least_squares(residuals_bimodal, result.x,
                        bounds=([b[0] for b in bounds_bi], [b[1]
                                for b in bounds_bi]),
                        args=(stages_low, stages_high, activities_corrected),
                        method='trf')
    amad1, gsd1, frac1, amad2, gsd2, total = res.x
    if amad1 < amad2:
        amad1, amad2 = amad2, amad1
        gsd1, gsd2 = gsd2, gsd1
        frac1 = 1 - frac1
    ratio = amad1 / amad2 if amad2 > 0 else 1
    if ratio < 1.5 or frac1 < 0.1 or (1-frac1) < 0.10:
        amad_uni, gsd_uni, _ = fit_unimodal(activities_corrected)
        return amad_uni, gsd_uni, 1.0, np.nan, np.nan, total_act
    return amad1, gsd1, frac1, amad2, gsd2, total


# ==================== 8. 直线拟合 正态概率图法 ====================

def fit_linear_probit(activities):
    """
    基于probit直线拟合计算AMAD和GSD

    参数:
        activities: 原始活度数组（9个值，对应A-H和Filter）

    返回:
        D50: 中位粒径
        GSD: 几何标准差
        D84, D16: 84%和16%分位数
        R2: 拟合优度
    """
    raw_acts = activities[:8]
    filter_act = activities[8]
    total = np.sum(raw_acts) + filter_act

    f = raw_acts / total
    f_filt = filter_act / total
    f_all = np.concatenate([f, [f_filt]])

    cum_from_fine = np.cumsum(np.flip(f_all))[:-1]
    cum_less = np.flip(cum_from_fine) * 100
    valid = (cum_less > 1) & (cum_less < 99)

    if np.sum(valid) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    x = np.log(cut_diameters[valid])
    y = norm.ppf(cum_less[valid] / 100)
    reg = LinearRegression().fit(x.reshape(-1, 1), y)
    slope, intercept = reg.coef_[0], reg.intercept_

    # 核心算法
    x_D50 = (0 - intercept) / slope
    D50 = np.exp(x_D50)

    x_D84 = (1 - intercept) / slope
    D84 = np.exp(x_D84)

    x_D16 = (-1 - intercept) / slope
    D16 = np.exp(x_D16)

    GSD = D84 / D50
    R2 = 1 - np.sum((y - reg.predict(x.reshape(-1, 1)))**2) / \
        np.sum((y - np.mean(y))**2)
    return D50, GSD, D84, D16, R2


# ==================== 9. AIC / BIC 信息准则 ====================

def calc_aic_bic(n, k, rss):
    """
    计算赤池信息准则 (AIC) 与贝叶斯信息准则 (BIC)

    参数:
        n:   样本数（数据点个数）
        k:   模型参数个数
        rss: 残差平方和

    返回:
        (aic, bic)
    """
    if rss <= 0 or n <= k:
        return np.inf, np.inf
    sigma2 = rss / n
    aic = n * np.log(sigma2) + 2 * k
    bic = n * np.log(sigma2) + k * np.log(n)
    return aic, bic


def calc_unimodal_stats(amad, gsd, total_act, activities):
    """
    单峰模型 AIC/BIC

    参数:
        amad, gsd, total_act: 拟合参数
        activities: 效率校正后的 9 级活度

    返回:
        (aic, bic) 或 (nan, nan)
    """
    if np.isnan(amad) or np.isnan(gsd):
        return np.nan, np.nan
    pred = unimodal_activities(amad, gsd, total_act, stages_low, stages_high)
    residuals = activities - pred
    rss = np.sum(residuals ** 2)
    n = len(activities)      # 9
    k = 3                     # AMAD, GSD, total_act
    return calc_aic_bic(n, k, rss)


def calc_bimodal_stats(amad1, gsd1, frac1, amad2, gsd2, total_act, activities):
    """
    双峰模型 AIC/BIC

    参数:
        amad1, gsd1, frac1, amad2, gsd2, total_act: 拟合参数
        activities: 效率校正后的 9 级活度

    返回:
        (aic, bic) 或 (nan, nan)
    """
    if np.isnan(amad2) or np.isnan(gsd2):
        return np.nan, np.nan
    pred = bimodal_activities(amad1, gsd1, frac1, amad2, gsd2,
                              total_act, stages_low, stages_high)
    residuals = activities - pred
    rss = np.sum(residuals ** 2)
    n = len(activities)      # 9
    k = 6                     # amad1, gsd1, frac1, amad2, gsd2, total_act
    return calc_aic_bic(n, k, rss)


def calc_probit_stats(activities):
    """
    正态概率图法 AIC/BIC（仅用有效数据点）

    参数:
        activities: 原始 9 级活度

    返回:
        (D50, GSD, R2, aic, bic, n_valid)
        若有效数据点不足则返回 (nan, ...)
    """
    raw_acts = activities[:8]
    filter_act = activities[8]
    total = np.sum(raw_acts) + filter_act
    if total <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0

    f = raw_acts / total
    f_all = np.concatenate([f, [filter_act / total]])
    cum_from_fine = np.cumsum(np.flip(f_all))[:-1]
    cum_less = np.flip(cum_from_fine) * 100
    valid = (cum_less > 1) & (cum_less < 99)

    if np.sum(valid) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, int(np.sum(valid))

    x = np.log(cut_diameters[valid])
    y = norm.ppf(cum_less[valid] / 100)
    reg = LinearRegression().fit(x.reshape(-1, 1), y)
    slope, intercept_ = reg.coef_[0], reg.intercept_

    ln_AMAD = -intercept_ / slope
    D50 = np.exp(ln_AMAD)
    ln_GSD = 1 / slope
    GSD = np.exp(ln_GSD)

    ss_res = np.sum((y - reg.predict(x.reshape(-1, 1))) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    n = len(y)               # 有效数据点数
    k = 2                     # slope, intercept
    aic, bic = calc_aic_bic(n, k, ss_res)

    return D50, GSD, R2, aic, bic, n


# ==================== 10. 数据质量检测 ====================

def count_nonzero_stages(activities, threshold=1e-8):
    """统计浓度 > threshold 的级数"""
    acts = np.asarray(activities, dtype=float)
    return int(np.sum(acts > threshold))


def is_low_activity(activities, abs_threshold=0.001, rel_threshold=0.01):
    """
    判断是否为低活度（接近本底）数据

    参数:
        activities:   原始 9 级浓度数组
        abs_threshold: 绝对阈值 (Bq/m³)，总浓度低于此值视为低活度
        rel_threshold: 相对阈值，最大单级 / 总活度超过此比例且总活度很小时触发

    返回:
        (is_low, reason_str)
    """
    acts = np.asarray(activities, dtype=float)
    total = float(np.sum(acts))

    if total < abs_threshold:
        return True, f"总活度 {total:.2e} Bq/m³ < 绝对阈值 {abs_threshold} Bq/m³"

    max_stage = float(np.max(acts))
    if total > 0 and max_stage / total > rel_threshold and total < abs_threshold * 10:
        return True, (f"单级占比过高 ({max_stage/total:.1%}) 且总活度偏低 "
                      f"({total:.2e} Bq/m³)，数据可能受本底影响")

    return False, ""


def data_quality_assessment(activities):
    """
    综合数据质量评估

    返回:
        dict with:
            n_nonzero:      非零级数
            n_stages:       总级数 (9)
            total_activity: 总活度
            is_low:         是否低活度
            low_reason:     低活度原因（若触发）
            quality_flag:   'good' / 'marginal' / 'poor'
    """
    acts = np.asarray(activities, dtype=float)
    total = float(np.sum(acts))
    n_nonzero = count_nonzero_stages(acts)
    is_low, low_reason = is_low_activity(acts)

    if is_low or n_nonzero < 4:
        quality_flag = 'poor'
    elif n_nonzero < 6:
        quality_flag = 'marginal'
    else:
        quality_flag = 'good'

    return {
        'n_nonzero': n_nonzero,
        'n_stages': 9,
        'total_activity': total,
        'is_low': is_low,
        'low_reason': low_reason,
        'quality_flag': quality_flag,
    }


# ==================== 11. 自动判断单双峰 ====================

def judge_distribution(workshop, amad1, amad2, frac1, linear_R2):
    """
    自动判断数据适合单峰还是双峰分布

    判断规则：
    - 如果双峰参数中细峰AMAD为NaN，返回Unimodal
    - 如果粗峰/细峰AMAD比值 >= 1.5 且 两峰占比都 >= 5%，返回Bimodal
    - 否则返回Unimodal
    """
    if np.isnan(amad2):
        return "Unimodal"
    ratio = amad1 / amad2 if amad2 > 0 else 1
    frac_fine = 1 - frac1
    if ratio >= 1.5 and frac1 >= 0.05 and frac_fine >= 0.05:
        return "Bimodal"
    else:
        return "Unimodal"


# ==================== 12. 自动推荐方法 ====================

def recommend_method(amad2, ratio_amads, frac1, linear_R2,
                     aic_uni=None, bic_uni=None,
                     aic_bi=None, bic_bi=None,
                     quality_flag='good',
                     n_nonzero=9):
    """
    自动推荐计算方法（v2: 加入 AIC/BIC 与数据质量判据）

    优先级：
      ① 数据质量差 (quality_flag='poor') → 强制国标法
      ② 物理真实双峰 → Bimodal
      ③ AIC/BIC 信息准则（双峰显著优于单峰时采纳）
      ④ 正态概率图 R² ≥ 0.85 → Probit
      ⑤ 其余 → Unimodal

    参数:
        amad2, ratio_amads, frac1, linear_R2: 原有参数
        aic_uni, bic_uni: 单峰 AIC/BIC
        aic_bi, bic_bi:   双峰 AIC/BIC
        quality_flag:     数据质量标志 ('good' / 'marginal' / 'poor')
        n_nonzero:        非零级数

    返回:
        推荐方法字符串
    """
    reasons = []

    # ── ① 数据质量强制检查 ──
    if quality_flag == 'poor':
        reasons.append(f"数据质量差（非零级数={n_nonzero}/9）")
        reasons.append("强烈建议使用国标单AMAD法（5 μm）")
        return "国标单AMAD法 (5 μm) [数据质量不足，强制推荐]"

    # ── ② 物理双峰判断 ──
    is_true_bimodal = False
    if not np.isnan(amad2) and amad2 > 0:
        frac_fine = 1 - frac1
        if ratio_amads >= 1.5 and frac1 >= 0.05 and frac_fine >= 0.05:
            is_true_bimodal = True

    if is_true_bimodal:
        reasons.append("检测到物理意义明确的粗细双峰")
        # 用 AIC/BIC 确认
        if (aic_bi is not None and aic_uni is not None
                and not np.isnan(aic_bi) and not np.isnan(aic_uni)):
            delta_aic = aic_uni - aic_bi
            if delta_aic > 2:
                reasons.append(f"ΔAIC={delta_aic:.1f}（双峰显著优于单峰）")
            reasons.append(f"AIC: 单峰={aic_uni:.1f}  双峰={aic_bi:.1f}")
            reasons.append(f"BIC: 单峰={bic_uni:.1f}  双峰={bic_bi:.1f}")
        return "Bimodal (Recommended)"

    # ── ③ AIC/BIC 辅助判断：双峰在统计上是否显著优于单峰 ──
    favor_bimodal_by_ic = False
    if (aic_bi is not None and aic_uni is not None
            and not np.isnan(aic_bi) and not np.isnan(aic_uni)):
        delta_aic = aic_uni - aic_bi
        delta_bic = bic_uni - bic_bi if (bic_uni is not None and bic_bi is not None) else 0
        if delta_aic > 4 and delta_bic > 2:
            # 双峰在高惩罚下仍显著优于单峰（说明数据确实需要更多参数）
            favor_bimodal_by_ic = True
            reasons.append(f"ΔAIC={delta_aic:.1f}  ΔBIC={delta_bic:.1f}（双峰信息准则显著占优）")

    if favor_bimodal_by_ic and not np.isnan(amad2):
        reasons.append(f"AIC: 单峰={aic_uni:.1f}  双峰={aic_bi:.1f}")
        reasons.append(f"BIC: 单峰={bic_uni:.1f}  双峰={bic_bi:.1f}")
        return "Bimodal (Recommended by AIC/BIC)"

    # ── ④ 正态概率图 ──
    if not np.isnan(linear_R2) and linear_R2 >= 0.85:
        if quality_flag == 'marginal':
            reasons.append(f"数据质量边际（非零级数={n_nonzero}），但仍可用正态概率图法")
        reasons.append(f"R²={linear_R2:.4f} ≥ 0.85")
        return "正态概率图法 (Recommended)"

    # ── ⑤ 回退单峰 ──
    reasons.append("未检测到双峰且正态概率图 R² 不足")
    if aic_uni is not None and not np.isnan(aic_uni):
        reasons.append(f"单峰 AIC={aic_uni:.1f}  BIC={bic_uni:.1f}")
    return "Unimodal (Recommended)"


# ==================== 13. 绘图函数 ====================

def plot_fitting_results(workshop, sampling_id, raw_activities, corrected_activities,
                         amad_uni, gsd_uni, total_uni,
                         amad1, gsd1, frac1, amad2, gsd2, total_bi,
                         total_activity_corr, output_dir='plots'):
    """绘制拟合结果图"""
    import matplotlib.pyplot as plt
    import os
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

    plt.figure(figsize=(18, 5))

    # 1. 单峰拟合图
    plt.subplot(1, 3, 1)
    plt.semilogx(x_plot, y_unimodal, 'b-', linewidth=2,
                 label=f'Unimodal AMAD={amad_uni:.1f} μm')
    plt.scatter(midpoints, data_density, color='red',
                s=60, edgecolor='black', zorder=5)
    plt.xlabel('Aerodynamic Diameter (μm)')
    plt.ylabel('dA/dln(dp) normalized')
    plt.title(f'Unimodal Fit | {workshop} {sampling_id}')
    plt.grid(ls='--', alpha=0.6)
    plt.xlim(0.1, 50)
    plt.ylim(0)
    plt.legend()

    # 2. 双峰拟合图
    plt.subplot(1, 3, 2)
    plt.semilogx(x_plot, y_bimodal, 'k-', linewidth=2, label='Total fit')
    if not np.isnan(amad2):
        plt.semilogx(x_plot, frac1*lognormal_pdf(x_plot, amad1,
                     gsd1), 'b--', label=f'Coarse {amad1:.1f} μm')
        plt.semilogx(x_plot, (1-frac1)*lognormal_pdf(x_plot, amad2,
                     gsd2), 'r--', label=f'Fine {amad2:.1f} μm')
    plt.scatter(midpoints, data_density, color='red',
                s=60, edgecolor='black', zorder=5)
    plt.xlabel('Aerodynamic Diameter (μm)')
    plt.title(f'Bimodal Fit | {workshop} {sampling_id}')
    plt.grid(ls='--', alpha=0.6)
    plt.xlim(0.1, 50)
    plt.ylim(0)
    plt.legend()

    # 3. 概率图法
    plt.subplot(1, 3, 3)

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

        ax = plt.gca()
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

        ax.set_title(f'正态概率图 | {workshop} {sampling_id}')
        ax.legend()

        y_pred = reg.predict(x.reshape(-1, 1))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        plt.text(0.5, 0.5, '数据不足，无法绘制正态概率图',
                 ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, f'{workshop}_{sampling_id}.png'), dpi=150)
    plt.close()


# ==================== 14. 批量处理主函数 ====================

def process_sampling_data(csv_file):
    """
    处理分级测量数据的主函数

    参数:
        csv_file: CSV文件路径

    返回:
        results_df: 包含所有采样点拟合结果的DataFrame
    """
    df = pd.read_csv(csv_file)
    conc_cols = ['A_conc', 'B_conc', 'C_conc', 'D_conc',
                 'E_conc', 'F_conc', 'G_conc', 'H_conc', 'Filter_conc']

    results = []
    for idx, row in df.iterrows():
        workshop = row['Workshop']
        sampling_id = row['SamplingID']
        raw = row[conc_cols].values.astype(float)
        if np.sum(raw) == 0:
            continue

        corr = apply_efficiency_correction(raw)
        total_corr = np.sum(corr)

        # 非线性拟合
        amad_uni, gsd_uni, total_uni = fit_unimodal(corr)
        amad1, gsd1, frac1, amad2, gsd2, total_bi = fit_bimodal(corr)

        # 直线拟合
        D50_lin, GSD_lin, D84_lin, D16_lin, R2_lin = fit_linear_probit(raw)

        # 自动判断分布类型
        dist_type = judge_distribution(workshop, amad1, amad2, frac1, R2_lin)

        # 自动计算粗细峰比值
        ratio_amads = amad1 / \
            amad2 if (not np.isnan(amad2) and amad2 > 0) else np.nan

        # 自动推荐最终方法
        recommend = recommend_method(amad2, ratio_amads, frac1, R2_lin)

        # 绘图
        try:
            plot_fitting_results(workshop, sampling_id, raw, corr,
                                 amad_uni, gsd_uni, total_uni,
                                 amad1, gsd1, frac1, amad2, gsd2, total_bi, total_corr)
        except Exception as e:
            print(f"绘图失败 {workshop}_{sampling_id}: {e}")

        results.append({
            'Workshop': workshop,
            'SamplingID': sampling_id,
            'Distribution': dist_type,
            'Linear_D50': round(D50_lin, 3),
            'Linear_GSD': round(GSD_lin, 3),
            'Linear_D84': round(D84_lin, 3),
            'Linear_D16': round(D16_lin, 3),
            'Linear_R2': round(R2_lin, 4),
            'Unimodal_AMAD': round(amad_uni, 3),
            'Unimodal_GSD': round(gsd_uni, 3),
            'Bimodal_Coarse_AMAD': round(amad1, 3),
            'Bimodal_Coarse_GSD': round(gsd1, 3),
            'Bimodal_Fine_AMAD': round(amad2, 3),
            'Bimodal_Fine_GSD': round(gsd2, 3),
            'Bimodal_Fine_Fraction': round(1-frac1, 3) if not np.isnan(frac1) else np.nan,
            'Recommended_Method': recommend
        })

    return pd.DataFrame(results)

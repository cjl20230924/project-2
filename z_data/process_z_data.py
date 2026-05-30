"""
process_z_data.py
=================
解析 z_data 目录下所有 *.txt 文件（ICRP z(t) 数据），
生成对应的 *_zdata.csv（单文件）+ all_zdata.csv（合并全量）。

列名规范化策略
--------------
原始列名可能含括号后缀、星号，例如：
  "Whole Body (Th-234)"  →  whole_body
  "Alimentary Tract* (Th-234)"  →  alimentary_tract
  "Lungs*"  →  lungs
规则：先去掉 (xxx) 括号内容，再去掉星号，再 strip，最后按固定映射表匹配。
未命中映射表的列按 snake_case 转换保留。

运行方式
--------
cd z_data
python process_z_data.py

输出
----
  U_235_zdata.csv  U_238_zdata.csv  Pu_239_zdata.csv  Pu_240_zdata.csv
  all_zdata.csv    (所有文件合并)
"""

import re
import sys
from pathlib import Path

import pandas as pd

# =====================================================================
#   列名映射（清理后的原始名 → 标准列名）
# =====================================================================
_COL_MAP = {
    'Time, days':                         'time_days',
    'Whole Body':                         'whole_body',
    'Urine (24-hour sample)':             'urine_24h',
    'Faeces (24-hour sample)':            'faeces_24h',
    'Alimentary Tract':                   'alimentary_tract',
    'Lungs':                              'lungs',
    'Skeleton':                           'skeleton',
    'Liver':                              'liver',
}

# 元数据键名映射（txt 文件中的 key → DataFrame 列名）
_META_KEYS = {
    'Radionuclide':   'radionuclide',
    'Route of Intake': 'route_of_intake',
    'Material':        'material',
    'AMTD/AMAD, µm':   'amad_um',
}


def _normalize_col(raw: str) -> str:
    """
    清理原始列名：
      1. 去掉括号及括号内内容，如 "(Th-234)" → ""
         注意保留"Urine (24-hour sample)"这类小括号 → 只去掉末尾的 (元素符号) 类型括号
         策略：去掉 "(<大写字母开头的内容>)" 模式
      2. 去掉星号
      3. strip
      4. 查映射表；未命中则 snake_case 转换
    """
    # 去掉末尾的 (元素-数字) 类型括号，如 "(Th-234)", "(Ra-226)"
    cleaned = re.sub(r'\s*\([A-Z][a-z]?-\d+\)\s*$', '', raw)
    # 去掉星号
    cleaned = cleaned.replace('*', '').strip()

    # 查映射表
    if cleaned in _COL_MAP:
        return _COL_MAP[cleaned]

    # 未命中：snake_case 转换（保底）
    return re.sub(r'[^a-zA-Z0-9]+', '_', cleaned).lower().strip('_')


def parse_z_t_file(filepath: str | Path, output_csv: str | Path = None) -> pd.DataFrame:
    """
    解析单个 z(t) txt 文件，返回 DataFrame；若指定 output_csv 则同时保存。

    参数:
        filepath    : 原始 txt 文件路径
        output_csv  : 输出 CSV 路径（可选）

    返回:
        DataFrame，列包括：
          radionuclide, route_of_intake, material, fA, amad_um,
          time_days, whole_body, urine_24h, faeces_24h,
          alimentary_tract, lungs, skeleton, liver
    """
    filepath = Path(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i].strip()

        # ── 数据块起始：以 "Radionuclide" 开头的行 ──
        if not line.startswith('Radionuclide'):
            i += 1
            continue

        # ── 读取元数据 ──
        metadata = {}
        while i < n and lines[i].strip():
            kv = lines[i].strip()
            if '\t' in kv:
                key, val = kv.split('\t', 1)
                key, val = key.strip(), val.strip()
                if key in _META_KEYS:
                    metadata[_META_KEYS[key]] = val
            i += 1

        # 跳过空行
        while i < n and not lines[i].strip():
            i += 1

        # ── 可选描述行（Committed Effective Dose per ...）──
        if i < n and 'Committed Effective Dose' in lines[i]:
            i += 1

        # ── 必须找到表头（含 "Time, days"）──
        if i >= n or 'Time, days' not in lines[i]:
            continue

        raw_header = lines[i].strip()
        i += 1

        # 解析列名
        raw_cols = [c.strip() for c in raw_header.split('\t')]
        std_cols = [_normalize_col(c) for c in raw_cols]

        # 处理 amad_um："-" 表示无粒径
        amad_raw = metadata.get('amad_um', '')
        amad_val = None if (amad_raw == '-' or amad_raw == '') else amad_raw

        # 从 material 提取 fA
        mat = metadata.get('material', '')
        fa_match = re.search(r'fA=([\dEe+\-.]+)', mat, re.IGNORECASE)
        fa_val = fa_match.group(1) if fa_match else ''

        # ── 读取数据行，直到空行 / 下一数据块 ──
        while i < n:
            data_line = lines[i].strip()
            if not data_line:
                i += 1
                continue
            if data_line.startswith('Radionuclide'):
                break   # 下一块开始，不推进 i（外层循环处理）

            parts = data_line.split('\t')
            # 列数不匹配时跳过（容错）
            if len(parts) < len(std_cols):
                i += 1
                continue

            record = {
                'radionuclide': metadata.get('radionuclide', ''),
                'route_of_intake': metadata.get('route_of_intake', ''),
                'material': mat,
                'fA': fa_val,
                'amad_um': amad_val,
            }
            for idx, col in enumerate(std_cols):
                val = parts[idx].strip()
                if val == '-':
                    record[col] = None
                else:
                    try:
                        record[col] = float(val)
                    except ValueError:
                        record[col] = val

            records.append(record)
            i += 1

    if not records:
        print(f'  [警告] {filepath.name}: 未解析到任何数据，请检查格式')
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # 数值化
    for c in ['time_days', 'amad_um']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 列排序：元数据列 → time_days → 样本列（固定顺序）
    meta_cols = ['radionuclide', 'route_of_intake', 'material', 'fA', 'amad_um', 'time_days']
    preferred_data_cols = ['whole_body', 'urine_24h', 'faeces_24h',
                           'alimentary_tract', 'lungs', 'skeleton', 'liver']
    extra_cols = [c for c in df.columns
                  if c not in meta_cols and c not in preferred_data_cols]
    present_data_cols = [c for c in preferred_data_cols if c in df.columns] + extra_cols
    ordered = [c for c in meta_cols if c in df.columns] + present_data_cols
    df = df[ordered]

    if output_csv:
        output_csv = Path(output_csv)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f'  {filepath.name} → {output_csv.name}: '
              f'{len(df)} 行, {len(df.groupby(["route_of_intake","material","amad_um"],dropna=False))} 个数据块')

    return df


def process_all(z_data_dir: str | Path = None, merge_output: str | Path = None) -> pd.DataFrame:
    """
    批量处理 z_data 目录下所有 *.txt 文件。

    参数:
        z_data_dir   : txt 文件所在目录（默认 = 本脚本所在目录）
        merge_output : 合并 CSV 输出路径（默认 <z_data_dir>/all_zdata.csv）

    返回:
        合并后的完整 DataFrame
    """
    if z_data_dir is None:
        z_data_dir = Path(__file__).parent
    z_data_dir = Path(z_data_dir)

    txt_files = sorted(z_data_dir.glob('*.txt'))
    if not txt_files:
        print(f'[警告] {z_data_dir} 下未找到任何 *.txt 文件')
        return pd.DataFrame()

    print(f'发现 {len(txt_files)} 个 txt 文件：{[f.name for f in txt_files]}')

    all_dfs = []
    for txt_path in txt_files:
        stem = txt_path.stem                         # 如 "U_235"
        csv_path = z_data_dir / f'{stem}_zdata.csv'
        df = parse_z_t_file(txt_path, csv_path)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print('[错误] 所有文件均未解析到数据')
        return pd.DataFrame()

    merged = pd.concat(all_dfs, ignore_index=True)

    if merge_output is None:
        merge_output = z_data_dir / 'all_zdata.csv'
    merged.to_csv(merge_output, index=False, encoding='utf-8')

    nucs = sorted(merged['radionuclide'].unique())
    routes = sorted(merged['route_of_intake'].unique())
    print(f'\n合并完成 → {Path(merge_output).name}')
    print(f'  总行数  : {len(merged)}')
    print(f'  核素    : {nucs}')
    print(f'  途径    : {routes}')
    print(f'  样本列  : {[c for c in merged.columns if c not in {"radionuclide","route_of_intake","material","fA","amad_um","time_days"}]}')

    return merged


# =====================================================================
#   入口
# =====================================================================
if __name__ == '__main__':
    z_dir = Path(__file__).parent
    merged_df = process_all(z_dir)

    print('\n--- 前 3 行预览 ---')
    print(merged_df.head(3).to_string())

    print('\n--- 每个核素 × 途径 × 物质 的行数统计 ---')
    if not merged_df.empty:
        summary = (merged_df
                   .groupby(['radionuclide', 'route_of_intake', 'material'],
                            dropna=False)
                   .size()
                   .reset_index(name='rows'))
        print(summary.to_string(index=False))

import re
import pandas as pd


def parse_z_t_file(filepath, output_csv):
    """
    解析包含多个数据块的剂量-含量转换函数文件，输出统一CSV。

    参数:
        filepath: 原始文本文件路径 (如 'U_235.txt')
        output_csv: 输出的CSV文件路径 (如 'U_235_dose_per_content.csv')
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []  # 存储每条记录（每个时间点的数据+元数据）

    i = 0
    n = len(lines)

    # 定义需要提取的元数据键名（与文件中的键名一致）
    meta_keys = {
        'Radionuclide': 'radionuclide',
        'Route of Intake': 'route_of_intake',
        'Material': 'material',
        'AMTD/AMAD, µm': 'amad_um'
    }

    while i < n:
        line = lines[i].strip()
        # 查找数据块起始行
        if line.startswith('Radionuclide'):
            metadata = {}
            # 读取元数据：从当前行开始，直到遇到空行
            while i < n and lines[i].strip() != '':
                kv_line = lines[i].strip()
                if '\t' in kv_line:
                    key, val = kv_line.split('\t', 1)
                    key = key.strip()
                    val = val.strip()
                    if key in meta_keys:
                        metadata[meta_keys[key]] = val
                i += 1
            # 跳过可能的空行
            while i < n and lines[i].strip() == '':
                i += 1

            # 寻找表格头部（以 "Committed Effective Dose" 开头的描述行，可跳过）
            if i < n and 'Committed Effective Dose' in lines[i]:
                i += 1
            # 表格列名行（应包含 "Time, days"）
            if i >= n or 'Time, days' not in lines[i]:
                continue  # 格式错误，跳过此块
            header_line = lines[i].strip()
            # 解析列名，清理特殊字符和单位，统一为英文列名
            raw_cols = [c.strip() for c in header_line.split('\t')]
            # 映射关系（标准化）
            col_map = {
                'Time, days': 'time_days',
                'Whole Body': 'whole_body',
                'Urine (24-hour sample)': 'urine_24h',
                'Faeces (24-hour sample)': 'faeces_24h',
                'Alimentary Tract*': 'alimentary_tract',
                'Lungs*': 'lungs',
                'Skeleton*': 'skeleton',
                'Liver*': 'liver'
            }
            std_cols = []
            for c in raw_cols:
                # 去除星号和末尾空格
                c_clean = c.replace('*', '').strip()
                std_cols.append(col_map.get(
                    c_clean, c_clean.replace(' ', '_').lower()))
            i += 1

            # 读取数据行，直到遇到空行或下一个数据块开始
            while i < n:
                data_line = lines[i].strip()
                if not data_line:
                    i += 1
                    continue
                # 如果遇到下一个数据块的开始，停止读取
                if data_line.startswith('Radionuclide'):
                    break
                parts = data_line.split('\t')
                if len(parts) < len(std_cols):
                    i += 1
                    continue
                # 构建当前时间点的记录
                record = metadata.copy()  # 包含 radionuclide, route_of_intake, material, amad_um
                # 从 material 中提取 fA 值
                fA_match = re.search(
                    r'fA=([\dEe\+\-\.]+)', record.get('material', ''))
                record['fA'] = fA_match.group(1) if fA_match else ''
                # 处理 AMAD：如果是 "-" 则设为空
                if record.get('amad_um') == '-':
                    record['amad_um'] = None
                # 解析数值列
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
        else:
            i += 1

    if not records:
        print("未解析到任何数据，请检查文件格式。")
        return

    # 将记录列表转换为DataFrame
    df = pd.DataFrame(records)

    # 调整列顺序：元数据列在前，然后是时间，最后是各器官/排泄物列
    meta_cols = ['radionuclide', 'route_of_intake',
                 'material', 'fA', 'amad_um']
    time_col = 'time_days'
    data_cols = [c for c in df.columns if c not in meta_cols and c != time_col]
    ordered_cols = meta_cols + [time_col] + sorted(data_cols)
    df = df[ordered_cols]

    # 保存CSV
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"成功解析 {len(records)} 条记录，保存至 {output_csv}")
    return df


# 使用示例
if __name__ == '__main__':
    df = parse_z_t_file('U_235.txt', 'U_235_zdata.csv')
    # 可选：打印前几行查看
    print(df.head())

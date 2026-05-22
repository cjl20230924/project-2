import pandas as pd, sys, os
sys.stdout.reconfigure(encoding='utf-8')

data_dir = 'processed_nuclide_files'

targets = ['Processed_U_238.parquet', 'Processed_U_235.parquet',
           'Processed_Pu_239.parquet', 'Processed_Pu_238.parquet',
           'Processed_Pu_240.parquet', 'Processed_H_3.parquet']

for fname in targets:
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print(f'\n=== {fname} NOT FOUND ===')
        continue
    df = pd.read_parquet(fpath)
    print(f'\n=== {fname} ===')
    print(f'  Radionuclide: {df["radionuclide"].iloc[0]}')
    ct = df[['compound', 'aerosol_type', 'route_of_intake', 'fA']].drop_duplicates()
    for _, row in ct.iterrows():
        c = str(row.compound) if row.compound is not None else 'None'
        a = str(row.aerosol_type) if row.aerosol_type is not None else 'None'
        r = str(row.route_of_intake)
        print(f'    compound={c:35s} aerosol={a:35s} route={r:12s} fA={row.fA}')

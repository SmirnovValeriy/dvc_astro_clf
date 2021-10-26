import pandas as pd
from astropy.table import Table


df_base = {}
df_base['0'] = pd.read_csv('data/cv0.csv')
df_base['1'] = pd.read_csv('data/cv1.csv')
df_sdss = pd.read_csv('data/sdss.gz_pkl')
postfixes = ['01', '02', '11', '12']

for postfix in postfixes:
    df_j = Table.read('data/j_cv'+postfix+'.fits', format='fits').to_pandas()
    df_csv = df_base[postfix[0]].merge(df_j,
                                       left_on=['nrow', 'ra', 'dec'],
                                       right_on=['nrow_', 'ra_', 'dec_'],
                                       how='inner')
    df_csv.to_csv('data/cv'+postfix+'.csv', index=False)

for postfix in postfixes:
    df_csv = pd.read_csv('data/cv'+postfix+'.csv')
    df_star = df_csv[df_csv['class'] == 'STAR'].merge(df_sdss,
                                                      left_on=['ra', 'dec'],
                                                      right_on=['ra', 'dec'],
                                                      how='inner')
    df_star['zspec'] = df_star['z']
    del df_star['z']
    df_csv = pd.concat([df_csv[df_csv['class'] != 'STAR'], df_star],
                       ignore_index=True)
    df_csv.to_csv('data/1cv'+postfix+'.csv', index=False)

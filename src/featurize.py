import pickle
import pandas as pd
import sys
import yaml


if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 featurize.py"
                     "features-pkl-path df-csv-path\n")
    sys.exit(1)
features = pd.read_pickle(sys.argv[1])
out_filepath = sys.argv[2]

params = yaml.safe_load(open('params.yaml'))['featurize']
column = params['column']
mode = params['mode']
overview = params['overview']

sdss = [
    i for i in features
    if 'sdss' in i and
    'decals' not in i and
    column not in i
]
decals = [
    i for i in features
    if 'decals' in i and
    'sdss' not in i and
    'psdr' not in i and
    column not in i
]
wise = [
    i for i in decals
    if 'Lw' in i and
    column not in i
]
ps = [
    i for i in features
    if 'psdr' in i and
    'decals' not in i and
    column not in i
]

sdss_wise = [
    'sdssdr16_u_cmodel-decals8tr_Lw1',
    'sdssdr16_u_cmodel-decals8tr_Lw2',
    'sdssdr16_g_cmodel-decals8tr_Lw1',
    'sdssdr16_g_cmodel-decals8tr_Lw2',
    'sdssdr16_r_cmodel-decals8tr_Lw1',
    'sdssdr16_r_cmodel-decals8tr_Lw2',
    'sdssdr16_i_cmodel-decals8tr_Lw1',
    'sdssdr16_i_cmodel-decals8tr_Lw2',
    'sdssdr16_z_cmodel-decals8tr_Lw1',
    'sdssdr16_z_cmodel-decals8tr_Lw2'
]
sdss_nwise = [
    'sdssdr16_g_cmodel-decals8tr_g',
    'sdssdr16_r_cmodel-decals8tr_r',
    'sdssdr16_z_cmodel-decals8tr_z'
]
ps_decals = [
    'psdr2_g_kron-decals8tr_Lw1',
    'psdr2_g_kron-decals8tr_Lw2',
    'psdr2_r_kron-decals8tr_Lw1',
    'psdr2_r_kron-decals8tr_Lw2',
    'psdr2_i_kron-decals8tr_Lw1',
    'psdr2_i_kron-decals8tr_Lw2',
    'psdr2_z_kron-decals8tr_Lw1',
    'psdr2_z_kron-decals8tr_Lw2',
    'psdr2_y_kron-decals8tr_Lw1',
    'psdr2_y_kron-decals8tr_Lw2'
]

postfixes = ['01', '02', '11', '12']
df = pd.DataFrame()
for postfix in postfixes:
    df_postfix = pd.read_csv('data/1cv'+postfix+'.csv')
    df = pd.concat([df, df_postfix])

features_dict = {
    'sdssdr16+wise_decals8tr': [],
    'psdr2+wise_decals8tr': [],
    'sdssdr16+all_decals8tr': [],
    'psdr2+all_decals8tr': [],
    'decals8tr': [],
    'sdssdr16+psdr2+wise_decals8tr': [],
    'sdssdr16+psdr2+all_decals8tr': []
}

if mode == 'not_j':
    features_dict['sdssdr16+wise_decals8tr'] = sdss + wise + sdss_wise
    features_dict['psdr2+wise_decals8tr'] = ps + wise + ps_decals
    features_dict['sdssdr16+all_decals8tr'] = sdss + decals + sdss_wise + \
        sdss_nwise
    features_dict['psdr2+all_decals8tr'] = ps + decals + ps_decals
    features_dict['decals8tr'] = decals
    features_dict['sdssdr16+psdr2+wise_decals8tr'] = sdss + ps + wise + \
        sdss_wise + ps_decals
    features_dict['sdssdr16+psdr2+all_decals8tr'] = sdss + ps + decals + \
        ps_decals + sdss_wise + sdss_nwise

elif mode == 'j':
    sdss_j = [
        'sdssdr16_u_psf',
        'sdssdr16_g_psf',
        'sdssdr16_r_psf',
        'sdssdr16_i_psf',
        'sdssdr16_z_psf',
        'sdssdr16_u_cmodel',
        'sdssdr16_i_cmodel'
    ]
    ps_j = [
        'psdr2_i_kron',
        'psdr2_y_kron',
        'psdr2_g_psf',
        'psdr2_r_psf',
        'psdr2_i_psf',
        'psdr2_z_psf',
        'psdr2_y_psf'
    ]
    wise_j = [
        'decals8tr_Lw1',
        'decals8tr_Lw2'
    ]
    nwise_j = [
        'decals8tr_g',
        'decals8tr_r',
        'decals8tr_z'
    ]

    groups = [sdss_j, ps_j, wise_j, nwise_j]
    for group in groups:
        for j, feat in enumerate(group):
            df[feat + '-j'] = df[feat] - df[column]
            group[j] = feat + '-j'

    features_dict['sdssdr16+wise_decals8tr'] = sdss + wise + sdss_wise + \
        sdss_j + wise_j
    features_dict['psdr2+wise_decals8tr'] = ps + wise + ps_decals + \
        ps_j + wise_j
    features_dict['sdssdr16+all_decals8tr'] = sdss + decals + \
        sdss_wise + sdss_nwise + sdss_j + wise_j + nwise_j
    features_dict['psdr2+all_decals8tr'] = ps + decals + ps_decals + \
        ps_j + wise_j + nwise_j
    features_dict['decals8tr'] = decals + wise_j + nwise_j
    features_dict['sdssdr16+psdr2+wise_decals8tr'] = sdss + ps + wise + \
        sdss_wise + ps_decals + ps_j + wise_j + sdss_j
    features_dict['sdssdr16+psdr2+all_decals8tr'] = sdss + ps + decals + \
        ps_decals + sdss_wise + sdss_nwise + sdss_j + nwise_j + ps_j + wise_j

else:
    raise Exception('Bad parameter mode')

classes_encoder = {'STAR': 1, 'QSO': 2, 'GALAXY': 3}
df.replace({'class': classes_encoder}, inplace=True)
df = df.drop_duplicates(subset=['nrow', 'ra', 'dec'])
df_out = df[features_dict[overview] + [column] * (mode == 'j')]
df_out['class'] = df['class'].values
df_out.to_csv(out_filepath, index=False)

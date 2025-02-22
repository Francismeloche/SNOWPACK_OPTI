import pandas as pd
import datetime
import numpy as np
import snowpro
import sys


print('in python script')
# Import the Manuafl fidelity observation
#df= pd.read_excel('./Wx_obs/Fid Manual Obs.xlsx')
#df['Timestamp'] = pd.to_datetime(df['ManualObservationTimeStamp'], format=format)
#df = df.set_index(pd.DatetimeIndex(df['Timestamp'])) # je set ma nouvelle datetime comme index
#df_grainOBS =df.loc[(df['Timestamp'] > '2018-10-01 00:00:00') & (df['Timestamp'] < '2019-04-29 00:00:00')]
#df_grainOBS = df_grainOBS[['Timestamp','Snowpack', 'HN24', 'WeightNew', 'WaterEquivalent', 'Density', 'SurfaceFormID', 'SurfaceSize']]

df_grainOBS = pd.read_pickle('./Wx_obs/FID_ManObs_precipSH.pkl')
df_grainOBS['SurfaceSize'] = pd.to_numeric(df_grainOBS['SurfaceSize'], errors='coerce')
df_grainOBS['SurfaceSize'].plot()
df_grainOBS['date'] = df_grainOBS['Timestamp'].dt.date
df_grainOBS['date'] = pd.to_datetime(df_grainOBS['date'])
df_grainOBS['SurfaceFormID'] = df_grainOBS['SurfaceFormID'].replace(float('nan'), np.nan)

#Load the snowpack profile
root = sys.argv[1]

snowpro_path = root + '/output/30_55_FID_1.0_PSUM_15m.pro'
profs,meta = snowpro.read_pro( snowpro_path, res='1d', keep_soil=False, consider_surface_hoar=True)

#get the snowsurface grain type from the simulation
date = []
gtype_season = []
gsize_season = []
for d,pro  in profs.items():
    date.append(d)
    gtype = pro['graintype']
    gsize = pro['grain size (mm)']
    if len(gtype) > 0 :
        gtype_season.append(gtype[-1][0])
        gsize_season.append(gsize[-1])
    else:
        gtype_season.append('NA')
        gsize_season.append('NA')

df_grainMOD = pd.DataFrame({'datetime':date, 'gtype': gtype_season, 'gsize': gsize_season})
df_grainMOD['date'] = df_grainMOD['datetime'].dt.date
df_grainMOD['date'] = pd.to_datetime(df_grainMOD['date'])
df_grainMOD =df_grainMOD.loc[(df_grainMOD['datetime'] > '2018-10-01 00:00:00') & (df_grainMOD['datetime'] <= '2019-05-1 00:00:00')]

#compute the similarity of grain type
def gtype2float(gtype_str):
    if gtype_str == 'PP':
        grainf = 1
    if gtype_str == 'DF':
        grainf = 2
    if gtype_str == 'RG':
        grainf = 3
    if gtype_str == 'FC':
        grainf = 4
    if gtype_str == 'FCxr':
        grainf = 4
    if gtype_str == 'DH':
        grainf = 5
    if gtype_str == 'MF':
        grainf = 6
    if gtype_str == 'MFcr':
        grainf = 6
    if gtype_str == 'WG':
        grainf = 6
    if gtype_str == 'IF':
        grainf = 7
    if gtype_str == 'SH':
        grainf = 8
    if gtype_str == 'PPgp':
        grainf = 9
    if gtype_str == 'PPsd':
        grainf = 9
    return grainf


def sim_gtype(g1,g2):
    g1f = gtype2float(g1)
    g2f = gtype2float(g2)
    F4l = 0.1
               #X   1PP  2DF  3RG  4FC  5DH  6MF  7IF  8SH  9PPgp
    GRAIN_G = [[0,  F4l, F4l, F4l, F4l, F4l, F4l, F4l, F4l, F4l], # X
              [F4l, 0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8], # 1PP
              [F4l, 0.2, 0.0, 0.2, 0.6, 1.0, 1.0, 1.0, 1.0, 0.6], # 2DF
              [F4l, 0.5, 0.2, 0.0, 0.6, 0.9, 1.0, 0.0, 1.0, 0.5], # 3RG
              [F4l, 0.8, 0.6, 0.6, 0.0, 0.2, 1.0, 0.0, 1.0, 0.2], # 4FC
              [F4l, 1.0, 1.0, 0.9, 0.2, 0.0, 1.0, 0.0, 1.0, 0.3], # 5DH
              [F4l, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.2, 1.0, 1.0], # 6MF
              [F4l, 1.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 1.0, 1.0], # 7IF
              [F4l, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0], # 8SH
              [F4l, 0.8, 0.6, 0.5, 0.2, 0.3, 1.0, 1.0, 1.0, 0.0]] # 9PPgp
    
    dist_g = GRAIN_G[g1f][g2f]
    return dist_g




dist_gtype_list = []
dist_gsize_list = []
gty_obs_list = []
gsz_obs_list = []
gty_mod_list = []
gsz_mod_list = []
date_list = []
for idx,d in df_grainOBS.iterrows():
    date = d['date']
    gobs = d['SurfaceFormID']
    gmod = df_grainMOD['gtype'].loc[df_grainMOD['date'] == date].values[0]

    gszobs = d['SurfaceSize']
    gszmod = df_grainMOD['gsize'].loc[df_grainMOD['date'] == date].values[0]
    if gobs is not np.nan and gmod is not np.nan:
        dist_g = sim_gtype(gobs,gmod)
        dist_gsize = ((gszobs - gszmod)**2)
    else:
        dist_g = np.nan
        dist_gsize = np.nan
    gty_obs_list.append(gobs)
    gsz_obs_list.append(gszobs)
    gty_mod_list.append(gmod)
    gsz_mod_list.append(gszmod)
    dist_gtype_list.append(dist_g)
    dist_gsize_list.append(dist_gsize)
    date_list.append(date)
dist_df = pd.DataFrame({'date': date_list, 'gtype_obs': gty_obs_list, 'gsize_obs':gsz_obs_list,'gtype_mod': gty_mod_list, 'gsize_mod':gsz_mod_list, 'dist_gtype':dist_gtype_list, 'dist_gsize':dist_gsize_list})

#group the result per snow grain type mod
grouped_gtype = dist_df.groupby('gtype_mod')['dist_gtype'].mean()
grouped_gsize = dist_df.groupby('gtype_mod')['dist_gsize'].mean()


# Compute the confusion matrix of the SH modelisation
SH_TP = dist_df[(dist_df['dist_gtype'] < 0.1) & (dist_df['gtype_obs'] == 'SH')].count()
SH_realmod = dist_df[(dist_df['dist_gtype'] < 0.1) & (dist_df['gtype_obs'] == 'SH')]
SH_TN = dist_df[(dist_df['dist_gtype'] < 0.1) & (dist_df['gtype_obs'] != 'SH')].count()
SH_FN = dist_df[(dist_df['dist_gtype'] > 0.9) & (dist_df['gtype_obs'] == 'SH')].count()
#SH_hit = dist_df[(dist_df['dist_gtype'] < 0.1) & (dist_df['gtype_mod'] == 'SH')].count()
SH_FP = dist_df[(dist_df['dist_gtype'] > 0.9) & (dist_df['gtype_mod'] == 'SH')].count()
SH_tot = dist_df[dist_df['gtype_obs'] == 'SH'].count()
SH_F1 = 2*SH_TP['gtype_obs']/(2*SH_TP['gtype_obs'] + SH_FP['gtype_obs'] + SH_FN['gtype_obs'])

# Compute the confusion matrix of the PP modelisation
PP_TP = dist_df[(dist_df['dist_gtype'] < 0.1) & (dist_df['gtype_obs'] == 'PP')].count()
PP_realmod = dist_df[(dist_df['dist_gtype'] < 0.1) & (dist_df['gtype_obs'] == 'PP')]
PP_TN = dist_df[(dist_df['dist_gtype'] < 0.1) & (dist_df['gtype_obs'] != 'PP')].count()
PP_FN = dist_df[(dist_df['dist_gtype'] > 0.9) & (dist_df['gtype_obs'] == 'PP')].count()
PP_FP = dist_df[(dist_df['dist_gtype'] > 0.9) & (dist_df['gtype_mod'] == 'PP')].count()
PP_tot = dist_df[dist_df['gtype_obs'] == 'PP'].count()
PP_F1 = 2*PP_TP['gtype_obs']/(2*PP_TP['gtype_obs'] + PP_FP['gtype_obs'] + PP_FN['gtype_obs'])

meanTPSH_dgsize = SH_realmod['dist_gsize'].mean()**0.5
meanTPPP_dgsize = PP_realmod['dist_gsize'].mean()**0.5
F1 = 1-(SH_F1*0.5 + PP_F1*0.5)
print('F1: ', F1)

#df_SHfinal = pd.DataFrame({'SH_TP':SH_TP['gtype_obs'], 'SH_TN':SH_TN['gtype_obs'], 'SH_FN':SH_FN['gtype_obs'], 'SH_FP':SH_FP['gtype_mod'], 'SH_tot':SH_tot['gtype_obs'], 'F1':F1, 'meanTPSH_dgsize':meanTPSH_dgsize}, index=[0])
#df_PPfinal = pd.DataFrame({'PP_TP':PP_TP['gtype_obs'], 'PP_TN':PP_TN['gtype_obs'], 'PP_FN':PP_FN['gtype_obs'], 'PP_FP':PP_FP['gtype_mod'], 'PP_tot':PP_tot['gtype_obs'], 'F1':F1, 'meanTPPP_dgsize':meanTPPP_dgsize}, index=[0])
#df_final = pd.DataFrame({'F1': F1, 'meanTPSH_dgsize': meanTPSH_dgsize}, index=[0])

#df_SHfinal.to_csv('./FID_OPTIMIZATION/SH/'+ filename +'_SH_confusion_matrix.csv')
#df_PPfinal.to_csv('./FID_OPTIMIZATION/PP/'+ filename +'_PP_confusion_matrix.csv')
#df_final.to_csv('./FID_OPTIMIZATION/tempo/'+ filename +'_F1.csv')

import pandas as pd
import numpy as np
import os

# shift angle for each camera
def lateral_shift(df, scale=(0.30,0.10)):
    # for recovery data, steering angles will have a larger scale
    # Steering Angles: Steer Left=(Negative), Drive Straight=Center(0.0), Steer Right=(Positive)
    cols = ['center','left','right','steering']
    df_sub = df.loc[:,cols]
    # left, right images: add correction based on scale input
    adjustment = lambda x,sign: x + sign*np.random.uniform(*scale, 1)[0]
    # left image,  add correction factor
    df_sub.loc[:, 'left_steering']  = df_sub['steering'].apply(lambda x: adjustment(x,1.0)  if x!=0.0 else x)
    # right image, subtract correction factor
    df_sub.loc[:, 'right_steering'] = df_sub['steering'].apply(lambda x: adjustment(x,-1.0) if x!=0.0 else x)
    return df_sub

def combine_dataset(df, include_center):
    # for recovery data, we should not include center steering
    # combine center, left, right images/adjust steering; left, right images included only if off center
    df_ct = df[df.steering == 0.0][['center', 'steering']].values
    df_lt = df[df.left_steering  != df.steering][['left', 'left_steering']].values
    df_rt = df[df.right_steering != df.steering][['right', 'right_steering']].values
    if include_center:
        df_comb = pd.DataFrame(np.concatenate((df_lt,df_ct,df_rt),axis=0), columns=['image', 'steering'])
    else:
        df_comb = pd.DataFrame(np.concatenate((df_lt,df_rt),axis=0), columns=['image', 'steering'])
    # shuffle combined data here
    return df_comb.sample(frac=1.0).reset_index(drop=True)

def reformat_csv(data_path, header=False):
    if not header:
        cols = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        df_drive = pd.read_csv(data_path + 'driving_log.csv', header=None)
        df_drive.columns = cols
    else:
        df_drive = pd.read_csv(data_path + 'driving_log.csv')
    # strip any white space or hard coded prefixes based on where data is saved originally
    reformat_loc = lambda data_path, x: data_path + ("IMG/" + x.split("IMG/")[-1]).strip()
    df_drive.loc[:,'center'] = df_drive['center'].apply(lambda x:  reformat_loc(data_path, x) )
    df_drive.loc[:,'left']   = df_drive['left'].apply(lambda x:    reformat_loc(data_path, x) )
    df_drive.loc[:,'right']  = df_drive['right'].apply(lambda x:   reformat_loc(data_path, x) )
    return df_drive

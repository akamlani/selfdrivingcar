import numpy as np
import pandas as pd
import os
import glob

from functools import reduce


def get_data(data_path):
    """
    Build a data dictionary maintaining the path for the source
    """
    image_types = os.listdir(data_path)
    class_type = data_path.split('data')[-1].strip('/')
    data_dict  = {imgtype: glob.glob(data_path+imgtype+'/*') for imgtype in image_types}
    ds_images  = pd.Series( reduce(lambda x,y: x+y, data_dict.values()) )
    print('Path: {}, Num Images: {}'.format(data_path, len(ds_images)))
    return data_dict, ds_images


def create_df(ds_vehicles, ds_nonvehicles):
    """
    Create a combined dataframe of vehicles and nonvehicles
    Create Source Type and Target as new columns
    """
    df_vehicles = pd.DataFrame([ds_vehicles, np.ones(len(ds_vehicles))]).T
    df_vehicles.columns = ['image', 'vehicle']
    df_nonvehicles = pd.DataFrame([ds_nonvehicles, np.zeros(len(ds_nonvehicles))]).T
    df_nonvehicles.columns = ['image', 'vehicle']
    df = pd.concat([df_vehicles, df_nonvehicles], axis=0)
    df['source'] = df.image.apply(lambda s: s.split('/')[2])
    df['vehicle'] = df.vehicle.astype(np.float64)
    return df

def get_data_full():
    """
    Get the data dictionary for vehicles, nonvehicles, and dataframe concatentated
    """
    data_path = 'data'
    vehicle_dir    = 'data/vehicles/'
    nonvehicle_dir = 'data/non-vehicles/'
    if not os.path.exists('data/vehicles.p'):
        vehicle_dict, vehicles = get_data(vehicle_dir)
        ds_vehicles = pd.Series( vehicles )                 # 8792
        ds_vehicles.to_pickle('data/vehicles.p')
    else:
        ds_vehicles = pd.read_pickle(os.path.join(data_path, 'vehicles.p'))

    if not os.path.exists('data/nonvehicles.p'):
        nonvehicle_dict, nonvehicles = get_data(nonvehicle_dir)
        ds_nonvehicles = pd.Series( nonvehicles )           # 8968
        ds_nonvehicles.to_pickle('data/nonvehicles.p')
    else:
        ds_nonvehicles = pd.read_pickle(os.path.join(data_path, 'nonvehicles.p'))

    df = create_df(ds_vehicles, ds_nonvehicles)
    return ds_vehicles, ds_nonvehicles, df




if __name__ == '__main__':
    # 1. read in vehicles and non-vehicles, desriptive statistics
    vehicle_dir    = 'data/vehicles/'
    nonvehicle_dir = 'data/non-vehicles/'
    ds_vehicles    = pd.Series( get_data(vehicle_dir) )           # 8792
    ds_nonvehicles = pd.Series( get_data(nonvehicle_dir) )        # 8968
    # serialize to disk
    ds_vehicles.to_pickle('data/vehicles.p')
    ds_nonvehicles.to_pickle('data/nonvehicles.p')

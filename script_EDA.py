#Phase 8 solo
#ver nulos phase 8
#Funciones para nan's para VALUE_FOB y total_used por vuelo (50% para arriba descartar)
#outliers y nan's
#suavizamiento

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Funcion que descarta todo lo que no sea FLIGHT_PHASE_COUNT = 8 o si el vuelo dura menos de una hora
# def phase8(df):
#     df_flight = df.groupby('Flight')
#     if df_flight.idxmax() - df_flight.idxmin() < 3600:
#         df = df.drop(columns= df_flight)
#     df = df[df['FLIGHT_PHASE_COUNT'] == 8]
#     return df

def longer_than_1_hour(df):
    df = df.reset_index()
    flight_durations = df.groupby('Flight')['UTC_TIME'].agg(['min', 'max'])
    flight_durations['duration'] = (flight_durations['max'] - flight_durations['min']).dt.total_seconds()
    long_flights = flight_durations[flight_durations['duration'] >= 3600].index
    df = df[df['Flight'].isin(long_flights)].set_index('UTC_TIME')

    return df

def phase8(df):
    # Filter for FLIGHT_PHASE_COUNT = 8
    df = df[df['FLIGHT_PHASE_COUNT'] == 8]
    return df


#Handling NaN's for FUEL_USED_1, FUEL_USED_2, FUEL_USED_3, FUEL_USED_4
def calculate_total_fuel_used(row):
    if any(pd.isna([row['FUEL_USED_1'], row['FUEL_USED_2'], row['FUEL_USED_3'], row['FUEL_USED_4']])):
        return np.nan
    else:
        return row['FUEL_USED_1'] + row['FUEL_USED_2'] + row['FUEL_USED_3'] + row['FUEL_USED_4']
    

# Getting the summary of NaN's for each column
def get_nan_summary(df):
    missing_summary_fob = df.groupby('Flight')['VALUE_FOB'].apply(lambda x: x.isnull().sum()).reset_index(name='missing_count_FOB')

    total_summary_fob = df.groupby('Flight')['VALUE_FOB'].count().reset_index(name='total_count_FOB')

    missing_summary_fuel = df.groupby('Flight')['TOTAL_FUEL_USED'].apply(lambda x: x.isnull().sum()).reset_index(name='missing_count_FUEL')

    total_summary_fuel = df.groupby('Flight')['TOTAL_FUEL_USED'].count().reset_index(name='total_count_FUEL')

    summary_fob = pd.merge(missing_summary_fob, total_summary_fob, on='Flight')
    summary_fuel = pd.merge(missing_summary_fuel, total_summary_fuel, on='Flight')

    summary = pd.merge(summary_fob, summary_fuel, on='Flight')

    # Calculate percentage of missing values for VALUE_FOB
    summary['missing_percentage_FOB'] = (summary['missing_count_FOB'] / (summary['missing_count_FOB'] + summary['total_count_FOB'])) * 100

    # Calculate percentage of missing values for FUEL_USED_1
    summary['missing_percentage_FUEL'] = (summary['missing_count_FUEL'] / (summary['missing_count_FUEL'] + summary['total_count_FUEL'])) * 100

    return summary

#Funcion borra los vuelos con VALUE_FOB y TOTAL_USED con un 50% de nulos o mas
def has_50_percent_or_more_nans(group, col):
    return group[col].isna().sum() >= 0.3 * len(group)

# Agrupar por 'Flight' y filtrar vuelos con 50% o más de NaNs en 'VALUE_FOB' o 'TOTL_FUEL_USED'
def remove_flights(df):
    return df.groupby('Flight').filter(lambda x: not (has_50_percent_or_more_nans(x, 'VALUE_FOB') or has_50_percent_or_more_nans(x, 'TOTAL_FUEL_USED')))

#Funcion para eliminar outliers
def remove_outliers(df):
    df = df[(df['VALUE_FOB'] < df['VALUE_FOB'].quantile(0.95)) & (df['VALUE_FOB'] > df['VALUE_FOB'].quantile(0.05))]
    df = df[(df['TOTAL_FUEL_USED'] < df['TOTAL_FUEL_USED'].quantile(0.95)) & (df['TOTAL_FUEL_USED'] > df['TOTAL_FUEL_USED'].quantile(0.05))]
    return df

def moving_average(df, column, window_size):
    return df.groupby('Flight')[column].transform(lambda x: x.rolling(window=window_size, min_periods=1, center=True).mean())

def additional_features(df):
    # Calculate the total fuel used for each flight
    df['TOTAL_FUEL_USED'] = df['FUEL_USED_1'] + df['FUEL_USED_2'] + df['FUEL_USED_3'] + df['FUEL_USED_4']
    
    # Calculate the total fuel loaded for each flight
    df['FUEL_LOADED_FOB'] = df.groupby(['Flight'])['VALUE_FOB'].transform('max')

    # The expected FOB at any point in time is the FOB at the beginning of the flight minus the total fuel used up to that point
    df['VALUE_FOB_EXPECTED'] = df['FUEL_LOADED_FOB'] - df['TOTAL_FUEL_USED'] 

    # The difference between the expected FOB and the actual FOB
    df['VALUE_FOB_DIFF'] = df['VALUE_FOB_EXPECTED'] -  df['VALUE_FOB'] # Potential fuel leak 
    
    return df
#Funcion 
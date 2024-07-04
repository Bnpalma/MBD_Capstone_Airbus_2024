#Phase 8 solo
#ver nulos phase 8
#Funciones para nan's para VALUE_FOB y total_used por vuelo (50% para arriba descartar)
#outliers y nan's
#suavizamiento

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.stats import zscore

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


# #Handling NaN's for FUEL_USED_1, FUEL_USED_2, FUEL_USED_3, FUEL_USED_4
# def calculate_total_fuel_used(row):
#     if any(pd.isna([row['FUEL_USED_1'], row['FUEL_USED_2'], row['FUEL_USED_3'], row['FUEL_USED_4']])):
#         return np.nan
#     else:
#         return row['FUEL_USED_1'] + row['FUEL_USED_2'] + row['FUEL_USED_3'] + row['FUEL_USED_4']
    

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
def has_x_percent_or_more_nans(group, col,x = 0.5):
    return group[col].isna().sum() >= x * len(group)

# Agrupar por 'Flight' y filtrar vuelos con 50% o más de NaNs en 'VALUE_FOB' o 'TOTL_FUEL_USED'
def remove_flights(df):
    return df.groupby('Flight').filter(lambda x: not (has_x_percent_or_more_nans(x, 'VALUE_FOB') or has_x_percent_or_more_nans(x, 'TOTAL_FUEL_USED')))

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
    df['ID'] = df['Flight'].astype(str) + '_' + df['MSN'].astype(str)
    return df



def plot_flights(df, random=False, flight=None):
    """
    Plots VALUE_FOB, VALUE_FOB_EXPECTED, and VALUE_FOB_DIFF for the flights in the dataframe.

    Args:
    df (pd.DataFrame): Dataframe containing flight data.
    random (bool): If True, plot 5 random flights. Default is False.
    flight (str or None): Specific flight to plot. If provided, only this flight will be plotted.

    Returns:
    None: Displays the plot.
    """
    if flight:
        flights = [flight]
    else:
        if random:
            flights = df['Flight'].sample(5).unique()
        else:
            flights = df['Flight'].unique()[:5]

    # Create a plotly subplots figure
    fig = sp.make_subplots(rows=len(flights), cols=1, shared_xaxes=False,
                           subplot_titles=[f'Flight {flight}' for flight in flights],
                           specs=[[{"secondary_y": True}] for _ in flights])

    for i, flight in enumerate(flights, start=1):
        df_flight = df[df['Flight'] == flight]
        max_diff = df_flight['VALUE_FOB_DIFF'].abs().max()
        # Add VALUE_FOB trace
        fig.add_trace(go.Scatter(
            x=df_flight.index,
            y=df_flight['VALUE_FOB'],
            mode='lines',
            name=f'Flight {flight} VALUE_FOB'
        ), row=i, col=1)
        # Add VALUE_FOB_EXPECTED trace
        fig.add_trace(go.Scatter(
            x=df_flight.index,
            y=df_flight['VALUE_FOB_EXPECTED'],
            mode='lines',
            name=f'Flight {flight} VALUE_FOB_EXPECTED'
        ), row=i, col=1)
        # Add VALUE_FOB_DIFF trace as secondary y-axis
        fig.add_trace(go.Scatter(
            x=df_flight.index,
            y=df_flight['VALUE_FOB_DIFF'],
            mode='lines',
            name=f'Flight {flight} VALUE_FOB_DIFF',
            line=dict(dash='dot')
        ), row=i, col=1, secondary_y=True)

        # Update secondary y-axis to center zero
        fig.update_yaxes(range=[-max_diff, max_diff], row=i, col=1, secondary_y=True, zeroline=True, zerolinewidth=2)

    # Update layout of the figure
    fig.update_layout(
        title='VALUE_FOB, VALUE_FOB_EXPECTED, and VALUE_FOB_DIFF for Flights',
        height=400 * len(flights)  # Adjust height based on number of flights
    )

    # Update y-axes titles
    for i in range(1, len(flights) + 1):
        fig.update_yaxes(title_text="VALUE_FOB / VALUE_FOB_EXPECTED", row=i, col=1, secondary_y=False)
        fig.update_yaxes(title_text="VALUE_FOB_DIFF", row=i, col=1, secondary_y=True)

    # Display the plot
    fig.show()
# Example usage (assuming df is your dataframe):
# plot_flights(df, random=True)

# def outliers_to_nan(df, column):
#     # detect outliers with IQR
#     q1 = df[column].quantile(0.25)
#     q3 = df[column].quantile(0.75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
#     df[column] = df[column].apply(lambda x: np.nan if x < lower_bound or x > upper_bound else x)
    
#     # # detect outliers with Z-score
#     # z = np.abs((df[column] - df[column].mean()) / df[column].std())
#     # df[column] = df[column].mask(z > 3)
    
#     return df

# Example usage (assuming df is your dataframe):
# plot_flights(df, random=True)



############################################################################################################
# from sklearn.linear_model import LinearRegression
# def filter_complete_flights(df):
#     """
#     Filters the flights with no missing values in the TOTAL_FUEL_USED column.
    
#     Parameters:
#     df (pd.DataFrame): The dataframe containing the flight data.
    
#     Returns:
#     pd.DataFrame: A dataframe with only the flights that have no missing TOTAL_FUEL_USED values.
#     """
#     # Group by 'Flight' and filter out the groups with any NaN in 'TOTAL_FUEL_USED'
#     complete_flights = df.groupby('Flight').filter(lambda x: x['TOTAL_FUEL_USED'].notna().all())
#     return complete_flights


# def calculate_average_slope(df):
#     """
#     Calculates the average slope of linear regressions for each complete flight.
    
#     Parameters:
#     df (pd.DataFrame): The dataframe containing the complete flight data.
    
#     Returns:
#     float: The average slope of the linear regressions.
#     """
#     slopes = []
    
#     # Group by 'Flight'
#     for flight, group in df.groupby('Flight'):
#         X = np.array(group.index).reshape(-1, 1)  # Using index as the time variable
#         y = group['TOTAL_FUEL_USED'].values
#         model = LinearRegression()
#         model.fit(X, y)
#         slopes.append(model.coef_[0])
    
#     average_slope = np.mean(slopes)
#     return average_slope


# def interpolate_missing_values(df, average_slope):
#     """
#     Interpolates missing values in the TOTAL_FUEL_USED column using the average slope.
    
#     Parameters:
#     df (pd.DataFrame): The dataframe containing the flight data.
#     average_slope (float): The average slope calculated from complete flights.
    
#     Returns:
#     pd.DataFrame: The dataframe with interpolated TOTAL_FUEL_USED values.
#     """
#     def interpolate_group(group):
#         last_valid_idx = group['TOTAL_FUEL_USED'].last_valid_index()
#         next_valid_idx = group['TOTAL_FUEL_USED'].first_valid_index()
        
#         if pd.isna(group['TOTAL_FUEL_USED']).any():
#             # Interpolating forward
#             for i in range(last_valid_idx + 1, len(group)):
#                 if pd.notna(group.at[i, 'TOTAL_FUEL_USED']):
#                     break
#                 group.at[i, 'TOTAL_FUEL_USED'] = group.at[last_valid_idx, 'TOTAL_FUEL_USED'] + average_slope * (i - last_valid_idx)
            
#             # Interpolating backward
#             for i in range(next_valid_idx - 1, -1, -1):
#                 if pd.notna(group.at[i, 'TOTAL_FUEL_USED']):
#                     break
#                 group.at[i, 'TOTAL_FUEL_USED'] = group.at[next_valid_idx, 'TOTAL_FUEL_USED'] - average_slope * (next_valid_idx - i)
        
#         return group
    
#     # Apply the interpolation function to each group
#     df = df.groupby('Flight').apply(interpolate_group)
#     return df

# def interpolate_missing_values(df, average_slope):
#     """
#     Interpolates missing values in the TOTAL_FUEL_USED column using the average slope.
    
#     Parameters:
#     df (pd.DataFrame): The dataframe containing the flight data.
#     average_slope (float): The average slope calculated from complete flights.
    
#     Returns:
#     pd.DataFrame: The dataframe with interpolated TOTAL_FUEL_USED values.
#     """
#     def interpolate_group(group):
#         last_valid_idx = group['TOTAL_FUEL_USED'].last_valid_index()
#         next_valid_idx = group['TOTAL_FUEL_USED'].first_valid_index()
        
#         if pd.isna(group['TOTAL_FUEL_USED']).any():
#             # Interpolating forward
#             for i in range(last_valid_idx + 1, len(group)):
#                 if pd.notna(group.at[i, 'TOTAL_FUEL_USED']):
#                     break
#                 group.at[i, 'TOTAL_FUEL_USED'] = group.at[last_valid_idx, 'TOTAL_FUEL_USED'] + average_slope * (i - last_valid_idx)
            
#             # Interpolating backward
#             for i in range(next_valid_idx - 1, -1, -1):
#                 if pd.notna(group.at[i, 'TOTAL_FUEL_USED']):
#                     break
#                 group.at[i, 'TOTAL_FUEL_USED'] = group.at[next_valid_idx, 'TOTAL_FUEL_USED'] - average_slope * (next_valid_idx - i)
        
#         return group
    
#     # Apply the interpolation function to each group
#     df = df.groupby('Flight').apply(interpolate_group)
#     return df



# def process_flight_data(df):
#     """
#     Processes the flight data to filter complete flights, calculate the average slope, and interpolate missing values.
    
#     Parameters:
#     df (pd.DataFrame): The dataframe containing the flight data.
    
#     Returns:
#     pd.DataFrame: The dataframe with interpolated TOTAL_FUEL_USED values.
#     """
#     # Step 1: Filter complete flights
#     complete_flights = filter_complete_flights(df)
    
#     # Step 2: Calculate average slope
#     average_slope = calculate_average_slope(complete_flights)
    
#     # Step 3: Interpolate missing values
#     interpolated_df = interpolate_missing_values(df, average_slope)
    
#     return interpolated_df


def interpolate_group(group):
    # Interpolate only inside the group
    return group.interpolate(method='time', limit_area='inside')



def mark_outliers(df, column, z_threshold):
    # Create copies of the DataFrame columns to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Calculate the difference for the specified column
    df_copy[f'{column}_diff'] = df_copy[column].diff().fillna(0) # Fill NaN with 0 for the diff calculation
    
    # Calculate the z-score for the specified column
    df_copy[f'{column}_zscore'] = np.abs(zscore(df_copy[f'{column}_diff']).fillna(0))  # Filling NaN with 0 for z-score calculation
    
    outliers = []
    reset = False
    
    for i in range(len(df_copy)):
        if i == 0:
            outliers.append(False)
            continue
        
        if reset:
            outliers.append(True)
            if df_copy[f'{column}_diff'].iloc[i] != 0:
                reset = False
            continue
        
        if df_copy[f'{column}_zscore'].iloc[i] > z_threshold:
            
               
            outliers.append(True)
            reset = True
        else:
            outliers.append(False)
    
    # Replace outliers with NaNs in the original column
    df_copy.loc[outliers, column] = np.nan
    
    # Optionally, drop the helper columns if they are no longer needed
    df_copy.drop(columns=[f'{column}_diff', f'{column}_zscore'], inplace=True)
    
    return df_copy


def apply_outliers_to_flights(df, column, z_threshold):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
    df_copy = df.copy()
    # Group by 'Flight' and apply the mark_outliers function to each group
    df_copy = df.groupby('Flight').apply(lambda group: mark_outliers(group.copy(), column, z_threshold))
    
    # Reset index if needed, because groupby may alter it
    df_copy.reset_index(drop=True, level=0, inplace=True)
    
    return df_copy



# def mark_outliers(df, column, z_threshold):
#     # Create copies of the DataFrame columns to avoid modifying the original DataFrame
#     df_copy = df.copy()
    
#     # Calculate the difference for the specified column
#     df_copy[f'{column}_diff'] = df_copy[column].diff().fillna(0)  # Fill NaN with 0 for the diff calculation
    
#     # Calculate the z-score for the specified column
#     df_copy[f'{column}_zscore'] = np.abs(zscore(df_copy[f'{column}_diff']).fillna(0))  # Filling NaN with 0 for z-score calculation
    
#     outliers = []
#     outlier_streak = False  # Track if we are in an outlier streak
    
#     for i in range(len(df_copy)):
#         if i == 0:
#             outliers.append(False)
#             continue
        
#         if df_copy[f'{column}_zscore'].iloc[i] > z_threshold:
#             outliers.append(True)
#             outlier_streak = True  # Mark the start of an outlier streak
#         elif outlier_streak:
#             outliers.append(True)  # Continue the outlier streak
#             if df_copy[f'{column}_diff'].iloc[i] == 0:
#                 outlier_streak = False  # End the outlier streak if difference is 0
#         else:
#             outliers.append(False)
    
#     # Replace outliers with NaNs in the original column
#     df_copy.loc[outliers, column] = np.nan
    
#     # Optionally, drop the helper columns if they are no longer needed
#     df_copy.drop(columns=[f'{column}_diff', f'{column}_zscore'], inplace=True)
    
#     return df_copy

# def apply_outliers_to_flights(df, column, z_threshold):
#     import warnings
#     warnings.filterwarnings("ignore", category=DeprecationWarning)
    
#     df_copy = df.copy()
#     # Group by 'Flight' and apply the mark_outliers function to each group
#     df_copy = df.groupby('Flight').apply(lambda group: mark_outliers(group.copy(), column, z_threshold))
    
#     # Reset index if needed, because groupby may alter it
#     df_copy.reset_index(drop=True, level=0, inplace=True)
    
#     return df_copy


def process_data_and_select_features(df):
    
    features = ['Flight','ID',
                'VALUE_FOB','VALUE_FOB_EXPECTED','VALUE_FOB_DIFF',
                'FUEL_USED_1','FUEL_USED_2','FUEL_USED_3','FUEL_USED_4','TOTAL_FUEL_USED','FUEL_LOADED_FOB',
                # 'VALUE_FUEL_QTY_CC1', 'VALUE_FUEL_QTY_CC2', 'VALUE_FUEL_QTY_CC3', 'VALUE_FUEL_QTY_CC4', 
                'VALUE_FUEL_QTY_CT', 'VALUE_FUEL_QTY_FT1',
                'VALUE_FUEL_QTY_FT2', 'VALUE_FUEL_QTY_FT3', 'VALUE_FUEL_QTY_FT4',
                'VALUE_FUEL_QTY_LXT', 'VALUE_FUEL_QTY_RXT']
    df['UTC_TIME'] = pd.to_datetime(df['UTC_TIME'])
    df.set_index('UTC_TIME',inplace=True)
    df.sort_index(inplace=True)
    ##########################################
    df = phase8(df) # order matters
    df = longer_than_1_hour(df)
    # df = remove_flights(df)
    # df = remove_outliers(df)

    df['FUEL_USED_1'] = df.groupby('Flight')['FUEL_USED_1'].apply(interpolate_group).values
    df['FUEL_USED_2'] = df.groupby('Flight')['FUEL_USED_2'].apply(interpolate_group).values
    df['FUEL_USED_3'] = df.groupby('Flight')['FUEL_USED_3'].apply(interpolate_group).values
    df['FUEL_USED_4'] = df.groupby('Flight')['FUEL_USED_4'].apply(interpolate_group).values

    df['VALUE_FOB'] = df.groupby('Flight')['VALUE_FOB'].apply(interpolate_group).values
    df = additional_features(df)
    df = apply_outliers_to_flights(df, 'VALUE_FOB', 3)
    df = apply_outliers_to_flights(df, 'TOTAL_FUEL_USED', 3)
    df['VALUE_FOB'] = df.groupby('Flight')['VALUE_FOB'].apply(interpolate_group).values
    df['TOTAL_FUEL_USED'] = df.groupby('Flight')['TOTAL_FUEL_USED'].apply(interpolate_group).values
    # Calculate this again after interpolation
    df['VALUE_FOB_DIFF'] = df['VALUE_FOB_DIFF'] = df['VALUE_FOB_EXPECTED'] -  df['VALUE_FOB']
    # df = df.dropna(subset=['VALUE_FOB', 'TOTAL_FUEL_USED'])
    
    # Drop na's (that we couldn't impute) and select features
    df = df.dropna()
    df = df[features]
    return df



def implant_leak(df, leak_flow):
    import numpy as np
    # LEAK_FLOW: flow rate of the leak in kg/s
    # Implant a leak in the fuel system
    acum = 0
    for idx in df.index[1:]:  # Skip the first row by slicing the index
        acum += leak_flow + np.random.normal(0, 0.1)
        df.at[idx, 'VALUE_FOB'] -= acum
        if df.at[idx, 'VALUE_FOB'] < 0:
            df.at[idx, 'VALUE_FOB'] = 0
    # recalculate diff
    df['VALUE_FOB_DIFF'] = df['VALUE_FOB_EXPECTED'] -  df['VALUE_FOB']
    return df
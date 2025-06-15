#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:24:55 2025

@author: lawrencebinding
"""

#%% Libraries
import numpy as np 
import pandas as pd 
import ggseg
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle 
from neuroCombat import neuroCombat
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#%% Find patients with paired TAU and CT scans 

#Set up data directory
data_dir='/Users/lawrencebinding/Desktop/projects/POND/ADNI_data/'

#Load in FDG data data
FDG_search = pd.read_csv(os.path.join('/Users/lawrencebinding/Desktop/projects/github/LDM-FDG/data/', 'FDG_preproc_search.csv'))
FDG_search['PTID'] = FDG_search['Subject ID'].str.replace('_', '-')
#Convert date to datetime format
FDG_search['EXAMDATE'] = pd.to_datetime(FDG_search['Archive Date'], format='%m/%d/%Y')

# Clean FDG search data by filtering rows based on keywords
keywords = ['Coreg', 'Avg', 'Std', 'Img', 'Vox', 'Siz', 'Uniform']
FDG_search = FDG_search[FDG_search['Description'].apply(lambda desc: all(keyword in desc for keyword in keywords) and 'dynamic' not in desc and 'dyn' not in desc and 'DYN' not in desc and 'DYNAMIC' not in desc and '4i/16s' not in desc)]

#Load in T1 data
T1_search = pd.read_csv(os.path.join('/Users/lawrencebinding/Desktop/projects/POND/p004_Workshop/', 'T1_search.csv'))
T1_search['PTID'] = T1_search['Subject ID'].str.replace('_', '-')
#Convert date to datetime format
T1_search['EXAMDATE'] = pd.to_datetime(T1_search['Archive Date'], format='%m/%d/%Y')

#Load in ADNI data
ADNImerge = pd.read_csv(os.path.join(data_dir, 'ADNIMERGE_12Mar2024.csv'))
ADNImerge['PTID'] = ADNImerge['PTID'].str.replace('_', '-')
#Convert date to datetime format
ADNImerge['EXAMDATE'] = pd.to_datetime(ADNImerge['EXAMDATE'], format='%d/%m/%Y')

#Load in Demographic data 
data_demographics = pd.read_csv(os.path.join(data_dir, 'Demographics_18Mar2024.csv'))
data_demographics['PTID'] = data_demographics['PTID'].str.replace('_', '-')
#Check which subjects don't have the correct PTDOB
mask = data_demographics['PTDOB'].fillna(0).astype(str).str.len() != 6
#Extract that data 
valid_data_demographics = data_demographics[~mask]
#Create a DD-mon-YYYY
valid_data_demographics['DOB'] = valid_data_demographics['PTDOB'].str.split('-').str[0]
valid_data_demographics['DOB'] = '01-' + valid_data_demographics['DOB']
valid_data_demographics['DOB'] = valid_data_demographics['DOB']+ '-' + (valid_data_demographics['PTDOBYY'].fillna(0).astype(int).astype(str))
valid_data_demographics['DOB'] = pd.to_datetime(valid_data_demographics['DOB'], format='%d-%b-%Y')
del data_demographics, mask

#Extract unique subject IDs
unique_subject_ids = FDG_search['PTID'].unique()

#Setup output list for dataframe 
adni_dataframe_list = []
amyl_dataframe_list = []
FDG_dataframe_list = []
T1_dataframe_list = []

# Loop through Subjects  
for subject_id in unique_subject_ids:
    #Extract all that subject tau-PET data 
    target_FDG_data = FDG_search[FDG_search['PTID'] == subject_id]
    #Extract all that subject ADNI-data
    target_adni_data = ADNImerge[ADNImerge['PTID'] == subject_id]
    #Extract demographics
    target_demo_data = valid_data_demographics[valid_data_demographics['PTID'] == subject_id]
    target_DOB = target_demo_data['DOB'].iloc[0]
    #Extract T1 data
    target_T1_data = T1_search[T1_search['PTID'] == subject_id]

    #Rank the dates
    target_FDG_data.sort_values(by='EXAMDATE', ascending=False, inplace=True)
    #Loop through tau-PET data and find scans or check-ups within 1y 
    for index, row in target_FDG_data.iterrows():
        both_within_year=False
        longitudinal_CT=False
        #Find the difference between scans for adni data 
        if not target_adni_data.empty:
            adni_value_diff = [((row['EXAMDATE'] - date) / np.timedelta64(1, 'D')) for date in target_adni_data['EXAMDATE']]
            adni_year_diff = [days / 365 for days in adni_value_diff]
            #Find the smallest index 
            adni_min_index = np.argmin(abs(np.array(adni_year_diff)))
            #Find the difference in scans for adni data
        if not target_T1_data.empty:
            T1_value_diff = [((row['EXAMDATE'] - date) / np.timedelta64(1, 'D')) for date in target_T1_data['EXAMDATE']]
            T1_year_diff = [days / 365 for days in T1_value_diff]
            #Find the smallest index 
            T1_min_index = np.argmin(abs(np.array(T1_year_diff)))
            #Find the index around 12 months
            T1_year_diff_mod = np.asarray(T1_year_diff.copy(),dtype=float) - T1_year_diff[T1_min_index]
            T1_year_diff_mod[T1_min_index] = 90
            T1_min_index_long = np.argmin(abs(np.array(T1_year_diff_mod)))

        
        #if patient tau-pet scan has adni data and amyl data within one year set break to true 
        if not target_adni_data.empty and not target_T1_data.empty:
            if (abs(adni_year_diff[adni_min_index]) < 1) & (abs(T1_year_diff[T1_min_index]) < 1):
                both_within_year=True 
                tau_pet_index=index
            if both_within_year is True:
                longitudinal_CT=True
            
        #break if true 
        if not target_adni_data.empty and not target_FDG_data.empty:
            if both_within_year is True:
                break 
    
    #If loop has exited with true, index and assign
    if both_within_year is True:
        #Assign out data 
        MostSimilarADNImerge_data = pd.DataFrame(target_adni_data.iloc[adni_min_index]).transpose()
        MostSimilarADNImerge_data['Tau_difference'] = adni_year_diff[adni_min_index]
        MostSimilarADNImerge_data['EXAMDATE'] = pd.to_datetime(MostSimilarADNImerge_data['EXAMDATE'])
        MostSimilarADNImerge_data['ADNI_Age'] = ((MostSimilarADNImerge_data['EXAMDATE'] - target_DOB).dt.days / 365.25).astype(int) #365.25 more accurate for calculating age 
        #
        MostSimilarT1_data = pd.DataFrame(target_T1_data.iloc[T1_min_index]).transpose()
        MostSimilarT1_data['Tau_difference'] = T1_year_diff[T1_min_index]
        MostSimilarT1_data['EXAMDATE'] = pd.to_datetime(MostSimilarT1_data['EXAMDATE'])
        MostSimilarT1_data['T1_Age'] = ((MostSimilarT1_data['EXAMDATE'] - target_DOB).dt.days / 365.25).astype(int)
        MostSimilarT1_data['PTID'] = subject_id
        MostSimilarT1_data['Timepoint'] = 1
        #
        MostSimilarFDG_data = pd.DataFrame(target_FDG_data.loc[tau_pet_index]).transpose() #This is using loc instead of iloc because its the direct index 
        MostSimilarFDG_data['SCANDATE'] = pd.to_datetime(MostSimilarFDG_data['EXAMDATE'])
        MostSimilarFDG_data['FDG_Age'] = ((MostSimilarFDG_data['SCANDATE'] - target_DOB).dt.days / 365.25).astype(int)
        #Extract the most recent diagnosis for this patient 
        target_adni_data.sort_values(by='EXAMDATE', ascending=False, inplace=True)
        latestDX = target_adni_data['DX'][pd.notna(target_adni_data['DX'])]
        #Output the most recent diagnosis
        if not latestDX.empty:
            MostSimilarADNImerge_data['Newest_DX'] = latestDX.iloc[0]
        else: 
            MostSimilarADNImerge_data['Newest_DX'] = np.nan
    else:
        #Output nans 
        MostSimilarADNImerge_data = pd.DataFrame(columns=target_adni_data.columns)
        MostSimilarADNImerge_data.loc[0] = np.nan
        MostSimilarADNImerge_data['PTID'] = subject_id
        
        MostSimilarT1_data = pd.DataFrame(columns=target_T1_data.columns)
        MostSimilarT1_data.loc[0] = np.nan
        MostSimilarT1_data['Tau_difference'] = np.nan
        MostSimilarT1_data['EXAMDATE'] = np.nan
        MostSimilarT1_data['T1_Age'] = np.nan
        MostSimilarT1_data['PTID'] = subject_id
        MostSimilarT1_data['Timepoint'] = 1


        MostSimilarFDG_data = pd.DataFrame(columns=target_FDG_data.columns)
        MostSimilarFDG_data.loc[0] = np.nan
        MostSimilarFDG_data['PTID'] = subject_id

        MostSimilarADNImerge_data['Newest_DX'] = np.nan
        
    
    #Output data 
    adni_dataframe_list.append(MostSimilarADNImerge_data)
    FDG_dataframe_list.append(MostSimilarFDG_data)
    T1_dataframe_list.append(MostSimilarT1_data)

#Concatinate list into dataframe 
ADNImerge_matched = pd.concat(adni_dataframe_list)
data_FDG_matched = pd.concat(FDG_dataframe_list)
data_T1_matched = pd.concat(T1_dataframe_list)

#%% Clean data and select cases with only full data 
    
# Reset index
ADNImerge_matched = ADNImerge_matched.reset_index(drop=True)
data_FDG_matched = data_FDG_matched.reset_index(drop=True)
data_T1_matched = data_T1_matched.reset_index(drop=True)

#Refactor the sites
data_FDG_matched['SITEID'] = data_FDG_matched['PTID'].str[:3]
data_FDG_matched['SITEID'] = pd.factorize(data_FDG_matched['SITEID'])[0]



# Find index of unacceptable data 
#   I should have made a note why I removed some of these... 
#   > Tau differences is related to the difference between CT and Tau scans 
#   > I then remove cases if they have nans for any row (why not impute?)
#   > I then remove tau / amyloid if they have SUVR +3 / 2.2? 
#      -> I couldn't work out why I did this so I removed it 
#   > Remove if they didn't have age 
rmv_r_idx = (
    (data_T1_matched.isna().all(axis=1)) |
    (data_FDG_matched.isna().all(axis=1)) |
    #(data_tau_targetDATA['META_TEMPORAL_SUVR'] > 3) |
    #(data_amyloid_matched['SUMMARY_SUVR'] > 2.2) |
    (data_FDG_matched['FDG_Age'].isna()) |
    data_T1_matched['Tau_difference'].isna())

# Remove index from all dataframes 
ADNImerge_matched = ADNImerge_matched[~rmv_r_idx]
data_FDG_matched = data_FDG_matched[~rmv_r_idx]
data_T1_matched = data_T1_matched[~rmv_r_idx]

# Find those with only one site  
category_counts = data_FDG_matched['SITEID'].value_counts()
is_unique_category = data_FDG_matched['SITEID'].map(category_counts) == 1

#Get overlap of rmv_r_idx and is_unique 
#rmv_r_idx = is_unique_category | rmv_r_idx

# Remove index from all dataframes 
ADNImerge_matched = ADNImerge_matched[~is_unique_category]
data_FDG_matched = data_FDG_matched[~is_unique_category]
data_T1_matched = data_T1_matched[~is_unique_category]

#Refactor the sites
data_FDG_matched['SITEID'] = pd.factorize(data_FDG_matched['SITEID'])[0]
# Reset index
ADNImerge_matched = ADNImerge_matched.reset_index(drop=True)
data_FDG_matched = data_FDG_matched.reset_index(drop=True)
data_T1_matched = data_T1_matched.reset_index(drop=True)

del category_counts, is_unique_category, rmv_r_idx

#%% 
#Extract those whose diagnosis doesn't match
mask_different_values = ADNImerge_matched['DX'] != ADNImerge_matched['Newest_DX']
Diagnosis_diff = ADNImerge_matched[mask_different_values]
#Exclude nans 
Diagnosis_diff = Diagnosis_diff[(pd.notna(Diagnosis_diff['DX']) & pd.notna(Diagnosis_diff['Newest_DX']))]
#Extract number of progressed individuals 
print("Number of MCI to DEM PTS:", len(Diagnosis_diff[(Diagnosis_diff['DX'] == 'MCI') & (Diagnosis_diff['Newest_DX'] == 'Dementia')]))
print("Number of CN to MCI PTS:", len(Diagnosis_diff[(Diagnosis_diff['DX'] == 'CN') & (Diagnosis_diff['Newest_DX'] == 'MCI')]))

#Extract index of those who regressed from MCI to CN, Dementia to CN or MCI 
regress_indices = ((ADNImerge_matched['DX'] == 'MCI') & (ADNImerge_matched['Newest_DX'] == 'CN') | \
                       (ADNImerge_matched['DX'] == 'Dementia') & (ADNImerge_matched['Newest_DX'] == 'MCI') | \
                       (ADNImerge_matched['DX'] == 'Dementia') & (ADNImerge_matched['Newest_DX'] == 'CN') \
                           )

ADNImerge_matched = ADNImerge_matched[~regress_indices]
data_FDG_matched = data_FDG_matched[~regress_indices]
data_T1_matched = data_T1_matched[~regress_indices]

# Reset index
ADNImerge_matched = ADNImerge_matched.reset_index(drop=True)
data_FDG_matched = data_FDG_matched.reset_index(drop=True)
data_T1_matched = data_T1_matched.reset_index(drop=True)

del mask_different_values, Diagnosis_diff, regress_indices 

#%% Save pandas dataframe 
data_FDG_matched.to_csv('/Users/lawrencebinding/Desktop/projects/github/LDM-FDG/data/FDG_data_matched.csv')
data_T1_matched.to_csv('/Users/lawrencebinding/Desktop/projects/github/LDM-FDG/data/T1_data_matched.csv')
ADNImerge_matched.to_csv('/Users/lawrencebinding/Desktop/projects/github/LDM-FDG/data/ADNImerge_data_matched.csv')


#%% Split into X amount of groups
#   ADNI IDA sucks so can only take 200ish subjects
# Split data_FDG_matched into CSVs with a maximum of 200 rows each
chunk_size = 200
output_dir = '/Users/lawrencebinding/Desktop/projects/github/LDM-FDG/data/split_FDG_data/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Split the dataframe into chunks and save each chunk as a separate CSV
for i, chunk in enumerate(range(0, len(data_FDG_matched), chunk_size)):
    chunk_data = data_FDG_matched.iloc[chunk:chunk + chunk_size]
    chunk_data.to_csv(os.path.join(output_dir, f'FDG_data_matched_part_{i + 1}.csv'), index=False)

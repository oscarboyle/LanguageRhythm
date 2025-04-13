import pandas as pd
import os
import json


def process_folkKern(file):
    """
    Reads folkKern file as dataframe
    """
    df = pd.read_csv(file)
    df['origin'] = 'folkKern'
    df['language'] = df['language'].combine_first(df['language.1'])
    df = df.dropna(subset=['nPVI'])
    df['year'] = pd.to_numeric(df['year'], errors='coerce')  # invalids become NaN
    df = df[(df['year'].isna()) | (df['year'] > 1850)]
    df.rename(columns={'artist_name': 'artist'}, inplace=True)
    
    return df

def process_OpenEWLD(file):
    """"
    Process OpenEWLD dataset, keeps only rows with year between 1850 and 1950 and converts
    genre to Folk since pop and rock did not exist in that time.
    """
    df = pd.read_csv(file)
    df['origin'] = 'OpenEWLD'
    df['language'] = df['language'].combine_first(df['language.1'])
    df = df.dropna(subset=['nPVI'])
    df['year'] = pd.to_numeric(df['year'], errors='coerce')  # invalids become NaN
    df = df[(df['year'].isna()) | ((df['year'] > 1850) & (df['year'] < 1950))]
    df['genre'] = 'Folk'
    df.rename(columns={'common_name': 'artist'}, inplace=True)

    return df

def process_PDMX(file):
    """
    Process OpenEWLD dataset, keeps only rows with data
    and removes duplicate columns
    """

    df = pd.read_csv(file)
    df['origin'] = 'PDMX'
    df['language'] = df['language'].combine_first(df['language.1'])
    df = df.dropna(subset=['nPVI'])
    df.rename(columns={'artist_name': 'artist','genres':'genre'}, inplace=True)

    return df 

def process_Wikifonia(file):
    """
    Process Wikifonia dataset, keeps only rows with data
    and removes duplicate columns
    """

    df = pd.read_csv(file)
    df['id'] = df.index+300000
    df['origin'] = 'Wikifonia'
    df = df.dropna(subset=['nPVI'])
    df.rename(columns={'Title':'title','Artist': 'artist','Genre':'genre','Year':'year',}, inplace=True)
    df.loc[df['genre'].str.contains('pop', case=False, na=False), 'genre'] = 'Pop'
    df.loc[df['genre'].str.contains('rock', case=False, na=False), 'genre'] = 'Rock'
    df.loc[df['genre'].str.contains('folk', case=False, na=False), 'genre'] = 'Folk'
    df = df[df['genre'].isin(['Pop', 'Rock', 'Folk'])]
    df['year'] = pd.to_numeric(df['year'], errors='coerce')  # invalids become NaN
    df = df[~((df['genre'].isin(['Pop', 'Rock'])) & (df['year'] < 1950))]
    return df


def combine_datasets(folkKern, OpenEWLD, PDMX, Wikifonia):
    combined = pd.concat([folkKern, OpenEWLD, PDMX,Wikifonia], ignore_index=True)
    combined = combined.drop_duplicates(subset=['title'], keep='first')
    # Convert to csv
    combined.to_csv('features/all_features.csv', index=False)
    return combined

folkKern = process_folkKern('features/folkKern_rhythmic_features.csv')
OpenEWLD = process_OpenEWLD('features/OpenEWLD_rhythmic_features.csv')
PDMX = process_PDMX('features/PDMX_rhythmic_features.csv')
Wikifonia = process_Wikifonia('features/notAnnotatedWiki_rhythmic_features.csv')

# Combine datasets
combined = combine_datasets(folkKern, OpenEWLD, PDMX, Wikifonia)

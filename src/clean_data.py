"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""
from __future__ import annotations
import pandas as pd



def clean_data(df=pd.DataFrame)-> pd.DataFrame:
    '''
    Clean raw dataset and return a clean DataFrame.
    '''
    df=df.copy()
    #Column name standardization
    df.columns=(
        df.columns.astype("str")
        .str.strip()
        .str.lower()
        .str.replace(" ","_",regex=False)
    )
    #Trim whitespaces
    obj_cols= df.select_dtypes(include="object").columns
    if len(obj_cols) >0:
        df[obj_cols]=(
                      df[obj_cols]
                      .apply(
                          lambda x: x.str.strip()
                                         )
                      )
    
    #Drop exact duplicates rows
    df=df.drop_duplicates()

    #Standardizing missing values
    df = df.replace(
        ["NA",
         "N/A",
         "",
         "?",
         "null",
         "None",
         "missing",
         -999,
         ],
        pd.NA
        )
    


    return df

if __name__ == "__main__":
    clean_data(df)
    print(df.head())

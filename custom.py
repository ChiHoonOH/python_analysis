# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:12:21 2018

@author: ChiHoon
"""

import pandas as pd
def struct(df):
    if isinstance(df, pd.DataFrame):
        df.info()
        print()        
        for element in df.columns:
            result = df[element]            
            print(element, ":", len(result.unique()), result.unique())

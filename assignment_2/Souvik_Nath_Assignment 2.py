# -*- coding: utf-8 -*-
"""
Author: Souvik Nath
Dated: 14.12.2019
"""

# Declare file name
file_name = 'dummy.xlsx'

import pandas as pd # Importing necessary libraries

xls = pd.ExcelFile(file_name) # Import the file

# Declaring empty dictionary to store all the sheets from the excel file
sheet_dict = {}

for sheet in xls.sheet_names:
    sheet_dict[sheet] = xls.parse(sheet)
    sheet_dict[sheet].to_csv(sheet+'.csv')
    
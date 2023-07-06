# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:12:12 2017
Updated 6/1/22 to replace df.set_value with df.at

A script to read and process Ian Richardson's ICME list.

Some pre-processing is required:
    Download the following webpage as a html file: 
        http://www.srl.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm
    Open in Excel, remove the year rows, delete last column (S) which is empty
    Save as a CSV.

@author: vy902033
"""
import pandas as pd
import os as os
import numpy as np
import re  #for dealing with non-numeric characters in a string of unknown length
from datetime import datetime
import matplotlib.pyplot as plt

def ICMElist(filepath = os.environ['DBOX'] + 
             'Data\\ICME_list\\List of Richardson_Cane ICMEs Since January1996_2022.csv'):
    
    
    icmes=pd.read_csv(filepath,header=None)
    #delete the first row
    icmes.drop(icmes.index[0], inplace=True)
    icmes.index = range(len(icmes))
    
    for rownum in range(0,len(icmes)):
        for colnum in range(0,3):
            #convert the three date stamps
            datestr=icmes[colnum][rownum]
            year=int(datestr[:4])
            month=int(datestr[5:7])
            day=int(datestr[8:10])
            hour=int(datestr[11:13])
            minute=int(datestr[13:15])
            #icmes.set_value(rownum,colnum,datetime(year,month, day,hour,minute,0))
            icmes.at[rownum,colnum] = datetime(year,month, day,hour,minute,0)
            
        #tidy up the plasma properties
        for paramno in range(10,17):
            dv=str(icmes[paramno][rownum])
            if dv == '...' or dv == 'dg' or dv == 'nan':
                #icmes.set_value(rownum,paramno,np.nan)
                icmes.at[rownum,paramno] = np.nan
            else:
                #remove any remaining non-numeric characters
                dv=re.sub('[^0-9]','', dv)
                #icmes.set_value(rownum,paramno,float(dv))
                icmes.at[rownum,paramno] = float(dv)
        
    
    #chage teh column headings
    icmes=icmes.rename(columns = {0:'Shock_time',
                                  1:'ICME_start',
                                  2:'ICME_end',
                                  10:'dV',
                                  11: 'V_mean',
                                  12:'V_max',
                                  13:'Bmag',
                                  14:'MCflag',
                                  15:'Dst',
                                  16:'V_transit'})
    return icmes
    
#icmes=ICMElist()             

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 16:33:52 2018

@author: ghawady ehmaid
"""

# determine the best match(es) for a single string

import pandas
import Levenshtein as lv
import datetime
from ngram import NGram
import jellyfish
#import os.path
#import numpy

#Below can be enabled or disabled depending on the need
enableED=1
enableNG=1
enableSX=1

start=datetime.datetime.now()
print("Start Time: ",start)
outfileprefix = 'Local_'
statisticsfilepath=outfileprefix+"statistics_interm.csv"

# load the entire dictionary into a list
f = open("../2018S2-90049P1-data/dict.txt",'r')
my_dict = f.readlines()
f.close()

#to continue from last saved record incase it crashes!
print("-------NEW Process--------------")
misspell_file=pandas.read_csv("../2018S2-90049P1-data/wiki_misspell.txt",header=None, names=['Input']) 
correct_file=pandas.read_csv("../2018S2-90049P1-data/wiki_correct.txt",header=None, names=['Correct'])

#print(fuzzy.Soundex(0)('abalienating'))

#matrix=numpy.column_stack((misspell_file,correct_file))
matrix=misspell_file
matrix['Correct']=correct_file
matrix['LVPrediction']=''
matrix['LVWeight']=0
matrix['LVCorrectWeight']=0
matrix['NG1Prediction']=''
matrix['NG1Weight']=0.0
matrix['NG1CorrectWeight']=0.0
matrix['NG2Prediction']=''
matrix['NG2Weight']=0.0
matrix['NG2CorrectWeight']=0.0
matrix['NG5Prediction']=''
matrix['NG5Weight']=0.0
matrix['NG5CorrectWeight']=0.0
matrix['SXPrediction']=''
matrix['SXWeight']=''

matrix['LVMatchFlag']=0
matrix['LVOneMatchFlag']=0
matrix['LVPredectionCount']=0
matrix['NG1MatchFlag']=0
matrix['NG1OneMatchFlag']=0
matrix['NG1PredectionCount']=0
matrix['NG2MatchFlag']=0
matrix['NG2OneMatchFlag']=0
matrix['NG2PredectionCount']=0
matrix['NG5MatchFlag']=0
matrix['NG5OneMatchFlag']=0
matrix['NG5PredectionCount']=0
matrix['SXMatchFlag']=0
matrix['SXOneMatchFlag']=0
matrix['SXPredectionCount']=0

#print(matrix)
#print("----------------")
#matrix['MatchFlag']=matrix.apply(lambda r: tuple(r), axis=4).apply(numpy.array)

matrix.to_csv(outfileprefix+"statistics_init.csv")
#print(matrix)
#print("----------------")
print(matrix.shape)
print("----------------")
print("----------------")

sx_string=''
x=0
j=0
for idx, line in matrix.iterrows():
    string = line['Input']
    correctstring = line['Correct'].strip()
    #print(line)
    lv_bestv=100000000 # This is intentionally overkill
    lv_bests =""
    ng1_bestv=-1 
    ng1_bests =""
    ng2_bestv=-1 
    ng2_bests =""
    ng5_bestv=-1 
    ng5_bests =""
    sx_bestv=0 
    sx_bests =""
    for entry in my_dict:
        dictentry=entry.strip()
        if enableED == 1:
            #print("LV CHECK----------------",string,dictentry)
            lv_thisv = lv._levenshtein.distance(string,dictentry)
            if(lv_thisv < lv_bestv):
                lv_bests = dictentry
                lv_bestv = lv_thisv
                #reset counters
                matrix.at[x,'LVPredectionCount'] = 1
                if(lv_bests == correctstring):
                    matrix.at[x,'LVCorrectWeight']=lv_thisv
            #regardless of attempt times, got a match then set the match flag to calculate recall
            elif(lv_thisv == lv_bestv):
                matrix.at[x,'LVPredectionCount'] += 1
                if(dictentry == correctstring):
                    #count other predictions of same weight
                    matrix.at[x,'LVCorrectWeight']=lv_thisv

        if enableNG == 1:
            #print("NG CHECK----------------",string,dictentry)
            ng1_thisv = NGram.compare(string,dictentry,N=10)
            if(ng1_thisv > ng1_bestv): #with ngram the higher the better
                ng1_bests = dictentry
                ng1_bestv = ng1_thisv
                #reset counters
                matrix.at[x,'NG1PredectionCount']=1
                if(ng1_bests == correctstring):
                    matrix.at[x,'NG1CorrectWeight']=ng1_thisv
            #regardless of attempt times, got a match then set the match flag to calculate recall
            elif(ng1_thisv == ng1_bestv):
                matrix.at[x,'NG1PredectionCount'] += 1
                if(dictentry == correctstring):
                    #count other predictions of same weight
                    matrix.at[x,'NG1CorrectWeight']=ng1_thisv
             
               
            #print("NG CHECK----------------",string,dictentry)
            ng2_thisv = NGram.compare(string,dictentry,N=2)
            if(ng2_thisv > ng2_bestv): #with ngram the higher the better
                ng2_bests = dictentry
                ng2_bestv = ng2_thisv
                #reset counters
                matrix.at[x,'NG2PredectionCount']=1
                if(ng2_bests == correctstring):
                    matrix.at[x,'NG2CorrectWeight']=ng2_thisv
            #regardless of attempt times, got a match then set the match flag to calculate recall
            elif(ng2_thisv == ng2_bestv):
                matrix.at[x,'NG2PredectionCount'] += 1
                if(dictentry == correctstring):
                    #count other predictions of same weight
                    matrix.at[x,'NG2CorrectWeight']=ng2_thisv
                    
            #print("NG CHECK----------------",string,dictentry)
            ng5_thisv = NGram.compare(string,dictentry,N=5)
            if(ng5_thisv > ng5_bestv): #with ngram the higher the better
                ng5_bests = dictentry
                ng5_bestv = ng5_thisv
                #reset counters
                matrix.at[x,'NG5PredectionCount']=1
                if(ng5_bests == correctstring):
                    matrix.at[x,'NG5CorrectWeight']=ng5_thisv
            #regardless of attempt times, got a match then set the match flag to calculate recall
            elif(ng5_thisv == ng5_bestv):
                matrix.at[x,'NG5PredectionCount'] += 1
                if(dictentry == correctstring):
                    #count other predictions of same weight
                    matrix.at[x,'NG5CorrectWeight']=ng5_thisv
           
        '''
        print("SX CHECK----------------",string,dictentry)
        print("SX CHECK----------------",fuzzy.Soundex(20)(string))
        print("SX CHECK----------------",fuzzy.Soundex(20)(dictentry))
        
        if(fuzzy.Soundex(20)(string)==fuzzy.Soundex(20)(dictentry)):
            sx_bests = dictentry
            sx_bestv = fuzzy.Soundex(20)(string)
    '''
        if enableSX == 1:
            sx_thisv=jellyfish.soundex(dictentry)
            sx_string=jellyfish.soundex(string)
            if(sx_string==sx_thisv):
                sx_bests = dictentry
                sx_bestv = sx_thisv
                if(sx_bests == correctstring):
                    matrix.at[x,'SXOneMatchFlag']=1
                #if(sx_thisv == jellyfish.soundex(correctstring)):
                #count other predictions of same weight
                matrix.at[x,'SXPredectionCount'] += 1
        LVmatch=(correctstring==dictentry);
        SXmatch=(sx_string==sx_thisv);
        #print(string,dictentry,correctstring,lv_thisv,ng1_thisv,ng2_thisv,ng5_thisv,sx_thisv)
        #print("NEXT CHECK----------------")
    
    #print("Value SET----------------")
    #print(x, string, bests, bestv,line[idx_correct])
    
    if enableED == 1:
        matrix.at[x,'LVPrediction']=lv_bests
        matrix.at[x,'LVWeight']=lv_bestv
        
        if(lv_bests.strip() == correctstring):
            matrix.at[x,'LVMatchFlag']=1
        if(matrix.at[x,'LVCorrectWeight'] == lv_bestv):
            matrix.at[x,'LVOneMatchFlag']=1
    
    if enableNG == 1:
        matrix.at[x,'NG1Prediction']=ng1_bests
        matrix.at[x,'NG1Weight']=ng1_bestv
        
        matrix.at[x,'NG2Prediction']=ng2_bests
        matrix.at[x,'NG2Weight']=ng2_bestv
        matrix.at[x,'NG5Prediction']=ng5_bests
        matrix.at[x,'NG5Weight']=ng5_bestv
        
        
        if(ng1_bests.strip() == correctstring):
            matrix.at[x,'NG1MatchFlag']=1
        
        if(ng2_bests.strip() == correctstring):
            matrix.at[x,'NG2MatchFlag']=1
        if(ng5_bests.strip() == correctstring):
            matrix.at[x,'NG5MatchFlag']=1
        
        if(matrix.at[x,'NG1CorrectWeight'] == ng1_bestv):
            matrix.at[x,'NG1OneMatchFlag']=1
        
        if(matrix.at[x,'NG2CorrectWeight'] == ng2_bestv):
            matrix.at[x,'NG2OneMatchFlag']=1
        if(matrix.at[x,'NG5CorrectWeight'] == ng5_bestv):
            matrix.at[x,'NG5OneMatchFlag']=1
        
    
    if enableSX == 1:
        matrix.at[x,'SXPrediction']=sx_bests
        matrix.at[x,'SXWeight']=sx_bestv
        
        if(sx_bests.strip() == correctstring):
            matrix.at[x,'SXMatchFlag']=1
    
    
    if(x>0 and x%10==0):
        print(datetime.datetime.now()-start,"Finished ",x," words...")
        #partial save to see how it is!
        matrix.to_csv(outfileprefix+"statistics_interm.csv")
        #print('Updating')
    x+=1

matrix.to_csv(outfileprefix+"statistics_final.csv")

end=datetime.datetime.now()
print("End Time: ",end)     
print("Run Time: ",end-start)  

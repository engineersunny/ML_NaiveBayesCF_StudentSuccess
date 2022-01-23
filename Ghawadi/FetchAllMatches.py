#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:00:03 2018

@author: ghawady ehmaid
"""

# A simple script which uses the "editdistance" package
# to determine the best match for a single string

import datetime
import pandas
import csv

import Levenshtein as lv
from ngram import NGram
import jellyfish


enableED=0
enableNG=1
enableSX=0

start=datetime.datetime.now()
print("Start Time: ",start)
outfileprefix = 'NG1_'

# load the entire dictionary into a list
f = open("../2018S2-90049P1-data/dict.txt",'r')
my_dict = f.readlines()
f.close()

print("-------NEW Process--------------")
misspell_file=pandas.read_csv("../2018S2-90049P1-data/wiki_misspell.txt",header=None, names=['Input']) 
correct_file=pandas.read_csv("../2018S2-90049P1-data/wiki_correct.txt",header=None, names=['Correct'])
matrix=pandas.read_csv("Local_statistics_Summary.csv");
#print(statistics['LVWeight'])
#print(statistics)
#matrix=numpy.column_stack((misspell_file,correct_file))
#print(matrix)
#print("----------------")
#matrix['MatchFlag']=matrix.apply(lambda r: tuple(r), axis=4).apply(numpy.array)

#print(matrix)
#print("----------------")
print("----------------")
x=0
j=1
'''polysaccaride
dictsize=len(my_dict)
wordsize=len(misspell_file)
maxsize=dictsize*wordsize
print('max dict size',dictsize)
print('max size',wordsize)
print('max size',maxsize)
predictionlist = ['']#*270000
''
LVWeightlist = ['']*size
missspelllist = ['']*size
SXWeightlist = ['']*size


print(type(predictionlist))
print(len(predictionlist))
print(type(my_dict))
print(len(my_dict))
'''

with open(outfileprefix+'datadump.csv','w') as dumpfile:
    dumpwriter=csv.writer(dumpfile,delimiter=',')
    dumpwriter.writerow(['Input','Correct','Prediction','Weight'])
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
                if(lv_thisv == line['LVWeight']):
                    #print("writing...",string,correctstring,dictentry,lv_thisv)
                    dumpwriter.writerow(['LV',string,correctstring,dictentry,lv_thisv])
            if enableNG == 1:
                
                #print("NG CHECK----------------",string,dictentry)
                ng1_thisv = NGram.compare(string,dictentry,N=1)
                #round used as the values in excel was rounded when saved from previous program run
                if(round(ng1_thisv,9) == line['NG1Weight']):
                    dumpwriter.writerow(['NG1',string,correctstring,dictentry,round(ng1_thisv,9)])
                
                #print("NG CHECK----------------",string,dictentry)
                ng2_thisv = NGram.compare(string,dictentry,N=2)
                if(round(ng2_thisv,9) == line['NG2Weight']):
                    dumpwriter.writerow(['NG2',string,correctstring,dictentry,round(ng2_thisv,9)])
                
                #print("NG CHECK----------------",string,dictentry)
                ng5_thisv = NGram.compare(string,dictentry,N=5)
                if(round(ng5_thisv,9) == line['NG5Weight']):
                    dumpwriter.writerow(['NG3',string,correctstring,dictentry,round(ng5_thisv,9)])
               
            if enableSX == 1:
                sx_thisv=jellyfish.soundex(dictentry)
                if(sx_thisv == line['SXWeight']):
                    dumpwriter.writerow(['SX',string,correctstring,dictentry,sx_thisv])
                    
        if(x>0 and x%5==0):
            print(datetime.datetime.now()-start,"Finished ",x," words...")
            #print to see something running!
            #break;
            
        x+=1
dumpfile.close()

end=datetime.datetime.now()
print("End Time: ",end)     
print("Run Time: ",end-start)  

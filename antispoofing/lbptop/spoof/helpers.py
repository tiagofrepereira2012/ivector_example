#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Mon Sep 17 2:15:00 CET 2012

"""
Helpers methods to retrieve informations from LBPTOP outputs
"""

"""
" Get a line from a text file
"""
def getLine(inputFile, lineNumber):

  #open the input file
  f = open(inputFile,'r')
  data = f.read()
  data = str.split(data,'\n')

  #Getting the raw threshold
  data = data[lineNumber]

  return data


"""
Open the perf_table.txt and get the development and test HTER for LBPTOP XY XT YT

In the perf_table.txt this in informations are in the line 28 and 29

@param inputFile Path to the input file
@return devel,test HTER

"""
def getLBPTOPHTER(inputFile):

  develLine = getLine(inputFile,27)
  testLine = getLine(inputFile,28)

  develLine = str.split(develLine,' ')
  develHTER = develLine[len(develLine)-2]

  testLine = str.split(testLine,' ')
  testHTER = testLine[len(testLine)-2]

  return float(develHTER.rstrip('%')),float(testHTER.rstrip('%'))


"""
Open the perf_table.txt and get the development and test HTER for LBP XY 

In the perf_table.txt this in informations are in the line 4 and 5

@param inputFile Path to the input file
@return devel,test HTER

"""
def getLBPHTER(inputFile):

  develLine = getLine(inputFile,3)
  testLine = getLine(inputFile,4)

  develLine = str.split(develLine,' ')
  develHTER = develLine[len(develLine)-2]

  testLine = str.split(testLine,' ')
  testHTER = testLine[len(testLine)-2]

  return float(develHTER.rstrip('%')),float(testHTER.rstrip('%'))


"""
Open the perf_table.txt and get it threshold for LBPTOP XY XT YT

With the PCA activated, the threshold is in the line 27

@param inputFile Path to the input File
@return Return the threshold value
"""
def getLBPTOPthreshold(inputFile):

  #open the input file
  f = open(inputFile,'r')
  data = f.read()
  data = str.split(data,'\n')

  #Getting the raw threshold
  data = data[26]
  data = str.split(data,':')
  
  threshold = float(data[1])
  
  return threshold 


"""
Open the perf_table.txt and get it threshold

With the PCA activated, the threshold is in the line 3

@param inputFile Path to the input File
@return Return the threshold value
"""
def getLBPthreshold(inputFile):

  #open the input file
  f = open(inputFile,'r')
  data = f.read()
  data = str.split(data,'\n')

  #Getting the raw threshold
  data = data[2]
  data = str.split(data,':')

  threshold = float(data[1])

  return threshold


"""
" Get the plane index in the parameter file
" XY - 0
" XT - 1
" YT - 2
" XY-YT - 3
" XY-XT-YT - 4
"""
def getPlaneIndex(planeName):
  lut = {'XY':0,'XT':1,'YT':2,'XT-YT':3,'XY-XT-YT':4}
  return lut[planeName]


#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Tue Sep 04 2:56:00 CET 2012

"""Support methods to organize scores
"""

import numpy
import bob

"""
" For each LBPTOP plane and it combinations save, in a HDF5 file, the following data:
"
" For each LBP-TOP plane
"   Score Predicted_Label Correct_Label
"     *         *              *
"     *         *              *
"
" Label = 0 - Real Access
" Label = 1 - Attack
"""
def saveLBPTOPScoresPredictions(realScores,attackScores,thres,outputFile):

  realScoresSize   = len(realScores[0])
  attackScoresSize = len(attackScores[0])
  valuesSize = realScoresSize + attackScoresSize

  data = numpy.zeros(shape=(5,valuesSize,3))

  #For each LBP-TOP plane and it combinations
  for i in range(5):
    
     ##Predicting real acess
     #Assign scores
     data[i,0:realScoresSize,0] = realScores[i]
     
     #Predicted labels
     #data[i,0:realScoresSize,1] = not(realScores[i] > thres[i])
     predictions = realScores[i] > thres[i]
     predictions = numpy.array([not(item) for item in predictions]) #Just to be readable
     data[i,0:realScoresSize,1] = predictions

     #Expected labels
     data[i,0:realScoresSize,2] = 0


     ##Predicting attacks acess
     #Assign scores
     data[i,realScoresSize:valuesSize,0] = attackScores[i]
     
     #Predicted labels
     data[i,realScoresSize:valuesSize,1] = (attackScores[i] <= thres[i])

     #Expected labels
     data[i,realScoresSize:valuesSize,2] = 1


  hdf5File = bob.io.HDF5File(outputFile, openmode_string='w')
  hdf5File.set('data',data)
  del hdf5File


"""
" Read the HDF5 file in the format below for ONE unique plane
"
"   Score Predicted_Label Correct_Label
"     *         *              *
"     *         *              *
"
" This function will return the realAccess Scores, Attack Scores, The predicted labels and the expected labels
"
"
" Label = 0 - Real Access
" Label = 1 - Attack
"
"""
def readLBPTOPOnePlaneData(data):

  [realAccessScores, attacksScores] = getScores(data)

  predictedLabels = data[:,1]
  expectedLabels  = data[:,2]

  return [realAccessScores, attacksScores, predictedLabels, expectedLabels]


"""
" Extract real access and attack scores from the defined format below
"
"   Score Predicted_Label Correct_Label
"     *         *              *
"     *         *              *

"
"""
def getScores(rawData):
  ##Reading devel scores
  indexes = numpy.where(rawData[:,2]==0)[0]
  realAccessScores = rawData[indexes][:,0]

  indexes = numpy.where(rawData[:,2]==1)[0]
  attacksScores = rawData[indexes][:,0]

  #Converting to use in BOB
  realAccessScores = numpy.array(realAccessScores,copy=True,order='C',dtype='float')
  attacksScores    = numpy.array(attacksScores,copy=True,order='C',dtype='float')

  return [realAccessScores, attacksScores]




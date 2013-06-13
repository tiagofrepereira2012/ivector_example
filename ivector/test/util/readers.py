###################################################################################
# @file readers.py                                                                #
# @date Mai 03 2013                                                               #
# @author Mario Uliani Neto <uliani@cpqd.com.br>                                  #
#                                                                                 #
# Copyright (C) 2013 CPqD - Research and Development Center in Telecommunications #
#                               Campinas, Brazil                                  #
###################################################################################


import os
import numpy
import array


def gmmread(fileUBM, N_MIXTURES, DIMENSION):
  """ 
  Reads a binary file containing a GMM.
  Returns a vector with the GMM parameters (weights, means and variances)
  """
  numberOfFloats = os.path.getsize(fileUBM)/4  # each parameter is a float
  file = open(fileUBM,mode = 'rb')  # opens GMM file
  parameters_GMM = array.array('f')
  parameters_GMM.fromfile(file,numberOfFloats)
  parameters_GMM = numpy.array(parameters_GMM, dtype=numpy.float64)
  file.close()

  weights   = []
  means     = []
  variances = []
  weights = parameters_GMM[0:256]
  for i in range(N_MIXTURES):
    means.append(parameters_GMM[(i)*DIMENSION*2+N_MIXTURES:(i)*DIMENSION*2+DIMENSION+N_MIXTURES])
    variances.append(parameters_GMM[(i)*DIMENSION*2+DIMENSION+N_MIXTURES:(i)*DIMENSION*2+DIMENSION*2+N_MIXTURES])

  UBM           = []
  UBM.append(weights)
  UBM.append(means)
  UBM.append(variances)

  return UBM



def paramlistread(list_in):
  """ 
   Receives a text file with a list of feature files as input. 
   Returns the feature arrays in the form of a list (each element corresponding to an entry of the input file)  
  """

  list_file = open(list_in, 'r')
  linha = list_file.readline().strip() #reads first line (feature file name)
  list_parameters = []
  while len(linha) > 0 :
    parameters = paramread(linha) #reads current line's features
    list_parameters.append(parameters) #ads features to output list
    linha = list_file.readline().strip() #reads next line (feature file name)
  list_file.close()

  return list_parameters



def paramread(arquivo):
  """
  Reads a feature file.
  Returns an array with the features.
  """

  numberOfFloats = os.path.getsize(arquivo)/4  # each feature is a float
  file = open(arquivo,mode = 'rb')  # opens feature input file
  parameters = array.array('f')
  parameters.fromfile(file,numberOfFloats)
  parameters = numpy.array(parameters, dtype=numpy.float64)
  file.close()

  return parameters




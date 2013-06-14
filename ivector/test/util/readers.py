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



def paramlistread(list_in, feature_dimension):
  """ 
   Receives a text file with a list of feature files as input. 
   Returns the feature arrays in the form of a list (each element corresponding to an entry of the input file)  
  """

  #Counting the number of frames
  #list_file = open(list_in, 'r')
  #linha = list_file.readline().strip() #reads first line (feature file name)
  #rows = 0
  #while len(linha) > 0 :
    #parameters = paramread(linha, feature_dimension) #reads current line's features
    #rows = rows + parameters.shape[0]
    #linha = list_file.readline().strip() #reads next line (feature file name)
  #list_file.close()


  #Reading
  list_file = open(list_in, 'r')
  linha = list_file.readline().strip() #reads first line (feature file name)
  list_parameters = []
  i = 0
  while len(linha) > 0 :
    parameters = paramread(linha, feature_dimension) #reads current line's features
    list_parameters.append(parameters)
    linha = list_file.readline().strip() #reads next line (feature file name)
  list_file.close()

  return list_parameters



def paramread(arquivo, feature_dimension):
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

  parameters = parameters.reshape(len(parameters)/feature_dimension, feature_dimension)

  return parameters



def Tmatrix_write_bob(T_Matrix,Tmatrix_file):
  """ Writes total variability matrix to file.

    T_Matrix: structure containing the total variability matrix
    Tmatrix_file: total variability matrix output file name
    
  """
  import struct
  import bob

  bob.db.utils.makedirs_safe(os.path.dirname(Tmatrix_file))
  out_file = open(Tmatrix_file,"wb")
  for j in range(T_Matrix.dim_cd):
    s = struct.pack('d'*T_Matrix.dim_rt, *T_Matrix.t[j])
    out_file.write(s)
  out_file.close()

  return


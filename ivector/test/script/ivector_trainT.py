#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Thu Jun 12 14:30:00 CEST 2012

"""
"""

import os, sys
import argparse
import bob
import numpy

from ivector.test.util import readers

def main():

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-o', '--output-dir', metavar='DIR', type=str, dest='output_dir', default='.', help='Output directory (defaults to "%(default)s")')

  parser.add_argument('-b', '--ubm_file', metavar='DIR', type=str, dest='ubm_file', default='./data/M256_MOBIO_ICB_2013_full.gmm', help='UBM directory (defaults to "%(default)s")')

  parser.add_argument('-l', '--training_files', metavar='DIR', type=str, dest='training_files', default='./data/MOBIO_ICB_2013_train_full_M256_Tmatrix.list', help='UBM directory (defaults to "%(default)s")')

  parser.add_argument('-m', '--n_mixtures', metavar='HIP', type=int, dest='n_mixtures', default=256, help='Number of gaussians (defaults to "%(default)s")')

  parser.add_argument('-f', '--feature_dim', metavar='HIP', type=int, dest='feature_dim', default=40, help='Features dimension (defaults to "%(default)s")')

  parser.add_argument('-t', '--t_dim', metavar='HIP', type=int, dest='t_dim', default=50, help='Total variability space dimension (defaults to "%(default)s")')

  parser.add_argument('-i', '--iterations', metavar='HIP', type=int, dest='iterations', default=10, help='Number of iterations to train the total variability matrix (defaults to "%(default)s")')

  parser.add_argument('-u', '--update_sigma', metavar='HIP', type=bool, dest='update_sigma', default=True, help='Update sigma param to train the total variability matrix (T matrix) (defaults to "%(default)s")')

  parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=True, help="Increase some verbosity")

  args = parser.parse_args()

  ########################
  # Loading Hiperparameters
  #########################
  OUTPUT_DIR          = args.output_dir # file containing UBM
  UBM_FILE            = args.ubm_file   # file containing UBM
  TRAINING_FILES      = args.training_files

  N_MIXTURES          = args.n_mixtures  # Number of gaussian mixtures
  DIMENSION           = args.feature_dim # Feature vector dimension
  T_MATRIX_DIM        = args.t_dim       # Total variability space dimension 
  T_MATRIX_ITERA      = args.iterations  # Number of iterations to train the total variability matrix
  UPDATE_SIGMA = True  # Update sigma param to train the total variability matrix (T matrix) - True or False

  VERBOSE = args.verbose

  #T_Matrix_list       = "/home/muliani0712a/test_ivector/MOBIO_ICB_2013_train_full_M256_Tmatrix.list"
  #T_Matrix_out_dir    = "/home/muliani0712a/test_ivector/Matriz_T/MOBIO_ICB_2013_16kHz_20MFCC_D_CMS_VAD_FW_bob/M256_MOBIO_ICB_2013_full/com_UpdateSigma" # dir with the total variability matrix


  ##########################
  # Reading the UBM
  ##########################
  if(VERBOSE):
    print("Reading the UBM ....")

  GMM_read = readers.gmmread(UBM_FILE, N_MIXTURES, DIMENSION)

  UBM           = bob.machine.GMMMachine(N_MIXTURES,DIMENSION)  # creates an object to store the UBM
  UBM.weights   = GMM_read[0]
  UBM.means     = GMM_read[1]
  UBM.variances = GMM_read[2]


  ##########################
  # Reads list of feature files used to train the total variability matrix
  ##########################
  if(VERBOSE):
    print("Reading the T-matrix files ....")

  T_Matrix_files = readers.paramlistread(TRAINING_FILES)

  return 0

if __name__ == "__main__":
  main()

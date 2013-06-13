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
  FEATURE_DIMENSION   = args.feature_dim # Feature vector dimension
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

  GMM_read = readers.gmmread(UBM_FILE, N_MIXTURES, FEATURE_DIMENSION)

  UBM           = bob.machine.GMMMachine(N_MIXTURES, FEATURE_DIMENSION)  # creates an object to store the UBM
  UBM.weights   = GMM_read[0]
  UBM.means     = GMM_read[1]
  UBM.variances = GMM_read[2]


  ##########################
  # Reads list of feature files used to train the total variability matrix
  ##########################
  if(VERBOSE):
    print("Reading the T-matrix files ....")
  T_Matrix_files = readers.paramlistread(TRAINING_FILES, FEATURE_DIMENSION)


  ##########################
  # Computes Baum-Welch statistics for all T-matrix training files
  ##########################
  if(VERBOSE):
    print("Compute statistics for T Matrix training files ....")

  stats = []
  for i in range(len(T_Matrix_files)):
    quadros = T_Matrix_files[i]
    s = bob.machine.GMMStats(N_MIXTURES,FEATURE_DIMENSION)
    for j in range(len(quadros)):
      UBM.acc_statistics(quadros[j],s)
    stats.append(s)

  return 0



  ##########################
  #Training the total variability matrix
  ##########################
  # Training steps...
  if(VERBOSE):
    print("Training T-MAtrix ...")


  for i in range(T_MATRIX_ITERA):
    if(VERBOSE):
      print "  Executing iteration %d ..." % (i+1)

    T_Matrix = bob.machine.IVectorMachine(UBM,T_MATRIX_DIM)
    trainer = bob.trainer.IVectorTrainer(UPDATE_SIGMA, 0.0, i+1, 0)  # ( update_sigma, convergence_threshold, max_iterations, compute_likelihood)
    trainer.train(T_Matrix,stats)
  # saves the total variability matrix
    output_file_Tmatrix = '{0}/matriz_T_M{1}_L{2}_T{3}_it{4}.T'.format(T_Matrix_out_dir,N_MIXTURES,DIMENSION,T_MATRIX_DIM,i+1)

    output_file_Tmatrix = os.path.join(OUTPUT_DIR,output_file_Tmatrix)
    Tmatrix_write_bob(T_Matrix,output_file_Tmatrix)


if __name__ == "__main__":
  main()

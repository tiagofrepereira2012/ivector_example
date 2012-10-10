#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Tue Jul 17 11:50:19 CET 2012

"""This script makes an LDA classification of data into two categories: real accesses and spoofing attacks for each LBP-TOP plane and it combinations. There is an option for normalizing and dimensionality reduction of the data prior to the LDA classification.
After the LDA, each data sample gets a score. Firstly, the EER threshold on the development set is calculated. The, according to this EER, the FAR, FRR and  HTER for the test and development set are calculated. The script outputs a text file with the performance results.
The details about the procedure are described in the paper: "LBP-TOP based countermeasure against facial spoofing attacks" - de Freitas Pereira, Tiago and Anjos, Andre and De Martino, Jose Mario and Marcel, Sebastien; ACCV - LBP 2012
"""

import os, sys
import argparse
import bob
import numpy

from .. import spoof
from ..spoof import calclbptop

from antispoofing.utils.ml import *
from antispoofing.utils.db import *

from antispoofing.lbptop.helpers import *


def main():

  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'lbp_features')
  OUTPUT_DIR = os.path.join(basedir, 'res')


  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-i', '--input-dir', metavar='DIR', type=str, dest='inputDir', default=INPUT_DIR, help='Base directory containing the scores to be loaded')
  parser.add_argument('-o', '--output-dir', metavar='DIR', type=str, dest='outputDir', default=OUTPUT_DIR, help='Base directory that will be used to save the results.')
  parser.add_argument('-n', '--normalize', action='store_true', dest='normalize', default=False, help='If True, will do zero mean unit variance normalization on the data before creating the LDA machine')

  parser.add_argument('-r', '--pca_reduction', action='store_true', dest='pca_reduction', default=False, help='If set, PCA dimensionality reduction will be performed to the data before doing LDA')

  parser.add_argument('-e', '--energy', type=str, dest="energy", default='0.99', help='The energy which needs to be preserved after the dimensionality reduction if PCA is performed prior to LDA')

  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Increases this script verbosity')

  # For SGE grid processing @ Idiap
  parser.add_argument('--grid', dest='grid', action='store_true', default=False, help=argparse.SUPPRESS)

  #######
  # Database especific configuration
  #######
  Database.create_parser(parser, implements_any_of='video')

  args = parser.parse_args()

  ###########
  # Loading the parsing variable
  ###########

  inputDir  = args.inputDir
  outputDir = args.outputDir
  grid      = args.grid

  if not os.path.exists(inputDir):
    parser.error("input directory does not exist")

  if not os.path.exists(outputDir): # if the output directory doesn't exist, create it
    bob.db.utils.makedirs_safe(outputDir)

  energy        = float(args.energy)
  normalize     = args.normalize
  pca_reduction = args.pca_reduction
  verbose       = args.verbose

  if(verbose):
    print "Loading input files..."

  ##########################
  # Loading the input files
  ##########################
  database = args.cls(args)
  trainReal, trainAttack = database.get_train_data()

  # create the full datasets from the file data
  train_real = calclbptop.create_full_dataset(trainReal,inputDir); train_attack = calclbptop.create_full_dataset(trainAttack,inputDir); 
 
  models = ['XY-plane','XT-Plane','YT-Plane','XT-YT-Plane','XY-XT-YT-plane']
  lines  = ['r','b','y','g^','c']

  ranges = range(len(models))
  if grid:
    pos = int(os.environ['SGE_TASK_ID']) - 1
    ranges = range(pos,pos+1)

  for i in ranges:

    if(verbose):
      print("Trainning the " + models[i])

    #### Loading the plane data
    train_real_plane   = train_real[i]
    train_attack_plane = train_attack[i]


    #### Training the counter measures
    if(verbose):
      print("Training the LDA machine ... ")

    [ldaMachine,pcaMachine] = ldaCountermeasure.train(train_real_plane, train_attack_plane, normalize=normalize, pca_reduction=pca_reduction,energy=energy)

    if(verbose):
      print("Saving the machines")

    #### Saving the machines
    if(pca_reduction):
      hdf5File_pca = bob.io.HDF5File(os.path.join(outputDir, 'pca_machine_'+ str(energy) + '-' + models[i] +'.txt'),openmode_string='w')
      pcaMachine.save(hdf5File_pca)
      del hdf5File_pca

    hdf5File_lda = bob.io.HDF5File(os.path.join(outputDir, 'lda_machine_'+ str(energy) + "-" + models[i] +'.txt'),openmode_string='w')
    ldaMachine.save(hdf5File_lda)
    del hdf5File_lda

 
if __name__ == '__main__':
  main()

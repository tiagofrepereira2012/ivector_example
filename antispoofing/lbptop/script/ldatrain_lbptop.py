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
import xbob.db.replay
import numpy

from .. import spoof
from ..spoof import calclbptop, scores

from antispoofing.utils.ml import *


def main():

  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'lbp_features')
  OUTPUT_DIR = os.path.join(basedir, 'res')

  protocols = xbob.db.replay.Database().protocols()
  protocols = [p.name for p in protocols]


  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-v', '--input-dir', metavar='DIR', type=str, dest='inputDir', default=INPUT_DIR, help='Base directory containing the scores to be loaded')
  parser.add_argument('-o', '--output-dir', metavar='DIR', type=str, dest='outputDir', default=OUTPUT_DIR, help='Base directory that will be used to save the results.')
  parser.add_argument('-n', '--normalize', action='store_true', dest='normalize', default=False, help='If True, will do zero mean unit variance normalization on the data before creating the LDA machine')
  parser.add_argument('-r', '--pca_reduction', action='store_true', dest='pca_reduction', default=False, help='If set, PCA dimensionality reduction will be performed to the data before doing LDA')
  parser.add_argument('-e', '--energy', type=str, dest="energy", default='0.99', help='The energy which needs to be preserved after the dimensionality reduction if PCA is performed prior to LDA')
  parser.add_argument('-p', '--protocol', type=str, dest="protocol", default='grandtest', help='The protocol type may be specified instead of the the id switch to subselect a smaller number of files to operate on', choices=protocols) 

  parser.add_argument('-pt', '--protocol-test', type=str, dest="protocol_test", default='grandtest', help='The REPLAY-ATTACK protocol type may be specified instead of the id switch to subselect a smaller number of files to operate on', choices=protocols)

  parser.add_argument('--sut', '--support-test', type=str, choices=('fixed', 'hand'), default='', dest='support_test', help='One of the valid supported attacks (fixed, hand) (defaults to "%(default)s")')

  parser.add_argument('--lit', '--light-test', type=str, choices=('controlled', 'adverse'), default='', dest='light_test', help='Types of illumination conditions (controlled,adverse) (defaults to "%(default)s")')

  # For SGE grid processing @ Idiap
  parser.add_argument('--grid', dest='grid', action='store_true', default=False, help=argparse.SUPPRESS)

  args = parser.parse_args()

  inputDir      = args.inputDir
  outputDir     = args.outputDir

  grid = args.grid

  if not os.path.exists(inputDir):
    parser.error("input directory does not exist")

  if not os.path.exists(outputDir): # if the output directory doesn't exist, create it
    bob.db.utils.makedirs_safe(outputDir)

  energy        = float(args.energy)
  normalize     = args.normalize
  pca_reduction = args.pca_reduction

  protocol      = args.protocol
  protocol_test = args.protocol_test
  support_test  = args.support_test
  light_test    = args.light_test


  print "Loading input files..."

  # loading the input files
  db = xbob.db.replay.Database()

  process_train_real = db.objects(protocol=protocol, groups='train', cls='real')
  process_train_attack = db.objects( protocol=protocol, groups='train', cls='attack')

  process_devel_real = db.objects(protocol=protocol, groups='devel', cls='real')
  process_devel_attack = db.objects(protocol=protocol, groups='devel', cls='attack')

  process_test_real = db.objects(protocol=protocol_test, groups='test', cls='real', support=support_test, light=light_test)
  process_test_attack = db.objects(protocol=protocol_test, groups='test', cls='attack',support=support_test,light=light_test)


  # create the full datasets from the file data
  train_real = calclbptop.create_full_dataset(process_train_real,inputDir); train_attack = calclbptop.create_full_dataset(process_train_attack,inputDir); 
  devel_real = calclbptop.create_full_dataset(process_devel_real,inputDir); devel_attack = calclbptop.create_full_dataset(process_devel_attack,inputDir); 
  test_real  =  calclbptop.create_full_dataset(process_test_real,inputDir); test_attack  = calclbptop.create_full_dataset(process_test_attack,inputDir); 
 
  models = ['XY-plane','XT-Plane','YT-Plane','XT-YT-Plane','XY-XT-YT-plane']
  lines  = ['r','b','y','g^','c']

  ranges = range(len(models))
  if grid:
    pos = int(os.environ['SGE_TASK_ID']) - 1
    ranges = range(pos,pos+1)

  for i in ranges:

    print("Trainning the " + models[i])

    #### Loading the plane data
    train_real_plane   = train_real[i]
    train_attack_plane = train_attack[i]

    devel_real_plane   = devel_real[i]
    devel_attack_plane = devel_attack[i]

    test_real_plane    = test_real[i]
    test_attack_plane  = test_attack[i]

    #### Training the counter measures
    print("Training the LDA machine ... ")

    [ldaMachine,pcaMachine] = ldaCountermeasure.train(train_real_plane, train_attack_plane, normalize=normalize, pca_reduction=pca_reduction,energy=energy)

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

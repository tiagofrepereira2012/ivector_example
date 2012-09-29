#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Fri Jul 13 14:30:00 CEST 2012

"""
This script will run feature vectors through a trained SVM and LDA and will produce score files for every individual video file in the database. Following the patter

for each plane

======+==============
score | isValidFrame


The procedure is described in the paper: "LBP-TOP based countermeasure against facial spoofing attacks" - de Freitas Pereira, Tiago and Anjos, Andre and De Martino, Jose Mario and Marcel, Sebastien; ACCV - LBP 2012

"""

import os, sys
import argparse
import bob
import xbob.db.replay
import numpy

from .. import spoof
from ..spoof import calclbptop, helpers
from antispoofing.utils.ml import *


def main():

  import math
  
  replayAttackProtocols = xbob.db.replay.Database().protocols()

  ##########
  # General configuration
  ##########

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-f', '--features-dir', type=str, dest='featuresDir', default='', help='Base directory containing the face features to a specific protocol  (defaults to "%(default)s")')

  parser.add_argument('-m', '--machine-file', type=str, dest='machineFile', default='', help='Path with the trained countermeasure')

  parser.add_argument('-a', '--machine-type', type=str, dest='machineType', default='', help='Type of the countermeasure machine', choices=('Linear','SVM'))

  parser.add_argument('-p', '--pca-machine', type=str, dest='pcaFile', default='', help='Path of the PCA machine')

  parser.add_argument('-n', '--normalization-file', type=str, dest='normalizationFile', default='', help='Base directory containing the normalization file')

  parser.add_argument('-o', '--output-dir', type=str, dest='outputDir', default='', help='Base directory that will be used to save the results.')

  parser.add_argument('-l', '--plane', type=str,dest='planeName',default='XY-XT-YT',choices=('XY','XT','YT','XT-YT','XY-XT-YT'),help="Plane name to calculate the scores")

  # For SGE grid processing @ Idiap
  parser.add_argument('--grid', dest='grid', action='store_true', default=False, help=argparse.SUPPRESS)

  #######
  # Database especific configuration
  #######
  subparsers = parser.add_subparsers(help="Database tests available")

  ##
  #Sub parsing replay attack database
  ##
  parser_replay = subparsers.add_parser('replay', help='Replay attack database')
  parser_replay.add_argument('--protocol', type=str, dest="replayProtocol", default='grandtest', help='The REPLAY-ATTACK protocol type may be specified instead of the id switch to subselect a smaller number of files to operate on', choices=replayAttackProtocols)

  parser_replay.add_argument('--support', type=str, choices=('fixed', 'hand'), default='', dest='replaySupport', help='One of the valid supported attacks (fixed, hand) (defaults to "%(default)s")')

  parser_replay.add_argument('--light', type=str, choices=('controlled', 'adverse'), default='', dest='replayLight', help='Types of illumination conditions (controlled,adverse) (defaults to "%(default)s")')

  parser_replay.add_argument('--group', type=str, choices=('train', 'devel', 'test'), default='', dest='replayGroup', help='One of the protocolar subgroups of data (train, devel, test) (defaults to "%(default)s")')

  
  parser_replay.set_defaults(which='replay') #Help to choose which subparser was selected

  args = parser.parse_args()

  ## Parsing

  #TODO: PLEASE CHECK THE CONSISTENCY OF THE NUMBER OF THE INPUTS
  database           = args.which
  featuresDir        = args.featuresDir
  machineFile        = args.machineFile
  pcaFile            = args.pcaFile
  normalizationFile  = args.normalizationFile
  machineType        = args.machineType
  outputDir          = args.outputDir
  planeName          = args.planeName

  #########
  # Loading some dataset
  #########
  #Loading Replay attack
  if(database == 'replay'):
    replayProtocol = args.replayProtocol
    replaySupport  = args.replaySupport
    replayLight    = args.replayLight
    replayGroup   = args.replayGroup

    db = xbob.db.replay.Database()
    process = db.objects(protocol=replayProtocol,groups=replayGroup,support=replaySupport,light=replayLight)

  
  # finally, if we are on a grid environment, just find what I have to process.
  if args.grid:
    key = int(os.environ['SGE_TASK_ID']) - 1
    if key >= len(process):
      raise RuntimeError, "Grid request for job %d on a setup with %d jobs" % \
          (key, len(process))
    process = [process[key]]

  
  # processing each video
  for index, obj in enumerate(process):

    #filename = str(obj.videofile(featuresDir)) 
    if(database=="replay"):
      filename = str(obj.videofile(featuresDir))

    
    sys.stdout.write("Processing file %s [%d/%d] \n" % (filename,
      index+1, len(process)))
    sys.stdout.flush()

    #Getting the datata
    dataset = calclbptop.create_full_dataset([obj],featuresDir,retrieveNanLines=True)
    featureVector = dataset[helpers.getPlaneIndex(planeName)]

     #Loading the PCA machine
    if(pcaFile != ''):
      hdf5File_pca    = bob.io.HDF5File(pcaFile,openmode_string='r')
      pcaMachine      = bob.machine.LinearMachine(hdf5File_pca)
      featureVector   = pca.pcareduce(pcaMachine, featureVector);

    #Loaging the normalization file for the min,max normalization
    if(normalizationFile != ''):
      [lowbound,highbound,mins,maxs] = readNormalizationData(normalizationFile)
      featureVector                  = norm.norm_range(featureVector, mins, maxs, lowbound, highbound)

    #Loading the machine
    if(machineType=='Linear'):
      hdf5File_linear = bob.io.HDF5File(machineFile,openmode_string='r')
      machine         = bob.machine.LinearMachine(hdf5File_linear)
      scores          = lda.get_scores(machine, featureVector)
    elif(machineType=='SVM'):
      machine = bob.machine.SupportVector(machineFile)
      scores  = svmCountermeasure.svm_predict(machine, featureVector)

    scores = numpy.reshape(scores,(1,len(scores)))

    # saves the output
    obj.save(scores,directory=outputDir,extension='.hdf5')  

  return 0


if __name__ == "__main__":
  main()

#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Fri Jul 27 14:30:00 CEST 2012

"""
This script will run the result analisys for the LBPTOP countermeasure


The procedure is described in the paper: "LBP-TOP based countermeasure against facial spoofing attacks" - de Freitas Pereira, Tiago and Anjos, Andre and De Martino, Jose Mario and Marcel, Sebastien; ACCV - LBP 2012

"""

import os, sys
import argparse
import bob
import xbob.db.replay
import numpy

from .. import spoof
from ..spoof import calclbptop, helpers, perf_lbptop

from antispoofing.utils.ml import *
from antispoofing.utils.helpers import *


def main():

  import math
  
  replayAttackProtocols = xbob.db.replay.Database().protocols()

  ##########
  # General configuration
  ##########

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-s', '--scores-dir', type=str, dest='scoresDir', default='', help='Base directory containing the Scores to a specific protocol  (defaults to "%(default)s")')

  parser.add_argument('-o', '--output-dir', type=str, dest='outputDir', default='', help='Base directory that will be used to save the results.')

  parser.add_argument('-n','--score-normalization',type=str, dest='scoreNormalization',default='', choices=('znorm','minmax',''))

  ##
  #Parsing replay attack database
  ##
  parser.add_argument('--protocol', type=str, dest="protocol", default='grandtest', help='The REPLAY-ATTACK protocol type may be specified instead of the id switch to subselect a smaller number of files to operate on', choices=replayAttackProtocols)

  parser.add_argument('--support', type=str, choices=('fixed', 'hand'), default='', dest='support', help='One of the valid supported attacks (fixed, hand) (defaults to "%(default)s")')

  parser.add_argument('--light', type=str, choices=('controlled', 'adverse'), default='', dest='light', help='Types of illumination conditions (controlled,adverse) (defaults to "%(default)s")')

  parser.add_argument('--group', type=str, choices=('train', 'devel', 'test'), default='', dest='group', help='One of the protocolar subgroups of data (train, devel, test) (defaults to "%(default)s")')

  args = parser.parse_args()

  scoresFolder = ["scores_XY","scores_XT","scores_YT","scores_XT-YT","scores_XY-XT-YT"]
  models = ['XY-plane','XT-Plane','YT-Plane','XT-YT-Plane','XY-XT-YT-plane']
  lines  = ['r','b','y','g^','c']
  energy = 0.0
  scoresRange = (-5,5)

  ## Parsing
  scoresDir          = args.scoresDir
  outputDir          = args.outputDir
  scoreNormalization = args.scoreNormalization

  if not os.path.exists(scoresDir):
    parser.error("scores-dir directory does not exist")

  if not os.path.exists(outputDir): # if the output directory doesn't exist, create it
    bob.db.utils.makedirs_safe(outputDir)


  #########
  # Loading some dataset
  #########
  #Loading Replay attack
  protocol = args.protocol
  support  = args.support
  light    = args.light
  group    = args.group

  db = xbob.db.replay.Database()

  process_train_real = db.objects(protocol=protocol, groups='train', cls='real')
  process_train_attack = db.objects( protocol=protocol, groups='train', cls='attack')

  process_devel_real = db.objects(protocol=protocol, groups='devel', cls='real')
  process_devel_attack = db.objects(protocol=protocol, groups='devel', cls='attack')

  process_test_real = db.objects(protocol=protocol, groups='test', cls='real', support=support, light=light)
  process_test_attack = db.objects(protocol=protocol, groups='test', cls='attack',support=support,light=light)


  #### Storing the scores in order to plot their distribution
  trainRealScores   = []
  trainAttackScores = []

  develRealScores   = []
  develAttackScores = []

  testRealScores    = []
  testAttackScores  = []

  thresholds        = []
  develTexts        = []
  testTexts        = []

  print("Generating test results ....")


  for i in range(len(scoresFolder)):

    print(models[i])
    scoresPlaneDir = os.path.join(scoresDir,scoresFolder[i])

    #Getting the scores
    realScores   = ScoreReader(process_train_real,scoresPlaneDir)
    attackScores = ScoreReader(process_train_attack,scoresPlaneDir)
    train_real_scores = realScores.getScores()
    train_attack_scores = attackScores.getScores()

    realScores   = ScoreReader(process_devel_real,scoresPlaneDir)
    attackScores = ScoreReader(process_devel_attack,scoresPlaneDir)
    devel_real_scores = realScores.getScores()
    devel_attack_scores = attackScores.getScores()

    realScores   = ScoreReader(process_test_real,scoresPlaneDir)
    attackScores = ScoreReader(process_test_attack,scoresPlaneDir)
    test_real_scores = realScores.getScores()
    test_attack_scores = attackScores.getScores()
 

    #Applying the score normaliztion
    if(scoreNormalization=="minmax"):
      scoreNorm = ScoreNormalization(numpy.concatenate((train_real_scores,train_attack_scores)))

      train_real_scores   = scoreNorm.calculateMinMaxNorm(train_real_scores)
      train_attack_scores = scoreNorm.calculateMinMaxNorm(train_attack_scores)

      devel_real_scores   = scoreNorm.calculateMinMaxNorm(devel_real_scores)
      devel_attack_scores = scoreNorm.calculateMinMaxNorm(devel_attack_scores)

      test_real_scores   = scoreNorm.calculateMinMaxNorm(test_real_scores)
      test_attack_scores = scoreNorm.calculateMinMaxNorm(test_attack_scores)

      scoresRange = (-1,1)
  
    elif(scoreNormalization=="znorm"):

      train_real_scores   = scoreNorm.calculateZNorm(train_real_scores)
      train_attack_scores = scoreNorm.calculateZNorm(train_attack_scores)

      devel_real_scores   = scoreNorm.calculateZNorm(devel_real_scores)
      devel_attack_scores = scoreNorm.calculateZNorm(devel_attack_scores)

      test_real_scores   = scoreNorm.calculateZNorm(test_real_scores)
      test_attack_scores = scoreNorm.calculateZNorm(test_attack_scores)

 

    if numpy.mean(devel_real_scores) < numpy.mean(devel_attack_scores):
      train_real_scores = train_real_scores * -1; train_attack_scores = train_attack_scores * -1
      devel_real_scores = devel_real_scores * -1; devel_attack_scores = devel_attack_scores * -1
      test_real_scores = test_real_scores * -1; test_attack_scores = test_attack_scores * -1

    
   #### Calculation of the error rates for one plane
    (test_hter,devel_hter),(test_text,devel_text),thres = perf.perf_hter([test_real_scores,test_attack_scores], [devel_real_scores,devel_attack_scores], bob.measure.eer_threshold)

    #Storing the scores
    trainRealScores.append(train_real_scores)
    trainAttackScores.append(train_attack_scores)

    develRealScores.append(devel_real_scores)
    develAttackScores.append(devel_attack_scores)

    testRealScores.append(test_real_scores)
    testAttackScores.append(test_attack_scores)

    #Storing the protocol issues for each plane
    thresholds.append(thres)
    develTexts.append(devel_text)
    testTexts.append(test_text)


  perf_lbptop.saveCounterMeasureResults(trainRealScores,trainAttackScores,develRealScores,develAttackScores,testRealScores,testAttackScores,thresholds,models,lines,develTexts,testTexts,energy,outputDir,scoresRange=scoresRange)
 
  return 0


if __name__ == "__main__":
  main()

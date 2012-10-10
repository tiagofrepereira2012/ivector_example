#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Thu Jul 21 15:56:55 CET 2012

"""This script calculates the chi2 difference between a model histogram and the data histograms for each LBP-TOP plane and it combinations, assigning scores to the data according to this. Firstly, the EER threshold on the development set is calculated. The, according to this EER, the FAR, FRR and HTER for the test and development set are calculated. The script outputs a text file with the performance results.
The details about the procedure are described in the paper: "LBP-TOP based countermeasure against facial spoofing attacks" - de Freitas Pereira, Tiago and Anjos, Andre and De Martino, Jose Mario and Marcel, Sebastien; ACCV - LBP 2012
"""

import os, sys
import argparse
import numpy
import bob

from .. import spoof
from ..spoof import calclbptop
from ..spoof import chi2

from antispoofing.utils.db import *
from antispoofing.lbptop.helpers import *


def main():

  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'lbp_features')
  INPUT_MODEL_DIR = os.path.join(basedir, 'res')
  OUTPUT_DIR = os.path.join(basedir, 'res')
  

  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-i', '--input-dir', metavar='DIR', type=str, dest='inputdir', default=INPUT_DIR, help='Base directory containing the scores to be loaded')

  parser.add_argument('-m', '--input-modeldir', metavar='DIR', type=str, dest='inputmodeldir', default=INPUT_MODEL_DIR, help='Base directory containing the histogram models to be loaded')

  parser.add_argument('-o', '--output-dir', metavar='DIR', type=str, dest='outputdir', default=OUTPUT_DIR, help='Base directory that will be used to save the results.')

  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Increases this script verbosity')


  #######
  # Database especific configuration
  #######
  Database.create_parser(parser, implements_any_of='video')
  args = parser.parse_args()

  verbose       = args.verbose


  if not os.path.exists(args.inputdir) or not os.path.exists(args.inputmodeldir):
    parser.error("input directory does not exist")

  if not os.path.exists(args.outputdir): # if the output directory doesn't exist, create it
    bob.db.utils.makedirs_safe(args.outputdir)
    
  if(verbose):
    print "Output directory set to \"%s\"" % args.outputdir
    print "Loading input files..."


  ##########################
  # Loading the input files
  ##########################
  database = args.cls(args)
  develReal, develAttack = database.get_devel_data()
  testReal, testAttack = database.get_test_data()

  # create the full datasets from the file data
  devel_real  = calclbptop.create_full_dataset(develReal,args.inputdir) 
  devel_attack = calclbptop.create_full_dataset(develAttack,args.inputdir)

  test_real = calclbptop.create_full_dataset(testReal,args.inputdir)
  test_attack = calclbptop.create_full_dataset(testAttack,args.inputdir)

  models = ['model_hist_real_XY','model_hist_real_XT','model_hist_real_YT','model_hist_real_XT_YT','model_hist_real_XY_XT_YT']
  lines  = ['r','b','y','g^','c']

  # loading the histogram models
  histmodelsfile = bob.io.HDF5File(os.path.join(args.inputmodeldir, 'histmodelsfile.hdf5'),'r')
  tf = open(os.path.join(args.outputdir, 'CHI-2_perf_table.txt'), 'w')

  for i in range(len(models)):
    if(verbose):
      print "Loading the model " + models[i]
    
    model_hist_real_plane = histmodelsfile.read(models[i])
    model_hist_real_plane = model_hist_real_plane[0,:]

    #Getting the histograms for each plane an its combinations
    devel_real_plane   = devel_real[i]
    devel_attack_plane = devel_attack[i]

    test_real_plane   = test_real[i]
    test_attack_plane = test_attack[i]

    if(verbose):
      print "Calculating the Chi-2 differences..."
    # calculating the comparison scores with chi2 distribution for each protocol subset   
    sc_devel_realmodel_plane = chi2.cmphistbinschimod(model_hist_real_plane, (devel_real_plane, devel_attack_plane))

    sc_test_realmodel_plane  = chi2.cmphistbinschimod(model_hist_real_plane, (test_real_plane, test_attack_plane))

    if(verbose):
      print "Saving the results in a file"
    # It is expected that the positives always have larger scores. Therefore, it is necessary to "invert" the scores by multiplying them by -1 (the chi-  square test gives smaller scores to the data from the similar distribution)
    sc_devel_realmodel_plane = (sc_devel_realmodel_plane[0] * -1, sc_devel_realmodel_plane[1] * -1)
    sc_test_realmodel_plane  = (sc_test_realmodel_plane[0] * -1, sc_test_realmodel_plane[1] * -1)

    perftable_plane, eer_thres_plane, mhter_thres_plane = performance_table(sc_test_realmodel_plane, sc_devel_realmodel_plane, "CHI-2 comparison in " + models[i]+ ", RESULTS")

    tf.write(perftable_plane)

  tf.close()
  del histmodelsfile
 

def performance_table(test, devel, title):
  """
  .. deprecated:: 1.1.0
   
   Returns a string containing the performance table. 
   
   I'm not using this function in the package. Just in this file.
   """

  def pline(group, far, attack_count, frr, real_count):
    fmtstr = " %s: FAR %.2f%% (%d / %d) / FRR %.2f%% (%d / %d) / HTER %.2f%%"
    return fmtstr % (group,
        100 * far, int(round(far*attack_count)), attack_count, 
        100 * frr, int(round(frr*real_count)), real_count, 
        50 * (far + frr))

  def perf(devel_scores, test_scores, threshold_func):
  
    from bob.measure import farfrr

    devel_attack_scores = devel_scores[1][:,0]
    devel_real_scores = devel_scores[0][:,0]
    test_attack_scores = test_scores[1][:,0]
    test_real_scores = test_scores[0][:,0]

    devel_real = devel_real_scores.shape[0]
    devel_attack = devel_attack_scores.shape[0]
    test_real = test_real_scores.shape[0]
    test_attack = test_attack_scores.shape[0]

    thres = threshold_func(devel_attack_scores, devel_real_scores)
    devel_far, devel_frr = farfrr(devel_attack_scores, devel_real_scores, thres)
    test_far, test_frr = farfrr(test_attack_scores, test_real_scores, thres)

    retval = []
    retval.append(" threshold: %.4f" % thres)
    retval.append(pline("dev ", devel_far, devel_attack, devel_frr, devel_real))
    retval.append(pline("test", test_far, test_attack, test_frr, test_real))

    return retval, thres

  retval = []
  retval.append(title)
  retval.append("")
  retval.append("EER @ devel")
  eer_table, eer_thres = perf(devel, test, bob.measure.eer_threshold)
  retval.extend(eer_table)
  retval.append("")
  retval.append("Mininum HTER @ devel")
  mhter_table, mhter_thres = perf(devel, test, bob.measure.min_hter_threshold)
  retval.extend(mhter_table)
  retval.append("")

  return ''.join([k+'\n' for k in retval]), eer_thres, mhter_thres

if __name__ == '__main__':
  main()

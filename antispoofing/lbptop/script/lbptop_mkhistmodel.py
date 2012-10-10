#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Mon Jul 16 08:30:00 CEST 2012

"""This script makes a histogram models for the real accesses videos in REPLAY-ATTACK by averaging the LBP histograms of each real access video for each LBP-TOP plane and it combinations. The output is an hdf5 file with the computed model histograms. The procedure is described in the paper: "LBP-TOP based countermeasure against facial spoofing attacks" - de Freitas Pereira, Tiago and Anjos, Andre and De Martino, Jose Mario and Marcel, Sebastien; ACCV - LBP 2012
"""

import os, sys
import argparse
import bob
import numpy

from .. import spoof
from antispoofing.utils.db import *
from antispoofing.lbptop.helpers import *


def main():

  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'lbp_features')
  OUTPUT_DIR = os.path.join(basedir, 'res')
  
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-i', '--input-dir', metavar='DIR', type=str, dest='inputdir', default=INPUT_DIR, help='Base directory containing the histogram features of all the videos')

  parser.add_argument('-d', '--output-dir', metavar='DIR', type=str, dest='outputdir', default=OUTPUT_DIR, help='Base directory that will be used to save the results (models).')

  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Increases this script verbosity')
  
  #######
  # Database especific configuration
  #######
  Database.create_parser(parser, implements_any_of='video')
  args = parser.parse_args()

  verbose       = args.verbose

  if not os.path.exists(args.inputdir):
    parser.error("input directory does not exist")
  
  if not os.path.exists(args.outputdir): # if the output directory doesn't exist, create it
    bob.db.utils.makedirs_safe(args.outputdir)
    

  ###################
  # Querying the database
  ###################
  if(verbose):
    print "Output directory set to \"%s\"" % args.outputdir
    print "Loading input files..."


  database = args.cls(args)
  trainReal,_ = database.get_train_data()

  # create the full datasets from the file data
  train_real = spoof.create_full_dataset(trainReal,args.inputdir);
  
  models = ['model_hist_real_XY','model_hist_real_XT','model_hist_real_YT','model_hist_real_XT_YT','model_hist_real_XY_XT_YT']
  histmodelsfile = bob.io.HDF5File(os.path.join(args.outputdir, 'histmodelsfile.hdf5'),'w')

  if(verbose):
    print "Creating the model for each frame and its combinations..."

  for i in range(len(models)):

    if(verbose):    
      print "Creating the model for " + models[i]

    train_real_plane =  train_real[i]

    model_hist_real_plane = numpy.sum(train_real_plane,axis=0,dtype='float64')
    model_hist_real_plane = numpy.divide(model_hist_real_plane,train_real_plane.shape[0])

    if(verbose):
      print "Saving the model histogram..."
    histmodelsfile.append(models[i], numpy.array(model_hist_real_plane))

  del histmodelsfile


 
if __name__ == '__main__':
  main()

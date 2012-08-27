#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Thu Jul 21 15:56:55 CET 2012

"""This script calculates the chi2 difference between a model histogram and the data histograms for each plane and it combinations in the LBP-TOP, assigning scores to the data according to this. Firstly, the EER threshold on the development set is calculated. The, according to this EER, the FAR, FRR and HTER for the test and development set are calculated. The script outputs a text file with the performance results.
The details about the procedure are described in the paper: "LBP-TOP based countermeasure against facial spoofing attacks" - de Freitas Pereira, Tiago and Anjos, Andre and De Martino, Jose Mario and Marcel, Sebastien; ACCV - LBP 2012
"""

import os, sys
import argparse
import bob
import numpy

def create_full_dataset(files):
  """Creates a full dataset matrix out of all the specified files"""
  dataset = None
  dataset_XY = None
  dataset_XT = None
  dataset_YT = None
  dataset_XT_YT = None
  dataset_XY_XT_YT = None

  dimXY = 0
  dimXT = 0
  dimYT = 0

  for key, filename in files.items():
    filename = os.path.expanduser(filename)
    fvs = bob.io.load(filename)


    if dataset_XY is None:
      #each individual plane
      dimXY = fvs[0][0][0]
      dimXT = fvs[0][0][1]
      dimYT = fvs[0][0][2]

      dataset_XY = numpy.array(fvs[1],copy=True,order='C',dtype='float')
      dataset_XT = numpy.array(fvs[2],copy=True,order='C',dtype='float')
      dataset_YT = numpy.array(fvs[3],copy=True,order='C',dtype='float')

      #Reshaping to the correct dimensions
      dataset_XY = dataset_XY[:,0:dimXY]
      dataset_XT = dataset_XT[:,0:dimXT]
      dataset_YT = dataset_YT[:,0:dimYT]

      #combining the temporal planes
      dataset_XT_YT = numpy.array(numpy.concatenate((dataset_XT,dataset_YT),axis=1),copy=True,order='C',dtype='float')

      #combining the all planes (space + time)
      dataset_XY_XT_YT = numpy.array(numpy.concatenate((dataset_XY,dataset_XT,dataset_YT),axis=1),copy=True,order='C',dtype='float')

    else:
      #appending each individual plane
      dataset_XY = numpy.concatenate((dataset_XY, fvs[1,:,0:dimXY]),axis=0)
      dataset_XT = numpy.concatenate((dataset_XT, fvs[2,:,0:dimXT]),axis=0)
      dataset_YT = numpy.concatenate((dataset_YT, fvs[3,:,0:dimYT]),axis=0)

      #appending temporal frames
      item_XT_YT    = numpy.concatenate((fvs[2,:,0:dimXT],fvs[3,:,0:dimYT]),axis=1)
      dataset_XT_YT = numpy.concatenate((dataset_XT_YT, item_XT_YT),axis=0)

      #appending all frames
      item_XY_XT_YT    = numpy.concatenate((fvs[1,:,0:dimXY],fvs[2,:,0:dimXT],fvs[3,:,0:dimYT]),axis=1)
      dataset_XY_XT_YT = numpy.concatenate((dataset_XY_XT_YT,item_XY_XT_YT),axis=0)
  
  dataset = [dataset_XY,dataset_XT,dataset_YT,dataset_XT_YT,dataset_XY_XT_YT]  
  return dataset


def main():

  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'lbp_features')
  INPUT_MODEL_DIR = os.path.join(basedir, 'res')
  OUTPUT_DIR = os.path.join(basedir, 'res')
  
  protocols = bob.db.replay.Database().protocols()

  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-v', '--input-dir', metavar='DIR', type=str, dest='inputdir', default=INPUT_DIR, help='Base directory containing the scores to be loaded')
  parser.add_argument('-m', '--input-modeldir', metavar='DIR', type=str, dest='inputmodeldir', default=INPUT_MODEL_DIR, help='Base directory containing the histogram models to be loaded')
  parser.add_argument('-d', '--output-dir', metavar='DIR', type=str, dest='outputdir', default=OUTPUT_DIR, help='Base directory that will be used to save the results.')
  parser.add_argument('-p', '--protocol', metavar='PROTOCOL', type=str, dest="protocol", default='grandtest', help='The protocol type may be specified instead of the the id switch to subselect a smaller number of files to operate on', choices=protocols) 

  from .. import ml
  from .. import spoof
  from ..ml import perf
  from ..spoof import chi2

  args = parser.parse_args()
  if not os.path.exists(args.inputdir) or not os.path.exists(args.inputmodeldir):
    parser.error("input directory does not exist")

  if not os.path.exists(args.outputdir): # if the output directory doesn't exist, create it
    bob.db.utils.makedirs_safe(args.outputdir)
    
  print "Output directory set to \"%s\"" % args.outputdir
  print "Loading input files..."

  # loading the input files (all the feature vectors of all the files in different subdatasets)
  db = bob.db.replay.Database()

  process_devel_real = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='devel', cls='real')
  process_devel_attack = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='devel', cls='attack')
  process_test_real = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='test', cls='real')
  process_test_attack = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='test', cls='attack')

  # create the full datasets from the file data
  devel_real  = create_full_dataset(process_devel_real) 
  devel_attack = create_full_dataset(process_devel_attack); 

  test_real = create_full_dataset(process_test_real)
  test_attack = create_full_dataset(process_test_attack)

  models = ['model_hist_real_XY','model_hist_real_XT','model_hist_real_YT','model_hist_real_XT_YT','model_hist_real_XY_XT_YT']
  lines  = ['r','b','y','g^','c']

  # loading the histogram models
  histmodelsfile = bob.io.HDF5File(os.path.join(args.inputmodeldir, 'histmodelsfile.hdf5'),'r')
  tf = open(os.path.join(args.outputdir, 'perf_table.txt'), 'w')

  for i in range(len(models)):
    print "Loading the model " + models[i]
    
    model_hist_real_plane = histmodelsfile.read(models[i])
    model_hist_real_plane = model_hist_real_plane[0,:]

    #Getting the histograms for each plane an its combinations
    devel_real_plane   = devel_real[i]
    devel_attack_plane = devel_attack[i]

    test_real_plane   = test_real[i]
    test_attack_plane = test_attack[i]


    print "Calculating the Chi-2 differences..."
    # calculating the comparison scores with chi2 distribution for each protocol subset   
    sc_devel_realmodel_plane = chi2.cmphistbinschimod(model_hist_real_plane, (devel_real_plane, devel_attack_plane))

    sc_test_realmodel_plane  = chi2.cmphistbinschimod(model_hist_real_plane, (test_real_plane, test_attack_plane))


    print "Saving the results in a file"
    # It is expected that the positives always have larger scores. Therefore, it is necessary to "invert" the scores by multiplying them by -1 (the chi-  square test gives smaller scores to the data from the similar distribution)
    sc_devel_realmodel_plane = (sc_devel_realmodel_plane[0] * -1, sc_devel_realmodel_plane[1] * -1)
    sc_test_realmodel_plane  = (sc_test_realmodel_plane[0] * -1, sc_test_realmodel_plane[1] * -1)

    perftable_plane, eer_thres_plane, mhter_thres_plane = perf.performance_table(sc_test_realmodel_plane, sc_devel_realmodel_plane, "CHI-2 comparison in " + models[i]+ ", RESULTS")

    tf.write(perftable_plane)
    
    #Plotting the ROC curves
    from .. import ml
    if(i==len(models)-1):
      hold=False
    else:
      hold=True


  tf.close()
  del histmodelsfile
 

if __name__ == '__main__':
  main()

#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Tue Jul 17 11:11:00 CEST 2012

"""This script can makes an SVM classification of data into two categories: real accesses and spoofing attacks. There is an option for normalizing between [-1, 1] and dimensionality reduction of the data prior to the SVM classification.
The probabilities obtained with the SVM are considered as scores for the data. Firstly, the EER threshold on the development set is calculated. The, according to this EER, the FAR, FRR and HTER for the test and development set are calculated. The script outputs a text file with the performance results.
The details about the procedure are described in the paper: "On the Effectiveness of Local Binary Patterns in Face Anti-spoofing" - Chingovska, Anjos & Marcel; BIOSIG 2012
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

def svm_predict(svm_machine, data):
  labels = [svm_machine.predict_class_and_scores(x)[1][0] for x in data]
  return labels


def main():

  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'lbp_features')
  OUTPUT_DIR = os.path.join(basedir, 'res')

  protocols = bob.db.replay.Database().protocols()

  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-v', '--input-dir', metavar='DIR', type=str, dest='inputdir', default=INPUT_DIR, help='Base directory containing the scores to be loaded')
  parser.add_argument('-d', '--output-dir', metavar='DIR', type=str, dest='outputdir', default=OUTPUT_DIR, help='Base directory that will be used to save the results.')
  parser.add_argument('-n', '--normalize', action='store_true', dest='normalize', default=False, help='If True, will do zero mean unit variance normalization on the data before creating the LDA machine')
  parser.add_argument('-r', '--pca_reduction', action='store_true', dest='pca_reduction', default=False, help='If set, PCA dimensionality reduction will be performed to the data before doing LDA')
  parser.add_argument('-e', '--energy', type=str, dest="energy", default='0.99', help='The energy which needs to be preserved after the dimensionality reduction if PCA is performed prior to LDA')
  parser.add_argument('-p', '--protocol', metavar='PROTOCOL', type=str, dest="protocol", default='grandtest', help='The protocol type may be specified instead of the the id switch to subselect a smaller number of files to operate on', choices=protocols) 

  parser.add_argument('-pt', '--protocol-test', metavar='PROTOCOL', type=str, dest="protocol_test", default='grandtest', help='The REPLAY-ATTACK protocol type may be specified instead of the id switch to subselect a smaller number of files to operate on', choices=protocols)

  parser.add_argument('--sut', '--support-test', metavar='SUPPORTTYPE', type=str, choices=('fixed', 'hand'), default='', dest='support_test', help='One of the valid supported attacks (fixed, hand) (defaults to "%(default)s")')

  parser.add_argument('--lit', '--light-test', metavar='LIGHT', type=str, choices=('controlled', 'adverse'), default='', dest='light_test', help='Types of illumination conditions (controlled,adverse) (defaults to "%(default)s")')


  parser.add_argument('-b', '--batch-size', metavar='INT', type=int,
      dest='batch', default=200, help='The number of samples per training iteration. Good values are greater than 100. Defaults to %(default)s')

  parser.add_argument('--ep', '--epoch', metavar='INT', type=int,
      dest='epoch', default=1, help='This is the number of training steps that need to be executed before we attempt to measure the error on the development set. Defaults to %(default)s')

  parser.add_argument('--hn', '--hidden-neurons', metavar='INT', type=int,
      dest='nhidden', default=5, help='The number hidden neurons in the (single) hidden layer of the MLP. Defaults to %(default)s')

  parser.add_argument('-m', '--maximum-iterations', metavar='INT', type=int,
      dest='maxiter', default=0, help='The maximum number of iterations to train for. A value of zero trains until a valley is detected on the development set. Defaults to %(default)s')

  parser.add_argument('-i', '--no-improvements', metavar='INT', type=int,
      dest='noimprov', default=0, help='The maximum number of iterations to wait for in case no improvements happen in the development set average RMSE. If that number of iterations is reached, the training is stopped. Values in the order of 10-20%% of the maximum number of iterations should be a reasonable default. If set to zero, do not consider this stop criteria. Defaults to %(default)s')

  parser.add_argument('--nt', '--number-train', metavar='INT', type=int,
      dest='nTrain', default=1, help='How many times do you want to run the MLP?. Defaults to %(default)s')

  parser.add_argument('-V', '--verbose', action='store_true', dest='verbose',
      default=False, help='Increases this script verbosity')

  from .. import ml
  from ..ml import pca, norm, rprop

  args = parser.parse_args()
  if not os.path.exists(args.inputdir):
    parser.error("input directory does not exist")

  if not os.path.exists(args.outputdir): # if the output directory doesn't exist, create it
    bob.db.utils.makedirs_safe(args.outputdir)

  outputdir = args.outputdir

  energy = float(args.energy)

  print "Loading input files..."
  # loading the input files
  db = bob.db.replay.Database()

  process_train_real = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='train', cls='real')
  process_train_attack = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='train', cls='attack')
  process_devel_real = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='devel', cls='real')
  process_devel_attack = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='devel', cls='attack')

  process_test_real = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol_test, groups='test', cls='real', support=args.support_test, light=args.light_test)
  process_test_attack = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol_test, groups='test', cls='attack',support=args.support_test,light=args.light_test)

  # create the full datasets from the file data
  train_real = create_full_dataset(process_train_real); train_attack = create_full_dataset(process_train_attack); 
  devel_real = create_full_dataset(process_devel_real); devel_attack = create_full_dataset(process_devel_attack); 
  test_real = create_full_dataset(process_test_real); test_attack = create_full_dataset(process_test_attack); 

  models = ['XY-plane','XT-Plane','YT-Plane','XT-YT-Plane','XY-XT-YT-plane']
  lines  = ['r','b','y','g^','c']

  tbl = []
  
  for i in range(len(models)):
    #Store the HTER in the test set in all nets
    hterDevelNets = []
    hterTestNets  = []


    for j in range(args.nTrain):

      print("Trainning number "+ str(j+1) +" the " + models[i])

      #Loading the plane data
      train_real_plane   = train_real[i]
      train_attack_plane = train_attack[i]

      devel_real_plane   = devel_real[i]
      devel_attack_plane = devel_attack[i]

      test_real_plane    = test_real[i]
      test_attack_plane  = test_attack[i]


      if args.normalize:  # normalization in the range [-1, 1] (recommended by LIBSVM)
        train_data = numpy.concatenate((train_real_plane, train_attack_plane), axis=0) 
        mins, maxs = norm.calc_min_max(train_data)
        train_real_plane = norm.norm_range(train_real_plane, mins, maxs, -1, 1); train_attack_plane = norm.norm_range(train_attack_plane, mins, maxs, -1, 1)
        devel_real_plane = norm.norm_range(devel_real_plane, mins, maxs, -1, 1); devel_attack_plane = norm.norm_range(devel_attack_plane, mins, maxs, -1, 1)
        test_real_plane  = norm.norm_range(test_real_plane, mins, maxs, -1, 1); test_attack_plane = norm.norm_range(test_attack_plane, mins, maxs, -1, 1)
  

      if args.pca_reduction: # PCA dimensionality reduction of the data
        train = bob.io.Arrayset() # preparing the train data for PCA (putting them altogether into bob.io.Arrayset)
        train.extend(train_real_plane)
        train.extend(train_attack_plane)
        pca_machine = pca.make_pca(train, energy, False) # performing PCA
        train_real_plane = pca.pcareduce(pca_machine, train_real_plane); train_attack_plane = pca.pcareduce(pca_machine, train_attack_plane)
        devel_real_plane = pca.pcareduce(pca_machine, devel_real_plane); devel_attack_plane = pca.pcareduce(pca_machine, devel_attack_plane)
        test_real_plane  = pca.pcareduce(pca_machine, test_real_plane); test_attack_plane = pca.pcareduce(pca_machine, test_attack_plane)


      print "Training MLP machine..."
      mlp, evolution = ml.rprop.make_mlp((train_real_plane,
      train_attack_plane), (devel_real_plane, devel_attack_plane),
      args.batch, args.nhidden, args.epoch, args.maxiter, args.noimprov,
      args.verbose)        

      print "Saving MLP..."
      if(args.verbose):
        mlpfile = bob.io.HDF5File(os.path.join(outputdir, 'mlp'+models[i]+'.hdf5'),'w')
        mlp.save(mlpfile)
        del mlpfile

      if(args.verbose):
        print "Saving result evolution..."
        evofile = bob.io.HDF5File(os.path.join(outputdir, 'training-evolution'+models[i]+'.hdf5'),'w')
        evolution.save(evofile)
        del evofile


      print "Computing devel and test scores..."
      devel_res, test_res = evolution.report(mlp, (test_real_plane, test_attack_plane),
        os.path.join(outputdir, 'plots_MLP_'+models[i]+'.pdf'),
        os.path.join(outputdir, 'error_MLP_'+models[i]+'.txt'))

      #Recording the HTER
      hterDevel = 100*((devel_res[0]+devel_res[1])/2)      
      hterTest  = 100*((test_res[0]+test_res[1])/2)
      hterTestNets.append(hterTest)
      hterDevelNets.append(hterDevel)
  
    averageDevel = numpy.average(hterDevelNets)
    stdDevel     = numpy.std(hterDevelNets)
    averageTest  = numpy.average(hterTestNets)
    stdTest      = numpy.std(hterTestNets)

    tbl = []
    tbl.append(" Performance in the devel set ")
    tbl.append(str(hterDevelNets))
    tbl.append(" Average %.2f%% " % averageDevel)
    tbl.append(" Standard deviation  %.2f " % stdDevel)
    tbl.append(" Performance in the test set ")
    tbl.append(str(hterTestNets))
    tbl.append(" Average %.2f%% " % averageTest)
    tbl.append(" Standard deviation  %.2f " % stdTest)
    txt = ''.join([k+'\n' for k in tbl])

    # write the results to a file 
    tf = open(os.path.join(outputdir, 'MLP_performance_'+ models[i] +'.txt'), 'w')
    tf.write(txt)
    tf.close()

 
if __name__ == '__main__':
  main()

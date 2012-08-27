#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Tue Jul 17 11:11:00 CEST 2012

"""This script can makes an SVM classification of data into two categories: real accesses and spoofing attacks for each LBP-TOP plane and it combinations. There is an option for normalizing between [-1, 1] and dimensionality reduction of the data prior to the SVM classification.
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


  from .. import ml
  from ..ml import pca, norm

  args = parser.parse_args()
  if not os.path.exists(args.inputdir):
    parser.error("input directory does not exist")

  if not os.path.exists(args.outputdir): # if the output directory doesn't exist, create it
    bob.db.utils.makedirs_safe(args.outputdir)

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
    print("Trainning the " + models[i])

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


    print "Training SVM machine..."
    svm_trainer = bob.trainer.SVMTrainer()
    svm_trainer.probability = True
    svm_machine = svm_trainer.train([train_real_plane, train_attack_plane])
    svm_machine.save(os.path.join(args.outputdir, 'svm_machine_'+ models[i] +'.txt'))

    print "Computing devel and test scores..."
    devel_real_plane_out   = svm_predict(svm_machine, devel_real_plane);
    devel_attack_plane_out = svm_predict(svm_machine, devel_attack_plane);
    test_real_plane_out    = svm_predict(svm_machine, test_real_plane);
    test_attack_plane_out  = svm_predict(svm_machine, test_attack_plane);

    # it is expected that the scores of the real accesses are always higher then the scores of the attacks. Therefore, a check is first made, if the average of the scores of real accesses is smaller then the average of the scores of the attacks, all the scores are inverted by multiplying with -1.
    if numpy.mean(devel_real_plane_out) < numpy.mean(devel_attack_plane_out):
      devel_real_plane_out = devel_real_plane_out * -1; devel_attack_plane_out = devel_attack_plane_out * -1
      test_real_plane_out = test_real_plane_out * -1; test_attack_plane_out = test_attack_plane_out * -1
    
    # calculation of the error rates
    thres = bob.measure.eer_threshold(devel_attack_plane_out, devel_real_plane_out)
    dev_far, dev_frr = bob.measure.farfrr(devel_attack_plane_out, devel_real_plane_out, thres)
    test_far, test_frr = bob.measure.farfrr(test_attack_plane_out, test_real_plane_out, thres)



    tbl.append(" ")
    tbl.append(models[i])
    if args.pca_reduction:
      tbl.append("EER @devel - (energy kept after PCA = %.2f" % (energy))
    tbl.append(" threshold: %.4f" % thres)
    tbl.append(" dev:  FAR %.2f%% (%d / %d) | FRR %.2f%% (%d / %d) | HTER %.2f%% " % \
        (100*dev_far, int(round(dev_far*len(devel_attack_plane))), len(devel_attack_plane), 
         100*dev_frr, int(round(dev_frr*len(devel_real_plane))), len(devel_real_plane),
         50*(dev_far+dev_frr)))
    tbl.append(" test: FAR %.2f%% (%d / %d) | FRR %.2f%% (%d / %d) | HTER %.2f%% " % \
        (100*test_far, int(round(test_far*len(test_attack_plane))), len(test_attack_plane),
         100*test_frr, int(round(test_frr*len(test_real_plane))), len(test_real_plane),
         50*(test_far+test_frr)))
  
    #Plotting the ROC curves
    from .. import ml
    if(i==len(models)-1):
      hold=False
    else:
      hold=True
    
    testHTER = round(50*(test_far+test_frr),2)
    ml.perf_lbptop.det_lbptop(test_real_plane_out,test_attack_plane_out,models[i]+" HTER = " + str(testHTER) + "%",hold,linestyle=lines[i],filename=os.path.join(args.outputdir,"DET_SVM.png"))




  txt = ''.join([k+'\n' for k in tbl])
  # write the results to a file 
  tf = open(os.path.join(args.outputdir, 'SVM_perf_table.txt'), 'w')
  tf.write(txt)


 
if __name__ == '__main__':
  main()

#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Fri Jul 13 14:30:00 CEST 2012

"""Calculates the LBPTop planes (XY,XT,YT) of the normalized faces in the videos in the REPLAY-ATTACK database. This code extract the LBP-TOP features using the MULTIRESOLUTION approach, setting more than one value for Rt. The result is the LBP histogram over all orthogonal frames of the video (XY,XT,YT). Different types of LBP operators are supported. The histograms can be computed for a subset of the videos in the database (using the protocols in REPLAY-ATTACK). The output is a single .hdf5 file for each video. The procedure is described in the paper: "LBP-TOP based countermeasure against facial spoofing attacks" - de Freitas Pereira, Tiago and Anjos, Andre and De Martino, Jose Mario and Marcel, Sebastien; ACCV - LBP 2012

"""

import os, sys
import argparse
import bob
import numpy

from .. import spoof
import antispoofing.utils

from antispoofing.utils.faceloc import *
from antispoofing.lbptop.helpers import *
from antispoofing.utils.db import *

def main():

  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'database')
  OUTPUT_DIR = os.path.join(basedir, 'lbp_features')


  ##########
  # General configuration
  ##########

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-v', '--input-dir', metavar='DIR', type=str, dest='inputDir', default=INPUT_DIR, help='Base directory containing the videos to be treated by this procedure (defaults to "%(default)s")')

  parser.add_argument('-d', '--directory', dest="directory", default=OUTPUT_DIR, help="This path will be prepended to every file output by this procedure (defaults to '%(default)s')")

  parser.add_argument('-n', '--normface-size', dest="normfacesize", default=64, type=int, help="this is the size of the normalized face box if face normalization is used (defaults to '%(default)s')")

  parser.add_argument('--ff', '--facesize_filter', dest="facesize_filter", default=50, type=int, help="all the frames with faces smaller then this number, will be discarded (defaults to '%(default)s')")

  parser.add_argument('-lXY', '--lbptypeXY', metavar='LBPTYPE', type=str, choices=('regular', 'riu2', 'uniform'), default='uniform', dest='lbptypeXY', help='Choose the type of LBP to use in the XY plane (defaults to "%(default)s")')

  parser.add_argument('-lXT', '--lbptypeXT', metavar='LBPTYPE', type=str, choices=('regular', 'riu2', 'uniform'), default='uniform', dest='lbptypeXT', help='Choose the type of LBP to use in the XT plane (defaults to "%(default)s")')

  parser.add_argument('-lYT', '--lbptypeYT', metavar='LBPTYPE', type=str, choices=('regular', 'riu2', 'uniform'), default='uniform', dest='lbptypeYT', help='Choose the type of LBP to use in the YT plane (defaults to "%(default)s")')


  parser.add_argument('-nXY', '--neighborsXY', type=int, default=8, dest='nXY', help='Number of Neighbors in the XY plane (defaults to "%(default)s")')
  parser.add_argument('-nXT', '--neighborsXT', type=int, default=8, dest='nXT', help='Number of Neighbors in the XT plane (defaults to "%(default)s")')
  parser.add_argument('-nYT', '--neighborsYT', type=int, default=8, dest='nYT', help='Number of Neighbors in the YT plane (defaults to "%(default)s")')

  parser.add_argument('-rX', '--radiusX', type=int, default=1, dest='rX', help='Radius of the X axis (defaults to "%(default)s")')
  parser.add_argument('-rY', '--radiusY', type=int, default=1, dest='rY', help='Radius of the Y axis (defaults to "%(default)s")')
  parser.add_argument('-rT', '--radiusT', type=int, default=1, dest='rT', help='Set of radius of the T axis (defaults to "%(default)s")',choices=xrange(10), nargs='+')

  parser.add_argument('-eXY', '--elbptypeXY', metavar='ELBPTYPE', type=str, choices=('regular', 'transitional', 'direction_coded','modified'),default='regular', dest='elbptypeXY', help='Choose the type of extended LBP features to compute in the XY plane (defaults to "%(default)s")')

  parser.add_argument('-eXT', '--elbptypeXT', metavar='ELBPTYPE', type=str, choices=('regular', 'transitional', 'direction_coded','modified'),default='regular', dest='elbptypeXT', help='Choose the type of extended LBP features to compute in the XT plane (defaults to "%(default)s")')

  parser.add_argument('-eYT', '--elbptypeYT', metavar='ELBPTYPE', type=str, choices=('regular', 'transitional', 'direction_coded','modified'),default='regular', dest='elbptypeYT', help='Choose the type of extended LBP features to compute in the YT plane (defaults to "%(default)s")')

  parser.add_argument('-cXY', '--circularXY', action='store_true', default=False, dest='cXY', help='Is circular neighborhood in XY plane?  (defaults to "%(default)s")')
  parser.add_argument('-cXT', '--circularXT', action='store_true', default=False, dest='cXT', help='Is circular neighborhood in XT plane?  (defaults to "%(default)s")')
  parser.add_argument('-cYT', '--circularYT', action='store_true', default=False, dest='cYT', help='Is circular neighborhood in YT plane?  (defaults to "%(default)s")')

  # For SGE grid processing @ Idiap
  parser.add_argument('--grid', dest='grid', action='store_true', default=False, help=argparse.SUPPRESS)

  #######
  # Database especific configuration
  #######
  #Database.create_parser(parser)
  Database.create_parser(parser, implements_any_of='video')

  args = parser.parse_args()

  lbphistlength = {'regular':256, 'riu2':10, 'uniform':59} # hardcoding the number of bins for the LBP variants

  inputDir  = args.inputDir
  directory = args.directory

  normfacesize    = args.normfacesize
  facesize_filter = args.facesize_filter

  nXY = args.nXY
  nXT = args.nXT
  nYT = args.nYT

  rX = args.rX
  rY = args.rY
  rT = args.rT

  cXY = args.cXY
  cXT = args.cXT
  cYT = args.cYT

  lbptypeXY =args.lbptypeXY
  lbptypeXT =args.lbptypeXT
  lbptypeYT =args.lbptypeYT

  elbptypeXY = args.elbptypeXY
  elbptypeXT = args.elbptypeXT
  elbptypeYT = args.elbptypeYT

  maxRadius = max(rX,rY,max(rT)) #Getting the max radius to extract the volume for analysis

  ########################
  #Querying the database
  ########################
  #database = new_database(databaseName,args=args)
  database = args.cls(args)
  realObjects, attackObjects = database.get_all_data()
  process = realObjects + attackObjects 
  

  # finally, if we are on a grid environment, just find what I have to process.
  if args.grid:
    key = int(os.environ['SGE_TASK_ID']) - 1
    if key >= len(process):
      raise RuntimeError, "Grid request for job %d on a setup with %d jobs" % \
          (key, len(process))
    process = [process[key]]

  # processing each video
  for index, obj in enumerate(process):

    #Loading the file
    filename = str(obj.videofile(inputDir))

    #Loading the video
    input = bob.io.VideoReader(filename)

    #Loading the face locations
    flocfile = obj.facefile(inputDir)
    locations = preprocess_detections(flocfile,input.number_of_frames,facesize_filter=facesize_filter)

    sys.stdout.write("Processing file %s (%d frames) [%d/%d] " % (filename,
      input.number_of_frames, index+1, len(process)))
    sys.stdout.flush()

    # start the work here...
    vin = input.load() # load the video
    nFrames = vin.shape[0]
      
    #Converting all frames to grayscale
    grayFrames = numpy.zeros(shape=(nFrames,vin.shape[2],vin.shape[3]))
    for i in range(nFrames):
      grayFrames[i] = bob.ip.rgb_to_gray(vin[i,:,:,:])


    ### STARTING the video analysis
    #Analysing each sub-volume in the video
    histVolumeXY = None
    histVolumeXT = None
    histVolumeYT = None
    for i in range(maxRadius,nFrames-maxRadius):

      histLocalVolumeXY = None
      histLocalVolumeXT = None
      histLocalVolumeYT = None

      #For each different radius
      for r in rT:
        #The max local radius to select the volume
        maxLocalRadius = max(rX,rY,r)

        #Select the volume to analyse
        rangeValues = range(i-maxLocalRadius,i+1+maxLocalRadius)
        normalizedVolume = spoof.getNormFacesFromRange(grayFrames,rangeValues,locations,normfacesize)

        #Calculating the histograms           
        histXY,histXT,histYT = spoof.lbptophist(normalizedVolume,nXY,nXT,nYT,rX,rY,r,cXY,cXT,cYT,lbptypeXY,lbptypeXT,lbptypeYT,elbptypeXY,elbptypeXT,elbptypeYT)

	#Concatenating in columns
        if(histLocalVolumeXY == None):
	  histLocalVolumeXY = histXY
	  histLocalVolumeXT = histXT
          histLocalVolumeYT = histYT
        else:
          #Is no necessary concatenate more elements in space with diferent radius in type
          histLocalVolumeXT= numpy.concatenate((histLocalVolumeXT, histXT),axis=1)
          histLocalVolumeYT= numpy.concatenate((histLocalVolumeYT, histYT),axis=1)

     #Concatenating in rows
      if(histVolumeXY == None):
        histVolumeXY= histLocalVolumeXY
        histVolumeXT= histLocalVolumeXT
        histVolumeYT= histLocalVolumeYT
      else:
        if(histLocalVolumeXY!=None):
          histVolumeXY= numpy.concatenate((histVolumeXY, histLocalVolumeXY),axis=0)
          histVolumeXT= numpy.concatenate((histVolumeXT, histLocalVolumeXT),axis=0)
          histVolumeYT= numpy.concatenate((histVolumeYT, histLocalVolumeYT),axis=0)

   
    #In the LBP we lose the R_t first and R_t last frames. For that reason, 
    #we need to add nan first and last R_t frames
  
    maxrT = max(rT)    
    nanParametersXY = numpy.ones(shape=(maxrT,histVolumeXY.shape[1]))*numpy.NaN
    nanParametersXT = numpy.ones(shape=(maxrT,histVolumeXT.shape[1]))*numpy.NaN
    nanParametersYT = numpy.ones(shape=(maxrT,histVolumeYT.shape[1]))*numpy.NaN

    #Add in the first R_t frames
    histVolumeXY = numpy.concatenate((nanParametersXY,histVolumeXY),axis=0)
    histVolumeXT = numpy.concatenate((nanParametersXT,histVolumeXT),axis=0)
    histVolumeYT = numpy.concatenate((nanParametersYT,histVolumeYT),axis=0)

    #Add in the last R_t frames
    histVolumeXY = numpy.concatenate((histVolumeXY,nanParametersXY),axis=0)
    histVolumeXT = numpy.concatenate((histVolumeXT,nanParametersXT),axis=0)
    histVolumeYT = numpy.concatenate((histVolumeYT,nanParametersYT),axis=0)

    #Saving the results into a file
    maxDim = max(histVolumeXY.shape[1],histVolumeXT.shape[1],histVolumeYT.shape[1])
    histData = numpy.zeros(shape=(4,histVolumeXY.shape[0],maxDim),dtype='float64')    
  
    #TODO: PROPOSE A BETTER SOLUTION TO STORE THE DIMENSIONS.
    dims = numpy.zeros(shape=(histVolumeXY.shape[0],maxDim))
    dims[0][0] = histVolumeXY.shape[1]  
    dims[0][1] = histVolumeXT.shape[1]
    dims[0][2] = histVolumeYT.shape[1]
    histData[0] = dims
    histData[1,:,0:dims[0][0]] = histVolumeXY
    histData[2,:,0:dims[0][1]] = histVolumeXT
    histData[3,:,0:dims[0][2]] = histVolumeYT


    sys.stdout.write('\n')
    sys.stdout.flush()

    # saves the output
    obj.save(histData,directory=directory,extension='.hdf5')
  

  return 0

if __name__ == "__main__":
  main()

#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Mon Jul 23 15:16:00 CEST 2012

"""
Create a LBPTop Video from an input video. The output is the 3 videos, one for each plane (XY,XT,YT)
"""

import os, sys
import argparse
import bob
import numpy

def main():

  import math
  
  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'video.mov')
  OUTPUT_DIR = os.path.join(basedir, 'output.avi')

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-i', '--input-file', metavar='DIR', type=str, dest='inputFile', default=INPUT_DIR, help='Base directory containing the videos to be treated by this procedure (defaults to "%(default)s")')

  parser.add_argument('-o', '--output-file', metavar='DIR', type=str, dest='outputFile', default=OUTPUT_DIR, help='The outputfile (defaults to "%(default)s")')

  parser.add_argument('-lXY', '--lbptypeXY', metavar='LBPTYPE', type=str, choices=('regular', 'riu2', 'uniform'), default='uniform', dest='lbptypeXY', help='Choose the type of LBP to use (defaults to "%(default)s")')

  parser.add_argument('-lXT', '--lbptypeXT', metavar='LBPTYPE', type=str, choices=('regular', 'riu2', 'uniform'), default='uniform', dest='lbptypeXT', help='Choose the type of LBP to use (defaults to "%(default)s")')

  parser.add_argument('-lYT', '--lbptypeYT', metavar='LBPTYPE', type=str, choices=('regular', 'riu2', 'uniform'), default='uniform', dest='lbptypeYT', help='Choose the type of LBP to use (defaults to "%(default)s")')


  parser.add_argument('-nXY', '--neighborsXY', type=int, default=8, dest='nXY', help='Number of Neighbors in the XY plane (defaults to "%(default)s")')
  parser.add_argument('-nXT', '--neighborsXT', type=int, default=8, dest='nXT', help='Number of Neighbors in the XT plane (defaults to "%(default)s")')
  parser.add_argument('-nYT', '--neighborsYT', type=int, default=8, dest='nYT', help='Number of Neighbors in the YT plane (defaults to "%(default)s")')

  parser.add_argument('-rX', '--radiusX', type=int, default=1, dest='rX', help='Radius of the X axis (defaults to "%(default)s")')
  parser.add_argument('-rY', '--radiusY', type=int, default=1, dest='rY', help='Radius of the Y axis (defaults to "%(default)s")')
  parser.add_argument('-rT', '--radiusT', type=int, default=1, dest='rT', help='Radius of the T axis (defaults to "%(default)s")')

  parser.add_argument('-cXY', '--circularXY', action='store_true', default=False, dest='cXY', help='Is circular neighborhood in XY plane?  (defaults to "%(default)s")')
  parser.add_argument('-cXT', '--circularXT', action='store_true', default=False, dest='cXT', help='Is circular neighborhood in XT plane?  (defaults to "%(default)s")')
  parser.add_argument('-cYT', '--circularYT', action='store_true', default=False, dest='cYT', help='Is circular neighborhood in YT plane?  (defaults to "%(default)s")')

  parser.add_argument('--el', '--elbptype', metavar='ELBPTYPE', type=str, choices=('regular', 'transitional', 'direction_coded', 'modified'), default='regular', dest='elbptype', help='Choose the type of extended LBP features to compute (defaults to "%(default)s")')


  args = parser.parse_args()

  lbphistlength = {'regular':256, 'riu2':10, 'uniform':59} # hardcoding the number of bins for the LBP variants

  from .. import spoof

  inputFile = args.inputFile
  outputFile = args.outputFile

  input = bob.io.VideoReader(inputFile)
    
  # start the work here...
  vin = input.load() # load the video
    
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

  nFrames = vin.shape[0]

  #Converting all frames to grayscale
  grayFrames = numpy.zeros(shape=(nFrames,vin.shape[2],vin.shape[3]))
  for i in range(nFrames):
    grayFrames[i] = bob.ip.rgb_to_gray(vin[i,:,:,:])

  volXY,volXT,volYT = spoof.lbptophist(grayFrames,nXY,nXT,nYT,rX,rY,rT,cXY,cXT,cYT,lbptypeXY,lbptypeXT,lbptypeYT,histrogramOutput=False)
  volDif_XT_YT = volXT-volYT

  height    = volXY.shape[1]
  width     = volXY.shape[2]
  framerate = input.frame_rate

  #voutXY = bob.io.VideoWriter(outputFile + "_xy.avi", height, width, framerate)
  #voutXT = bob.io.VideoWriter(outputFile + "_xt.avi", height, width, framerate) 
  #voutYT = bob.io.VideoWriter(outputFile + "_yt.avi", height, width, framerate)
  #voutXT_YT = bob.io.VideoWriter(outputFile + "_dif-xt-yt.avi", height, width, framerate)    
  vou = bob.io.VideoWriter(outputFile + "_LBPTOP.avi",height,4*width,framerate)

  #Normalizing
  volXY = numpy.divide(volXY,volXY.max()/255.)
  volXT = numpy.divide(volXT,volXT.max()/255.)
  volYT = numpy.divide(volXT,volXT.max()/255.)
  volDif_XT_YT = numpy.divide(volDif_XT_YT,volDif_XT_YT.max()/255.)



  for i in range(volXY.shape[0]):
    imxy    = numpy.zeros(shape=(3,height,width),dtype='uint8',order='C')
    imxy[0] = volXY[i].astype('uint8')
    imxy[1] = volXY[i].astype('uint8')
    imxy[2] = volXY[i].astype('uint8')
    #voutXY.append(imxy)

    imxt    = numpy.zeros(shape=(3,height,width),dtype='uint8',order='C')
    imxt[0] = volXT[i].astype('uint8')
    imxt[1] = volXT[i].astype('uint8')
    imxt[2] = volXT[i].astype('uint8')
    #voutXT.append(imxt)

    imyt    = numpy.zeros(shape=(3,height,width),dtype='uint8',order='C')
    imyt[0] = volYT[i].astype('uint8')
    imyt[1] = volYT[i].astype('uint8')
    imyt[2] = volYT[i].astype('uint8')
    #voutYT.append(imyt)

    imxtyt    = numpy.zeros(shape=(3,height,width),dtype='uint8',order='C')
    imxtyt[0] = volDif_XT_YT[i].astype('uint8')
    imxtyt[1] = volDif_XT_YT[i].astype('uint8')
    imxtyt[2] = volDif_XT_YT[i].astype('uint8')
    #voutXT_YT.append(imxtyt)

    imall = numpy.concatenate((imxy,imxt,imyt,imxtyt),axis=2)
    vou.append(imall)


  #voutXY.close()
  #voutXT.close()
  #voutYT.close()
  #voutXT_YT.close()
  vou.close()


  return 0

if __name__ == "__main__":
  main()

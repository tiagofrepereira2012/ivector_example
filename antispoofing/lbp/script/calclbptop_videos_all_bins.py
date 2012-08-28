#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Mon Aug 25 15:16:00 CEST 2012

"""
Create a LBP-TOP Video description from an input video. The output video has the following configuration:
_______________________________________
| VIDEO - Score over the time - Correlation between the face center bounding box and score
|
| XY frames for each set of bins
|
| XT frames for each set of bins
|
| YT frames for each set of bins
-----------------------------------------------

"""

import os, sys
import argparse
import bob
import xbob.db.replay
import numpy
from .. import ml
from ..ml import pca, lda, norm

#plot
import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
from pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.mlab import normpdf
from matplotlib.cm import gray as GrayColorMap

from .. import spoof
from .. import faceloc


"""
" Get the scores from some machine
" 
" @param grayFrames sequence

"""
def getScores(grayFrames,nXY,nXT,nYT,rX,rY,rT,cXY,cXT,cYT,lbptypeXY,lbptypeXT,lbptypeYT,inputFaceLocation,pcaMachinePath,classificationMachinePath):

  scores = numpy.zeros(shape=(0))
  nFrames = len(grayFrames)
  maxRadius = max(rX,rY,rT)

  #Loading face locations
  locations = faceloc.read_face(inputFaceLocation)
  locations = faceloc.expand_detections(locations, nFrames)


  sz = 64
  facesize_filter = 50

  for i in range(maxRadius,nFrames-maxRadius):

    #Select the volume to analyse
    rangeValues = range(i-maxRadius,i+1+maxRadius)
    normalizedVolume = spoof.getNormFacesFromRange(grayFrames,rangeValues,locations,facesize_filter)

    if(normalizedVolume==None):
      print("No frames in the volume")
      continue

    #Getting histogram
    histXY,histXT,histYT = spoof.lbptophist(normalizedVolume,nXY,nXT,nYT,rX,rY,rT,cXY,cXT,cYT,lbptypeXY,lbptypeXT,lbptypeYT)

    hist_XY_XT_YT         = numpy.concatenate((histXY,histXT,histYT),axis=1)
    
    hdf5File_pca          = bob.io.HDF5File(pcaMachinePath,openmode_string='r')
    pcaMachine            = bob.machine.LinearMachine(hdf5File_pca)

    hdf5File_lda          = bob.io.HDF5File(classificationMachinePath,openmode_string='r')
    classificationMachine = bob.machine.LinearMachine(hdf5File_lda)

    #PCA
    pcaData = pca.pcareduce(pcaMachine, hist_XY_XT_YT)

    #MACHINE
    scores = numpy.append(scores,lda.get_scores(classificationMachine, pcaData))

  return scores

"""
"Get the center coordinates of a bounding box
"""
def getCenterCoordinatesFromBB(inputFaceLocation,nFrames):

  centerCoordinates= numpy.zeros(shape=(nFrames,2))

  #Loading face locations
  locations = faceloc.read_face(inputFaceLocation)
  locations = faceloc.expand_detections(locations, nFrames)

  for i in range(nFrames):
    centerCoordinates[i][0] = locations[i].x+(locations[i].width/2)
    centerCoordinates[i][1] = locations[i].y+(locations[i].height/2)  

  return centerCoordinates

"""
" Plot the reference threshold and the current score
"""
def plotPoints(points,index,threshold=None,width=0,height=0,ylim=(-5,5), plotTitle=""):

  dpi=100
  handle = Figure(figsize=(width/dpi,height/dpi),dpi=dpi)
  axis = handle.add_subplot(111)
  grid(True)

  axis.set_title(plotTitle)

  #Number of frames
  frames = numpy.array(range(len(points)))


  if(threshold!=None):
    #Creating the threshold reference
    thresoldValues = threshold*numpy.ones(len(points))

    #plotting the reference value
    axis.plot(frames,thresoldValues,'r--')
  else:
    #Creating the threshold reference
    thresoldValues = ylim[0]*numpy.ones(len(points))

    #plotting the reference value
    axis.plot(frames,thresoldValues,'w--')


  currentPoints = points[0:index]

  #plotting the current point timeline
  axis.plot(frames[0:index],currentPoints,'b')

  axis.set_ylim(ylim)

  plotImage = fig2bzarray(handle)
 
  newPlotImage = numpy.zeros(shape=(3,height,width))
  newPlotImage[:,0:plotImage.shape[1],0:plotImage.shape[2]] = plotImage[:,:,:]
  newPlotImage = numpy.array(newPlotImage,dtype='uint8',order='C')

  return newPlotImage


"""
" Plot bars
"""
def plotBars(value,barWidth=1,width=0,height=0,plotTitle="",ylim=(-5,5)):

  dpi=100
  handle = Figure(figsize=(width/dpi,height/dpi),dpi=dpi)
  axis = handle.add_subplot(111)
  grid(True)

  axis.set_title(plotTitle)

  #plotting the current point timeline
  axis.bar(left=10,height=value,width=barWidth,color='b')

  plotImage = fig2bzarray(handle)

  axis.set_ylim(ylim)
 
  newPlotImage = numpy.zeros(shape=(3,height,width))
  newPlotImage[:,0:plotImage.shape[1],0:plotImage.shape[2]] = plotImage[:,:,:]
  newPlotImage = numpy.array(newPlotImage,dtype='uint8',order='C')

  return newPlotImage



def fig2bzarray(fig):
  """
  @brief Convert a Matplotlib figure to a 3D blitz array with RGB channels and
  return it
  @param fig a matplotlib figure
  @return a blitz 3D array of RGB values
  """
  #from matplotlib.backends.backend_agg import FigureCanvasAgg

  # draw the renderer
  canvas = FigureCanvasAgg(fig)
  fig.canvas.draw()

  # Get the RGB buffer from the figure, re-shape it adequately
  w,h = fig.canvas.get_width_height()
  buf = numpy.fromstring(fig.canvas.tostring_rgb(),dtype=numpy.uint8)
  buf.shape = (h,w,3)
  buf = numpy.transpose(buf, (2,0,1))

  return buf



def main():

  import math
  
  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = basedir
  OUTPUT_DIR = basedir

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-i', '--inputDir', metavar='DIR', type=str, dest='inputDir', default=INPUT_DIR, help='Base directory containing the videos to be treated by this procedure (defaults to "%(default)s")')

  parser.add_argument('-o', '--outputDir', metavar='DIR', type=str, dest='outputDir', default=OUTPUT_DIR, help='The output Dir (defaults to "%(default)s")')

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

  parser.add_argument('-lm', '--lda-machine', metavar='DIR', type=str, dest='ldaMachinePath', default='', help='LDA Machine Path (defaults to "%(default)s")')
  parser.add_argument('-pm', '--pca-machine', metavar='DIR', type=str, dest='pcaMachinePath', default='', help='PCA Machine Path (defaults to "%(default)s")')

  parser.add_argument('-t', '--threshold', metavar='DIR', type=float, dest='threshold', default=0, help='Threshold for classification (defaults to "%(default)s")')

  parser.add_argument('-vb', '--visualize-bins', metavar='', type=str, choices=('riu2', 'uniform','black'), default='black', dest='visualizeBins', help='What bins do you want to visualize (defaults to "%(default)s")')

  # For SGE grid processing @ Idiap
  parser.add_argument('--grid', dest='grid', action='store_true', default=False, help=argparse.SUPPRESS)

  args = parser.parse_args()

  lbphistlength = {'regular':256, 'riu2':10, 'uniform':59} # hardcoding the number of bins for the LBP variants

  from .. import spoof

  inputDir         = args.inputDir
  outputDir        = args.outputDir
  #inputFaceLocation = args.inputFaceLocation

  
  db = xbob.db.replay.Database()
  process = db.files(directory=inputDir, extension='.mov')
  outputFiles = db.files(directory=outputDir, extension='.mj2')
#  outputFiles = db.files(directory=outputDir, extension='.ogg')

  # where to find the face bounding boxes
  faceloc_dir = os.path.join(args.inputDir, 'face-locations')

    
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

  pcaMachinePath = args.pcaMachinePath
  ldaMachinePath = args.ldaMachinePath

  threshold     = args.threshold
  visualizeBins = args.visualizeBins


  # finally, if we are on a grid environment, just find what I have to process.
  if args.grid:
    pos = int(os.environ['SGE_TASK_ID']) - 1
    ordered_keys = sorted(process.keys())
    if pos >= len(ordered_keys):
      raise RuntimeError, "Grid request for job %d on a setup with %d jobs" % \
          (pos, len(ordered_keys))
    key = ordered_keys[pos] # gets the right key
    process = {key: process[key]}


  # processing each video
  for index, key in enumerate(sorted(process.keys())):
    filename   = process[key]
    outputFile = outputFiles[key]

    #load video
    input = bob.io.VideoReader(filename)
    vin = input.load() # load the video

    #load face location file
    inputFaceLocation = os.path.expanduser(db.paths([key], faceloc_dir, '.face')[0])
      
    # start the work here...

    nFrames = vin.shape[0]

    #Converting all frames to grayscale
    grayFrames = numpy.zeros(shape=(nFrames,vin.shape[2],vin.shape[3]))
    for i in range(nFrames):
      grayFrames[i] = bob.ip.rgb_to_gray(vin[i,:,:,:])


    ##################
    #Get the center coordinates
    centerCoordinates = getCenterCoordinatesFromBB(inputFaceLocation,len(grayFrames))


    #######################
    #Get scores from the machine
    scores = getScores(grayFrames,nXY,nXT,nYT,rX,rY,rT,cXY,cXT,cYT,lbptypeXY,lbptypeXT,lbptypeYT,inputFaceLocation,pcaMachinePath,ldaMachinePath)


    volXY,volXT,volYT = spoof.lbptophist(grayFrames,nXY,nXT,nYT,rX,rY,rT,cXY,cXT,cYT,lbptypeXY,lbptypeXT,lbptypeYT,histrogramOutput=False)
    #volDif_XT_YT = volXT-volYT

    height    = volXY.shape[1]
    width     = volXY.shape[2]

    framerate = input.frame_rate

    #vou = bob.io.VideoWriter(outputFile + "_LBPTOP.avi",5*height,10*width,framerate)
    vou = bob.io.VideoWriter(outputFile,4*height,10*width,framerate)
    print(vou.codec_name)
    exit()

    #graphTest = bob.io.VideoWriter(outputFile + "_SCORES.avi",height,2*width,framerate)
    #binsCount = numpy.zeros(shape=(volXY.shape[0],10))

    for i in range(volXY.shape[0]):

      #Plotting the score
      plotScores = plotPoints(scores,i,threshold=threshold,width=2*width,height=height,plotTitle="SCORES")

      #Plotting the X-Coordinate
      plotXCoordinates = plotPoints(centerCoordinates[:,0],i,width=2*width,height=height,ylim=(min(centerCoordinates[:,0])-5,max(centerCoordinates[:,0])+5),plotTitle="FACE DETECTOR X - COORDINATE")
      plotYCoordinates = plotPoints(centerCoordinates[:,1],i,width=2*width,height=height,ylim=(min(centerCoordinates[:,1])-5,max(centerCoordinates[:,1])+5),plotTitle="FACE DETECTOR Y - COORDINATE")

      #Plotting the Y-Coordinate

      imall_xy = numpy.zeros(shape=(3,height,10*width),dtype='uint8',order='C')
      imall_xt = numpy.zeros(shape=(3,height,10*width),dtype='uint8',order='C')
      imall_yt = numpy.zeros(shape=(3,height,10*width),dtype='uint8',order='C')

      #Sum the amount of riu2
      #imall_binsCount = numpy.zeros(shape=(3,height,10*width),dtype='uint8',order='C')

      #For each pattern
      for j in range(10):
     
        patterns = numpy.array(j)

        if(visualizeBins == "uniform"):
          colorLUT = getColorLUT_U2(patterns)
        elif(visualizeBins == "riu2"):
          colorLUT = getColorLUT_RIU2(patterns)
        else:
          if((j%2)==0):
            color = [0,0,0]
          else:
            color = [255,0,0]

        colorLUT = getColorLUT_OneColor(patterns,color)

        #Draw each pattern
        imxy    = numpy.zeros(shape=(3,height,width),dtype='uint8',order='C')
        imxy[0,:,:] = colorLUT[volXY[i]][:,:,0]
        imxy[1,:,:] = colorLUT[volXY[i]][:,:,1]
        imxy[2,:,:] = colorLUT[volXY[i]][:,:,2]

        imxt    = numpy.zeros(shape=(3,height,width),dtype='uint8',order='C')
        imxt[0,:,:] = colorLUT[volXT[i]][:,:,0]
        imxt[1,:,:] = colorLUT[volXT[i]][:,:,1]
        imxt[2,:,:] = colorLUT[volXT[i]][:,:,2]

        imyt    = numpy.zeros(shape=(3,height,width),dtype='uint8',order='C')
        imyt[0,:,:] = colorLUT[volYT[i]][:,:,0]
        imyt[1,:,:] = colorLUT[volYT[i]][:,:,1]
        imyt[2,:,:] = colorLUT[volYT[i]][:,:,2]

        #Sum all patterns
        #sumPatterns = sum(imxy!=255) + sum(imxt!=255) + sum(imyt!=255)
        #binsCount[i,j] = sumPatterns
        #plottedBars[:,:,j*width:j*width + width] = plotBars(sumPatterns,barWidth=1,width=width,height=height,plotTitle="",ylim=(5000,40000))
        #imall_binsCount[:,:,j*width:j*width + width] = plotPoints(binsCount[:,j],i,width=width,height=height,ylim=(5000,40000))
        

        imall_xy[:,:,j*width:j*width + width] = imxy
        imall_xt[:,:,j*width:j*width + width] = imxt
        imall_yt[:,:,j*width:j*width + width] = imyt


      #Generating video
      imall_origin = numpy.zeros(shape=(3,height,10*width),dtype='uint8',order='C')

      imall_origin[:,:,0:width] = vin[i,:,0:height,0:width]

      #Ploting scores
      begin = width
      end = begin+plotScores.shape[2]
      imall_origin[:,:,begin:end] = plotScores

      #Ploting X-Coordinate
      begin = end
      end = begin+plotXCoordinates.shape[2]
      imall_origin[:,:,begin:end] = plotXCoordinates

      #Ploting Y-Coordinate
      begin = end
      end = begin+plotYCoordinates.shape[2]
      imall_origin[:,:,begin:end] = plotYCoordinates


      imall =  numpy.concatenate((imall_origin,imall_xy,imall_xt,imall_yt),axis=1)
      #imall =  numpy.concatenate((imall_origin,imall_xy,imall_xt,imall_yt,imall_binsCount),axis=1)

      vou.append(imall)

    #voutXY.close()
    #voutXT.close()
    #voutYT.close()
    #voutXT_YT.close()
    vou.close()
    exit()
    #graphTest.close()

  return 0


#Black LUT
def getColorLUT_OneColor(patterns,color):
  m_lut_OneColor = numpy.ones(shape=(59,3),dtype='uint8',order='C')
  m_lut_OneColor = m_lut_OneColor*255

  #J) LBP patterns with 8 bits to 1
  if(len(numpy.where(patterns==8)[0])==1):
    m_lut_OneColor[58] = numpy.array(color)

  #A) all non uniform patterns have a label of 0.
  if(len(numpy.where(patterns==9)[0])==1):
    m_lut_OneColor[0] = numpy.array(color)

  #B) LBP pattern with 0 bit to 1
  if(len(numpy.where(patterns==0)[0])==1):
    m_lut_OneColor[1] = numpy.array(color)


  #C) LBP patterns with 1 bit to 1
  #2-9
  if(len(numpy.where(patterns==1)[0])==1):
    m_lut_OneColor[2] = numpy.array(color)
    m_lut_OneColor[3] = numpy.array(color)
    m_lut_OneColor[4] = numpy.array(color)
    m_lut_OneColor[5] = numpy.array(color)
    m_lut_OneColor[6] = numpy.array(color)
    m_lut_OneColor[7] = numpy.array(color)
    m_lut_OneColor[8] = numpy.array(color)
    m_lut_OneColor[9] = numpy.array(color)


  #D) LBP patterns with 2 bits to 1
  #10-17
  if(len(numpy.where(patterns==2)[0])==1):
    m_lut_OneColor[10] = numpy.array(color)
    m_lut_OneColor[11] = numpy.array(color)
    m_lut_OneColor[12] = numpy.array(color)
    m_lut_OneColor[13] = numpy.array(color)
    m_lut_OneColor[14] = numpy.array(color)
    m_lut_OneColor[15] = numpy.array(color)
    m_lut_OneColor[16] = numpy.array(color)
    m_lut_OneColor[17] = numpy.array(color)

  #E) LBP patterns with 3 bits to 1
  #18-25
  if(len(numpy.where(patterns==3)[0])==1):
    m_lut_OneColor[18] = numpy.array(color)
    m_lut_OneColor[19] = numpy.array(color)
    m_lut_OneColor[20] = numpy.array(color)
    m_lut_OneColor[21] = numpy.array(color)
    m_lut_OneColor[22] = numpy.array(color)
    m_lut_OneColor[23] = numpy.array(color)
    m_lut_OneColor[24] = numpy.array(color)
    m_lut_OneColor[25] = numpy.array(color)

  #F) LBP patterns with 4 bits to 1
  #26-33
  if(len(numpy.where(patterns==4)[0])==1):
    m_lut_OneColor[26] = numpy.array(color)
    m_lut_OneColor[27] = numpy.array(color)
    m_lut_OneColor[28] = numpy.array(color)
    m_lut_OneColor[29] = numpy.array(color)
    m_lut_OneColor[30] = numpy.array(color)
    m_lut_OneColor[31] = numpy.array(color)
    m_lut_OneColor[32] = numpy.array(color)
    m_lut_OneColor[33] = numpy.array(color)

  #G) LBP patterns with 5 bits to 1
  #34-41
  if(len(numpy.where(patterns==5)[0])==1):
    m_lut_OneColor[34] = numpy.array(color)
    m_lut_OneColor[35] = numpy.array(color)
    m_lut_OneColor[36] = numpy.array(color)
    m_lut_OneColor[37] = numpy.array(color)
    m_lut_OneColor[38] = numpy.array(color)
    m_lut_OneColor[39] = numpy.array(color)
    m_lut_OneColor[40] = numpy.array(color)
    m_lut_OneColor[41] = numpy.array(color)

  #H) LBP patterns with 6 bits to 1
  #42-49
  if(len(numpy.where(patterns==6)[0])==1):
    m_lut_OneColor[42] = numpy.array(color)
    m_lut_OneColor[43] = numpy.array(color)
    m_lut_OneColor[44] = numpy.array(color)
    m_lut_OneColor[45] = numpy.array(color)
    m_lut_OneColor[46] = numpy.array(color)
    m_lut_OneColor[47] = numpy.array(color)
    m_lut_OneColor[48] = numpy.array(color)
    m_lut_OneColor[49] = numpy.array(color)

  #I) LBP patterns with 7 bits to 1
  #50-57
  if(len(numpy.where(patterns==7)[0])==1):
    m_lut_OneColor[50] = numpy.array(color)
    m_lut_OneColor[51] = numpy.array(color)
    m_lut_OneColor[52] = numpy.array(color)
    m_lut_OneColor[53] = numpy.array(color)
    m_lut_OneColor[54] = numpy.array(color)
    m_lut_OneColor[55] = numpy.array(color)
    m_lut_OneColor[56] = numpy.array(color)
    m_lut_OneColor[57] = numpy.array(color)



  return m_lut_OneColor

#LUT for Rotation Invarian uniform patterns
def getColorLUT_RIU2(patterns):

  #m_lut_RIU2 = numpy.zeros(shape=(59,3),dtype='uint8',order='C')
  m_lut_RIU2 = numpy.ones(shape=(59,3),dtype='uint8',order='C')
  m_lut_RIU2 = m_lut_RIU2*255

  #J) LBP patterns with 8 bits to 1
  if(len(numpy.where(patterns==8)[0])==1):
    m_lut_RIU2[58] = numpy.array([255,0,0])

  #A) all non uniform patterns have a label of 0.
  if(len(numpy.where(patterns==9)[0])==1):
    m_lut_RIU2[0] = numpy.array([128,128,128])

  #B) LBP pattern with 0 bit to 1
  if(len(numpy.where(patterns==0)[0])==1):
    m_lut_RIU2[1] = numpy.array([0,0,0])


  #C) LBP patterns with 1 bit to 1
  #2-9
  if(len(numpy.where(patterns==1)[0])==1):
    m_lut_RIU2[2] = numpy.array([175,107,14])
    m_lut_RIU2[3] = numpy.array([175,107,14])
    m_lut_RIU2[4] = numpy.array([175,107,14])
    m_lut_RIU2[5] = numpy.array([175,107,14])
    m_lut_RIU2[6] = numpy.array([175,107,14])
    m_lut_RIU2[7] = numpy.array([175,107,14])
    m_lut_RIU2[8] = numpy.array([175,107,14])
    m_lut_RIU2[9] = numpy.array([175,107,14])


  #D) LBP patterns with 2 bits to 1
  #10-17
  if(len(numpy.where(patterns==2)[0])==1):
    m_lut_RIU2[10] = numpy.array([167,149,11])
    m_lut_RIU2[11] = numpy.array([167,149,11])
    m_lut_RIU2[12] = numpy.array([167,149,11])
    m_lut_RIU2[13] = numpy.array([167,149,11])
    m_lut_RIU2[14] = numpy.array([167,149,11])
    m_lut_RIU2[15] = numpy.array([167,149,11])
    m_lut_RIU2[16] = numpy.array([167,149,11])
    m_lut_RIU2[17] = numpy.array([167,149,11])

  #E) LBP patterns with 3 bits to 1
  #18-25
  if(len(numpy.where(patterns==3)[0])==1):
    m_lut_RIU2[18] = numpy.array([231,6,149])
    m_lut_RIU2[19] = numpy.array([231,6,149])
    m_lut_RIU2[20] = numpy.array([231,6,149])
    m_lut_RIU2[21] = numpy.array([231,6,149])
    m_lut_RIU2[22] = numpy.array([231,6,149])
    m_lut_RIU2[23] = numpy.array([231,6,149])
    m_lut_RIU2[24] = numpy.array([231,6,149])
    m_lut_RIU2[25] = numpy.array([231,6,149])

  #F) LBP patterns with 4 bits to 1
  #26-33
  if(len(numpy.where(patterns==4)[0])==1):
    m_lut_RIU2[26] = numpy.array([134,28,2])
    m_lut_RIU2[27] = numpy.array([134,28,2])
    m_lut_RIU2[28] = numpy.array([134,28,2])
    m_lut_RIU2[29] = numpy.array([134,28,2])
    m_lut_RIU2[30] = numpy.array([134,28,2])
    m_lut_RIU2[31] = numpy.array([134,28,2])
    m_lut_RIU2[32] = numpy.array([134,28,2])
    m_lut_RIU2[33] = numpy.array([134,28,2])

  #G) LBP patterns with 5 bits to 1
  #34-41
  if(len(numpy.where(patterns==5)[0])==1):
    m_lut_RIU2[34] = numpy.array([20,16,97])
    m_lut_RIU2[35] = numpy.array([20,16,97])
    m_lut_RIU2[36] = numpy.array([20,16,97])
    m_lut_RIU2[37] = numpy.array([20,16,97])
    m_lut_RIU2[38] = numpy.array([20,16,97])
    m_lut_RIU2[39] = numpy.array([20,16,97])
    m_lut_RIU2[40] = numpy.array([20,16,97])
    m_lut_RIU2[41] = numpy.array([20,16,97])

  #H) LBP patterns with 6 bits to 1
  #42-49
  if(len(numpy.where(patterns==6)[0])==1):
    m_lut_RIU2[42] = numpy.array([2,83,48])
    m_lut_RIU2[43] = numpy.array([2,83,48])
    m_lut_RIU2[44] = numpy.array([2,83,48])
    m_lut_RIU2[45] = numpy.array([2,83,48])
    m_lut_RIU2[46] = numpy.array([2,83,48])
    m_lut_RIU2[47] = numpy.array([2,83,48])
    m_lut_RIU2[48] = numpy.array([2,83,48])
    m_lut_RIU2[49] = numpy.array([2,83,48])

  #I) LBP patterns with 7 bits to 1
  #50-57
  if(len(numpy.where(patterns==7)[0])==1):
    m_lut_RIU2[50] = numpy.array([89,65,86])
    m_lut_RIU2[51] = numpy.array([89,65,86])
    m_lut_RIU2[52] = numpy.array([89,65,86])
    m_lut_RIU2[53] = numpy.array([89,65,86])
    m_lut_RIU2[54] = numpy.array([89,65,86])
    m_lut_RIU2[55] = numpy.array([89,65,86])
    m_lut_RIU2[56] = numpy.array([89,65,86])
    m_lut_RIU2[57] = numpy.array([89,65,86])

  return m_lut_RIU2 


def getColorLUT_U2(patterns):

  #m_lut_U2 = numpy.zeros(shape=(59,3),dtype='uint8',order='C')
  m_lut_U2 = numpy.ones(shape=(59,3),dtype='uint8',order='C')
  m_lut_U2 = m_lut_U2*255

  #J) LBP patterns with 8 bits to 1
  if(len(numpy.where(patterns==8)[0])==1):
    m_lut_U2[58] = numpy.array([255,0,0])

  #A) all non uniform patterns have a label of 0.
  if(len(numpy.where(patterns==9)[0])==1):
    m_lut_U2[0] = numpy.array([128,128,128])

  #B) LBP pattern with 0 bit to 1
  if(len(numpy.where(patterns==0)[0])==1):
    m_lut_U2[1] = numpy.array([0,0,0])


  #C) LBP patterns with 7 bit to 1
  if(len(numpy.where(patterns==7)[0])==1):
    m_lut_U2[50] = numpy.array([89,65,86])
    m_lut_U2[51] = numpy.array([116,40,115])
    m_lut_U2[52] = numpy.array([120,73,107])
    m_lut_U2[53] = numpy.array([149,109,141])
    m_lut_U2[54] = numpy.array([178,148,176])
    m_lut_U2[55] = numpy.array([206,181,202])
    m_lut_U2[56] = numpy.array([230,210,221])
    m_lut_U2[57] = numpy.array([238,225,234])

  #D) LBP patterns with 6 bits to 1
  if(len(numpy.where(patterns==6)[0])==1):
    m_lut_U2[42] = numpy.array([2,83,48])
    m_lut_U2[43] = numpy.array([15,102,72])
    m_lut_U2[44] = numpy.array([0,140,94])
    m_lut_U2[45] = numpy.array([6,172,133])
    m_lut_U2[46] = numpy.array([87,197,191])
    m_lut_U2[47] = numpy.array([136,206,185])
    m_lut_U2[48] = numpy.array([160,218,203])
    m_lut_U2[49] = numpy.array([184,216,201])

  #E) LBP patterns with 5 bits to 1
  if(len(numpy.where(patterns==5)[0])==1):
    m_lut_U2[34] = numpy.array([20,16,97])
    m_lut_U2[35] = numpy.array([29,32,118])
    m_lut_U2[36] = numpy.array([28,48,131])
    m_lut_U2[37] = numpy.array([37,57,131])
    m_lut_U2[38] = numpy.array([80,98,178])
    m_lut_U2[39] = numpy.array([116,136,189])
    m_lut_U2[40] = numpy.array([153,163,216])
    m_lut_U2[41] = numpy.array([163,165,244])

  #F) LBP patterns with 4 bits to 1
  if(len(numpy.where(patterns==4)[0])==1):
    m_lut_U2[26] = numpy.array([134,28,2])
    m_lut_U2[27] = numpy.array([192,50,27])
    m_lut_U2[28] = numpy.array([238,53,35])
    m_lut_U2[29] = numpy.array([238,81,50])
    m_lut_U2[30] = numpy.array([232,123,121])
    m_lut_U2[31] = numpy.array([250,152,154])
    m_lut_U2[32] = numpy.array([252,171,177])
    m_lut_U2[33] = numpy.array([243,196,174])

  #G) LBP patterns with 3 bits to 1
  if(len(numpy.where(patterns==3)[0])==1):
    m_lut_U2[18] = numpy.array([231,6,149])
    m_lut_U2[19] = numpy.array([245,21,156])
    m_lut_U2[20] = numpy.array([239,47,158])
    m_lut_U2[21] = numpy.array([242,83,153])
    m_lut_U2[22] = numpy.array([240,115,158])
    m_lut_U2[23] = numpy.array([228,134,150])
    m_lut_U2[24] = numpy.array([243,162,156])
    m_lut_U2[25] = numpy.array([241,181,150])

  #H) LBP patterns with 2 bits to 1
  if(len(numpy.where(patterns==2)[0])==1):
    m_lut_U2[10] = numpy.array([167,149,11])
    m_lut_U2[11] = numpy.array([175,176,42])
    m_lut_U2[12] = numpy.array([213,208,14])
    m_lut_U2[13] = numpy.array([234,230,38])
    m_lut_U2[14] = numpy.array([242,233,63])
    m_lut_U2[15] = numpy.array([250,237,84])
    m_lut_U2[16] = numpy.array([253,245,109])
    m_lut_U2[17] = numpy.array([248,242,190])

  #I) LBP patterns with 1 bits to 1
  if(len(numpy.where(patterns==1)[0])==1):
    m_lut_U2[2] = numpy.array([175,107,14])
    m_lut_U2[3] = numpy.array([196,132,58])
    m_lut_U2[4] = numpy.array([239,146,22])
    m_lut_U2[5] = numpy.array([245,152,44])
    m_lut_U2[6] = numpy.array([142,180,110])
    m_lut_U2[7] = numpy.array([246,204,151])
    m_lut_U2[8] = numpy.array([253,215,172])
    m_lut_U2[9] = numpy.array([240,233,186])

  return m_lut_U2 




if __name__ == "__main__":
  main()

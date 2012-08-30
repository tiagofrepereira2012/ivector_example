#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Thu Jul 19 12:49:15 CET 2012

"""Support methods to compute the Local Binary Pattern (LBP) histogram of an image
"""

import numpy
import math
import bob

""""
" Convert a sequence of RGB Frames to a sequence of Gray scale frames and do the face normalization over a given bounding box (bbx) in an image (around the detected face for example).

Keyword Parameters:
  rgbFrameSequence
    The sequence of frames in the RGB
  
  locations
    The face locations
  sz
   The size of the rescaled face bounding box

  bbxsize
    Considers as invalid all the bounding boxes with size smaller then this value

"""
def rgbVideo2grayVideo_facenorm(rgbFrameSequence,locations,sz,bbxsize_filter=0):
  grayFaceNormFrameSequence = numpy.empty(shape=(0,sz,sz))
  
  for k in range(0, rgbFrameSequence.shape[0]):
    bbx = locations[k]
    
    if bbx and bbx.is_valid() and bbx.height > bbxsize_filter:
      frame = bob.ip.rgb_to_gray(rgbFrameSequence[k,:,:,:])
      cutframe = frame[bbx.y:(bbx.y+bbx.height),bbx.x:(bbx.x+bbx.width)] # cutting the box region
      tempbbx = numpy.ndarray((sz, sz), 'float64')
      normbbx = numpy.ndarray((sz, sz), 'uint8')
      bob.ip.scale(cutframe, tempbbx) # normalization
      tempbbx_ = tempbbx + 0.5 #TODO: UNDERSTAND THIS
      tempbbx_ = numpy.floor(tempbbx_)
      normbbx = numpy.cast['uint8'](tempbbx_)
      
      #Preparing the data to append.
      #TODO: Maybe a there is a better solution
      grayFaceNormFrame = numpy.empty(shape=(1,sz,sz))
      grayFaceNormFrame[0] = normbbx
      grayFaceNormFrameSequence = numpy.append(grayFaceNormFrameSequence,grayFaceNormFrame,axis=0)

  return grayFaceNormFrameSequence


"""
" Select the reference bounding box for a volume. The valid bounding box is the center; if it is not exists, take the first valid, otherwise none is  returned

"""
def getReferenceBoundingBox(locations,rangeValues,bbxsize_filter):
  
  #First trying to get the center frame
  center = (max(rangeValues)+min(rangeValues))/2
  bbx = locations[center]

  if bbx and bbx.is_valid() and bbx.height > bbxsize_filter:
    return bbx

  bbx = None
  #if the center is invalid, try to get the first valid
  for i in rangeValues:
    if(i==center):
      continue

    if locations[i] and locations[i].is_valid() and locations[i].height > bbxsize_filter:
      bbx = locations[i]
      break

  return bbx


"""
" Get a volume with normalized faces
"""
def getNormFacesFromRange(grayFrameSequence,rangeValues,locations,sz,bbxsize_filter=0):

  #If there is no bounding boxes, this volume will no be analised  
  bbx = getReferenceBoundingBox(locations,rangeValues,bbxsize_filter)
  if(bbx==None):
    return None

  selectedFrames = grayFrameSequence[rangeValues]
  nFrames = selectedFrames.shape[0]
  selectedNormFrames = numpy.zeros(shape=(nFrames,sz,sz),dtype='uint8')

  for i in range(nFrames):
    frame = selectedFrames[i]
    cutframe = frame[bbx.y:(bbx.y+bbx.height),bbx.x:(bbx.x+bbx.width)] # cutting the box region
    tempbbx = numpy.ndarray((sz, sz), 'float64')
    #normbbx = numpy.ndarray((sz, sz), 'uint8')
    bob.ip.scale(cutframe, tempbbx) # normalization
    tempbbx_ = tempbbx + 0.5
    tempbbx_ = numpy.floor(tempbbx_)
    selectedNormFrames[i] = numpy.cast['uint8'](tempbbx_)
    
  return selectedNormFrames

"""
" Calculate the LBPTop histograms
"
" Keyword parameters
"
"  grayFaceNormFrameSequence
"     The sequence of FACE frames in grayscale
"   nXY
"     Number of points around the center pixel in the XY direction
"   nXT
"     Number of points around the center pixel in the XT direction
"   nYT
"     Number of points around the center pixel in the YT direction
"   rX
"     The radius of the circle on which the points are taken (for circular LBP) in the X axis
"   rY
"     The radius of the circle on which the points are taken (for circular LBP) in the Y axis
"   rT
"     The radius of the circle on which the points are taken (for circular LBP) in the T axis
"   cXY
"     True if circular LBP is needed in the XY plane, False otherwise
"   cXT
"     True if circular LBP is needed in the XT plane, False otherwise
"   cYT
"     True if circular LBP is needed in the YT plane, False otherwise
"   lbptype
"     The type of the LBP operator (regular, uniform or riu2)
"   histrogramOutput
"     If the output is really a histogram. Otherwise a LBPTop volume for each plane will be returned 
"""
def lbptophist(grayFaceNormFrameSequence,nXY,nXT,nYT,rX,rY,rT,cXY,cXT,cYT,lbptypeXY,lbptypeXT,lbptypeYT,histrogramOutput=True):
  
  uniformXY = False
  riu2XY    = False

  uniformXT = False
  riu2XT    = False

  uniformYT = False
  riu2YT    = False

  if(lbptypeXY=='uniform'):
    uniformXY = True
  else:    
    if(lbptypeXY=='riu2'):
      riu2XY=True

  if(lbptypeXT=='uniform'):
    uniformXT = True
  else:
    if(lbptypeXT=='riu2'):
      riu2XT=True

  if(lbptypeYT=='uniform'):
    uniformYT = True
  else:
    if(lbptypeYT=='riu2'):
      riu2YT=True


  timeLength = grayFaceNormFrameSequence.shape[0]
  width   = grayFaceNormFrameSequence.shape[1]
  height  = grayFaceNormFrameSequence.shape[2]


  #Creating the LBP operators for each plane
  lbp_XY = 0
  lbp_XT = 0
  lbp_YT = 0
  #XY
  if(nXY==4):
    lbp_XY = bob.ip.LBP4R(radius=rX, circular=cXY, uniform=uniformXY, rotation_invariant=riu2XY)
    lbp_XY.radius2 = rY
  elif(nXY==8):
    lbp_XY = bob.ip.LBP8R(radius=rX, circular=cXY, uniform=uniformXY, rotation_invariant=riu2XY)
    lbp_XY.radius2 = rY
  elif(nXY==16):
    lbp_XY = bob.ip.LBP16R(radius=rX, circular=cXY, uniform=uniformXY, rotation_invariant=riu2XY)
    lbp_XY.radius2 = rY


  #XT
  if(nXT==4):
    lbp_XT = bob.ip.LBP4R(radius=rX, circular=cXT, uniform=uniformXT, rotation_invariant=riu2XT)
    lbp_XT.radius2 = rT
  elif(nXT==8):
    lbp_XT = bob.ip.LBP8R(radius=rX, circular=cXT, uniform=uniformXT, rotation_invariant=riu2XT)
    lbp_XT.radius2 = rT
  elif(nXT==16):
    lbp_XT = bob.ip.LBP16R(radius=rX, circular=cXT, uniform=uniformXT, rotation_invariant=riu2XT)
    lbp_XT.radius2 = rT


  #YT
  if(nYT==4):
    lbp_YT = bob.ip.LBP4R(radius=rY, circular=cYT, uniform=uniformYT, rotation_invariant=riu2YT)
    lbp_YT.radius2 = rT
  elif(nYT==8):
    lbp_YT = bob.ip.LBP8R(radius=rY, circular=cYT, uniform=uniformYT, rotation_invariant=riu2YT)
    lbp_YT.radius2 = rT
  elif(nYT==16):
    lbp_YT = bob.ip.LBP16R(radius=rY, circular=cYT, uniform=uniformYT, rotation_invariant=riu2YT)
    lbp_YT.radius2 = rT

  #Creating the LBPTop object
  lbpTop = bob.ip.LBPTop(lbp_XY,lbp_XT,lbp_YT)

  #Alocating the LBPTop Images
  max_radius = max(lbp_XY.radius,lbp_XY.radius2,lbp_XT.radius2)

  xy_width  = width-(max_radius*2)
  xy_height = height-(max_radius*2)
  tLength   = timeLength-(max_radius*2)

  XY = numpy.zeros(shape=(tLength,xy_width,xy_height),dtype='uint16')
  XT = numpy.zeros(shape=(tLength,xy_width,xy_height),dtype='uint16')
  YT = numpy.zeros(shape=(tLength,xy_width,xy_height),dtype='uint16')

  #Calculanting the LBPTop Images
  lbpTop(grayFaceNormFrameSequence,XY,XT,YT)
  
  ### Calculating the histograms
  if(histrogramOutput):
    #XY
    histXY = numpy.zeros(shape=(XY.shape[0],lbp_XY.max_label))
    for i in range(XY.shape[0]):
      histXY[i] = bob.ip.histogram(XY[i], 0, lbp_XY.max_label-1, lbp_XY.max_label)
      #histogram normalization
      histXY[i] = histXY[i] / sum(histXY[i])

    #XT
    histXT = numpy.zeros(shape=(XT.shape[0],lbp_XT.max_label))
    for i in range(XT.shape[0]):
      histXT[i] = bob.ip.histogram(XT[i], 0, lbp_XT.max_label-1, lbp_XT.max_label)
      #histogram normalization
      histXT[i] = histXT[i] / sum(histXT[i])

    #YT
    histYT = numpy.zeros(shape=(YT.shape[0],lbp_YT.max_label))
    for i in range(YT.shape[0]):
      histYT[i] = bob.ip.histogram(YT[i], 0, lbp_YT.max_label-1, lbp_YT.max_label)
      #histogram normalization
      histYT[i] = histYT[i] / sum(histYT[i])

    return histXY,histXT,histYT

  else:
    #returning the volume
    return XY,XT,YT
   






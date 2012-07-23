#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Thu Jan 19 12:49:15 CET 2012

"""Support methods to compute the Local Binary Pattern (LBP) histogram of an image
"""

import numpy
import math
import bob

def lbphist(img, lbptype, elbptype='regular', rad=1, neighbors=8, circ=False):
  """Calculates a normalized LBP histogram over an image, using the bob LBP operator

  Keyword Parameters:

  img
    The image in gray-scale
  lbptype
    The type of the LBP operator (regular, uniform or riu2)
  elbptype
    The type of extended version of LBP (regular if not extended version is used, otherwise transitional, direction_coded or modified)
  rad
    The radius of the circle on which the points are taken (for circular LBP)
  neighbors
    The number of points around the central point on which LBP is computed (4, 8, 16)
  circ
    True if circular LBP is needed, False otherwise
  """

  elbps = {'regular':0, 'transitional':1, 'direction_coded':2, 'modified':0}

  if elbptype=='modified': 
    mct = True
  else: mct = False

  if lbptype == 'uniform':
    if neighbors==16:
      lbp = bob.ip.LBP16R(uniform=True, circular=circ, radius=rad, to_average=mct, elbp_type=elbps[elbptype])
    else: # we assume neighbors==8 in this case
      lbp = bob.ip.LBP8R(uniform=True, circular=circ, radius=rad, to_average=mct, elbp_type=elbps[elbptype])
  elif lbptype == 'riu2':
    if neighbors==16:
      lbp = bob.ip.LBP16R(uniform=True, rotation_invariant=True, radius=rad, circular=circ, to_average=mct, elbp_type=elbps[elbptype]) 
    else: # we assume neighbors==8 in this case
      lbp = bob.ip.LBP8R(uniform=True, rotation_invariant=True, radius=rad, circular=circ, to_average=mct, elbp_type=elbps[elbptype])
  else: # regular LBP
    if neighbors==16:
      lbp = bob.ip.LBP16R(circular=circ, radius=rad, to_average=mct, elbp_type=elbps[elbptype]) 
    else: # we assume neighbors==8 in this case
      lbp = bob.ip.LBP8R(circular=circ, radius=rad, to_average=mct, elbp_type=elbps[elbptype])
  
  lbpimage = numpy.ndarray(lbp.get_lbp_shape(img), 'uint16') # allocating the image with lbp codes
  lbp(img, lbpimage) # calculating the lbp image
  hist = bob.ip.histogram(lbpimage, 0, lbp.max_label-1, lbp.max_label)
  hist = hist / sum(hist) # histogram normalization
  return hist, 1 # the last argument is 1 if the frame was valid and 0 otherwise
     

def divideframe(frame, numbl):
  """ Divides given frame into numbl^2 blocks and returns list of the frames obtained in this way

  Keyword parameters:
  
  frame 
    The frame to be divided as a grey-scale image
  numbl 
    Square root of the number of blocks the frame is to be divided into (ex. 9 blocks => numbl = 3)
  """
  blockslist = [] 
  height = frame.shape[0] # current frame height
  width = frame.shape[1] # current frame width
  blheight = frame.shape[0] / numbl # height of the blocks
  blwidth = frame.shape[1] / numbl # width of the blocks
  newheight = blheight * numbl # width of the new smaller frame, the frame should be resized so that each subblock has the same size
  newwidth = blwidth * numbl # width of the new smaller frame, the frame should be resized so that each subblock has the same size
  
  newframe = frame[((height-newheight)/2):((height-newheight)/2+newheight), ((width-newwidth)/2):((width-newwidth)/2+newwidth)]  
  for i in range(0, numbl):
    for j in range(0, numbl):
      nextblock = newframe[(i*blheight):(i+1)*blheight, (j*blwidth):(j+1)*blwidth] # cut the subblock
      blockslist.append(nextblock)
  return blockslist # list of subblocks as frames


def divideframe_overlap(frame, numbl):
  """ Divides given frame into numbl^2 blocks with overlapping region of 13 pixels and returns list of the frames obtained in this way (hardcoded as in the paper: "Face Spoofing Detection from Single Images Using Micro-texture Analysis" - Maatta, Hadid, Pietikainen)

  Keyword parameters:
  
  frame 
    The frame to be divided as a grey-scale image
  numbl 
    Square root of the number of blocks the frame is to be divided into (ex. 9 blocks => numbl = 3)
  """

  # the size of the overlap region (hard-coded for the moment)
  # Note that Jukka's paper mention an overlap of 14, but considers that the
  # blocking operation is done on the top of the LBP image. The LBP image is
  # of the size N - 2, where N is the original image size (for NUAA that is
  # 64x64). Here, we calculate the LBP histograms directly on the image.
  # To make a compatible overlap setting, we need to set it to 16. If we
  # create blocks of the original image with a 16 overlap, than it is
  # equivalent to creating blocks with 14 overlap on an LBP image generated 
  # from the original image.
  ovl = 16
  blockslist = []
  height = frame.shape[0] + (numbl-1)*ovl # current frame height
  width = frame.shape[1] + (numbl-1)*ovl # current frame width
  blheight = height / numbl # height of the blocks
  blwidth = width / numbl # width of the blocks
    
  for i in range(0, numbl):
    for j in range(0, numbl):
      y0 = (i*blheight) - i*ovl
      x0 = (j*blwidth)  - j*ovl
      y1 = (i+1)*blheight - i*ovl
      x1 = (j+1)*blwidth  - j*ovl
      nextblock = frame[y0:y1, x0:x1] # cut the subblock
      blockslist.append(nextblock)
  return blockslist # list of subblocks as frames


def lbphist_frame(frame, lbptype, elbptype='regular', radius=1, neighbors=8, circ=False, numbl=1, overlap=False):
  """Calculates the normalized LBP histogram of an image, by blocks or on the full image, using the bob LBP operator. The blocks can be overlapping or not

  Keyword Parameters:

  frame
    The frame as a gray-scale image
  lbptype
    The type of the LBP operator (regular, uniform or riu2)
  elbptype
    The type of extended version of LBP (regular if not extended version is used, otherwise transitional, direction_coded or modified)
  radius
    The radius of the circle on which the points are taken (for circular LBP)
  neighbors
    The number of points around the central point on which LBP is computed (4, 8, 16)
  circ
    True if circular LBP is needed, False otherwise
  numbl
    Square root of the number of blocks the frame is to be divided into (ex. 9 blocks => numbl = 3)
  overlap
    True for overlapping blocks (hardcoded to 16 pixels overlap, as in the paper "Face Spoofing Detection from Single Images Using Micro-texture Analysis" - Maatta, Hadid, Pietikainen))
    
  """
  if numbl == 1: 
    finalhist, vf = lbphist(frame, lbptype, elbptype, radius, neighbors, circ) 
  else:
    finalhist = numpy.array([])
    if overlap == False:
      blockslist = divideframe(frame, numbl) # divide the frame into blocks 
    else:
      blockslist = divideframe_overlap(frame, numbl) # divide the frame into overlapping blocks
    for bl in blockslist: # calculate separate histogram for each frame subblock
      hist, vf = lbphist(bl, lbptype, elbptype, radius, neighbors, circ) 
      finalhist = numpy.append(finalhist, hist) # concatenate the subblocks' already normalized histograms
  return finalhist, vf


def lbphist_facenorm(frame, lbptype, bbx, sz, elbptype='regular', radius=1, neighbors=8, circ=False, numbl=1, overlap=False, bbxsize_filter=0):
  """Calculates the normalized 3x3 LBP histogram over a given bounding box (bbx) in an image (around the detected face for example), using the bob LBP operator, after first rescaling bbx to a predefined size. If bbx is None or invalid, returns an empty histogram.

  Keyword Parameters:

  frame
    The frame as a gray-scale image
  lbptype
    The type of the LBP operator (regular, uniform or riu2)
  bbx
    the face bounding box
  sz
    The size of the rescaled face bounding box
  elbptype
    The type of extended version of LBP (regular if not extended version is used, otherwise transitional, direction_coded or modified)
  radius
    The radius of the circle on which the points are taken (for circular LBP)
  neighbors
    The number of points around the central point on which LBP is computed (4, 8, 16)
  circ
    True if circular LBP is needed, False otherwise
  numbl
    Square root of the number of blocks the frame is to be divided into (ex. 9 blocks => numbl = 3)
  overlap
    True for overlapping blocks (hardcoded to 16 pixels overlap, as in the paper "Face Spoofing Detection from Single Images Using Micro-texture Analysis" - Maatta, Hadid, Pietikainen))
  bbxsize_filter
    Considers as invalid all the bounding boxes with size smaller then this value
  """ 
  # hardcoding the number of bins for the LBP variants
  if neighbors == 16:   lbphistlength = {'regular':65536, 'riu2':18, 'uniform':243}
  else:  lbphistlength = {'regular':256, 'riu2':10, 'uniform':59} 

  if bbx and bbx.is_valid() and bbx.height > bbxsize_filter:
    cutframe = frame[bbx.y:(bbx.y+bbx.height),bbx.x:(bbx.x+bbx.width)] # cutting the box region
    tempbbx = numpy.ndarray((sz, sz), 'float64')
    normbbx = numpy.ndarray((sz, sz), 'uint8')
    bob.ip.scale(cutframe, tempbbx) # normalization
    tempbbx_ = tempbbx + 0.5
    tempbbx_ = numpy.floor(tempbbx_)
    normbbx = numpy.cast['uint8'](tempbbx_)
    finalhist, vf = lbphist_frame(normbbx, lbptype, elbptype, radius, neighbors, circ, numbl, overlap)
    return finalhist, vf # the last argument is 1 if the frame was valid and 0 otherwise
  return  numpy.array(numbl * numbl * lbphistlength[lbptype] * [0]), 0 # return empty histogram if there is no valid bounding box (example: detected face in the frame)


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
    tempbbx_ = tempbbx + 0.5 #TODO: UNDERSTAND THIS
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
"""
def lbptophist(grayFaceNormFrameSequence,nXY,nXT,nYT,rX,rY,rT,cXY,cXT,cYT,lbptypeXY,lbptypeXT,lbptypeYT):
  
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

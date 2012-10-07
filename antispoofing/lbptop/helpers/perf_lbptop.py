#!/usr/bin/env python
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Fri 20 Jul 03:30:00 2012

import os
import bob
import numpy
from antispoofing.utils.ml import *

import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as mpl

from . import *

"""
A Set of helper functions for the LBPTOP
"""


def roc_lbptop(pos,neg,label,hold=False,linestyle='--',filename="ROC.pdf"):
  """Plots the ROC curve using Matplotlib"""

  import matplotlib
  matplotlib.use('pdf')
  import matplotlib.pyplot as mpl


  bob.measure.plot.roc(neg, pos, npoints=100,
      linestyle=linestyle, dashes=(6,2), alpha=0.5, label=label)
  

  if(not hold):
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(filename)

    mpl.title("ROC Curve")
    mpl.xlabel('FRR (%)')
    mpl.ylabel('FAR (%)')
    mpl.grid(True, alpha=0.3)
    mpl.legend()
    #mpl.show()
    #mpl.savefig(filename)
    pp.savefig()
    pp.close()



def det_lbptop(pos,neg,label,hold=False,linestyle='--',filename="DET.pdf"):
  """Plots the DET curve using Matplotlib"""

  import matplotlib
  matplotlib.use('pdf')
  import matplotlib.pyplot as mpl

  bob.measure.plot.det(neg, pos, npoints=100,
      linestyle=linestyle, dashes=(6,2), alpha=0.5, label=label)


  if(not hold):
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(filename)

    bob.measure.plot.det_axis([1, 40, 1, 40])

    mpl.title("DET Curve")
    mpl.xlabel('FRR (%)')
    mpl.ylabel('FAR (%)')
    mpl.grid(True, alpha=0.3)
    mpl.legend(loc='upper left')
    #mpl.show()
    #mpl.savefig(filename)
    pp.savefig()
    pp.close()


"""
Return a text with the perf table for each plane in the LBPTOP
"""
def perfTable(models,develTexts,testTexts,thres,energy=0):
  tbl = []
  
  for i in range(len(models)):
    tbl.append(models[i])
    if energy>0:
      tbl.append("EER @devel - energy kept after PCA = %.2f" % (energy))

    tbl.append(" threshold: %.4f" % thres[i])
    tbl.append(develTexts[i])
    tbl.append(testTexts[i])
    tbl.append("*****")

  txt = ''.join([k+'\n' for k in tbl])
  return txt



"""
Save all the information needed for an LBPTOP based countermeasure

@param trainRealScores 
@param trainAttackScores 
@param develRealScores
@param develAttackScores
@param testRealScores
@param testAttackScores
@param thresholds
@param models The name of the LBPTOP models
@param lines The color lines for the DET curve
@param develTexts 
@param testTexts 
@param energy The energy kept by PCA algorithm
@param outputDir The output dir to write all that stuff
"""
def saveCounterMeasureResults(trainRealScores,trainAttackScores,develRealScores,develAttackScores,testRealScores,testAttackScores,thresholds,models,lines,develTexts,testTexts,energy,outputDir,scoresRange=(-5,5)):

  ################################
  # Saving the performance table
  ################################
  txt = perfTable(models,develTexts,testTexts,thresholds,energy=energy)
  tf = open(os.path.join(outputDir, 'perf_table.txt'), 'w')
  tf.write(txt)



  ################################
  # Plotting the DET curver for each plane
  ################################
  for i in range(len(models)):

    if(i==len(models)-1):
      hold=False
    else:
      hold=True

    #Plotting the DET for each plane
    det_lbptop(testRealScores[i],testAttackScores[i],models[i],hold,linestyle=lines[i],filename=os.path.join(outputDir,"DET.pdf"))



  ################################
  # Plotting the score distribution
  ################################
  pp = PdfPages(os.path.join(outputDir,"Scores-Distribution.pdf"))
  for i in range(len(models)):
    fig = mpl.figure()

    train = [trainRealScores[i],trainAttackScores[i]]
    devel = [develRealScores[i]	,develAttackScores[i]]
    test  = [testRealScores[i],testAttackScores[i]]

    perf.score_distribution_plot(test, devel, train, bins=60, thres=thresholds[i],scoresRange=scoresRange,title=" - " + models[i])
    pp.savefig(fig)
  pp.close()






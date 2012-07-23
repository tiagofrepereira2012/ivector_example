#!/usr/bin/env python
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Fri 20 Jul 03:30:00 2012

import os
import bob
import numpy
import re

def roc_lbptop(pos,neg,label,hold=False,linestyle='--',filename="ROC.png"):
  """Plots the ROC curve using Matplotlib"""

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as mpl


  bob.measure.plot.roc(neg, pos, npoints=100,
      linestyle=linestyle, dashes=(6,2), alpha=0.5, label=label)
  

  if(not hold):
    mpl.title("ROC Curve")
    mpl.xlabel('FRR (%)')
    mpl.ylabel('FAR (%)')
    mpl.grid(True, alpha=0.3)
    mpl.legend()
    #mpl.show()
    mpl.savefig(filename)



def det_lbptop(pos,neg,label,hold=False,linestyle='--',filename="DET.png"):
  """Plots the DET curve using Matplotlib"""

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as mpl
  mpl.use('Agg')

  bob.measure.plot.det(neg, pos, npoints=100,
      linestyle=linestyle, dashes=(6,2), alpha=0.5, label=label)


  if(not hold):
   
    bob.measure.plot.det_axis([1, 40, 1, 40])


    mpl.title("DET Curve")
    mpl.xlabel('FRR (%)')
    mpl.ylabel('FAR (%)')
    mpl.grid(True, alpha=0.3)
    mpl.legend()
    #mpl.show()
    mpl.savefig(filename)


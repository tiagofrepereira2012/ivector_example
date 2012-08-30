#!/usr/bin/env python
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Fri 20 Jul 03:30:00 2012

import os
import bob
import numpy
import re

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
    mpl.legend()
    #mpl.show()
    #mpl.savefig(filename)
    pp.savefig()
    pp.close()


#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Thu Oct 04 14:44:00 CEST 2012

def perfTable(databases,develTexts,testTexts,thres):
  tbl = []
  
  for i in range(len(databases)):
    tbl.append(databases[i])

    tbl.append(" threshold: %.4f" % thres[i])
    tbl.append(develTexts[i])
    tbl.append(testTexts[i])
    tbl.append("*****")

  txt = ''.join([k+'\n' for k in tbl])
  return txt


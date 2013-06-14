===============================================================================
IVector sample
===============================================================================

This package implements an IVector based system using the toolbox Bob.

If you use this package and/or its results, please cite the following publications:

1. Bob as the core framework used to run the experiments::

    @inproceedings{Anjos_ACMMM_2012,
      author = {A. Anjos AND L. El Shafey AND R. Wallace AND M. G\"unther AND C. McCool AND S. Marcel},
      title = {Bob: a free signal processing and machine learning toolbox for researchers},
      year = {2012},
      month = oct,
      booktitle = {20th ACM Conference on Multimedia Systems (ACMMM), Nara, Japan},
      publisher = {ACM Press},
    }



Installation
------------

First, clone the package from the github::

$ git clone https://github.com/tiagofrepereira2012/ivector_example/

This code uses the bob 1.2.0 using the mr.developer recipe; please edit the file ``buildout.cfg`` and set the ``prefixes`` variable to the correct path to your Bob installation::

$ prefixes = <bob path>

Finally, follow the recipe bellow::

$ <bob path>/python bootstrap.py
$ ./bin/buildout


User guide
----------

Until now it is developed only the training of the Total Variability Matrix.

To gerenate the total variability matrix for each iteration::

$ ./bin/ivector_trainT.py --help


Problems
--------

In case of problems, please contact me.




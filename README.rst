===============================================================================
LBP-TOP based countermeasure against facial spoofing attacks
===============================================================================


This package implements an LBP-TOP based countermeasure to spoofing attacks to face recognition systems as described at the paper LBP-TOP based countermeasure against facial spoofing attacks, International Workshop on Computer Vision With Local Binary Pattern Variants, 2012.


If you use this package and/or its results, please cite the following publications:

1. The original paper with the counter-measure explained in details::

    @inproceedings{Pereira_LBP_2012,
      author = {Pereira, Tiago de Freitas and Anjos, Andr{\'{e}} and De Martino, Jos{\'{e}} Mario and Marcel, S{\'{e}}bastien},
      keywords = {Attack, Countermeasures, Counter-Spoofing, Face Recognition, Liveness Detection, Replay, Spoofing},
      month = dec,
      title = {LBP-TOP based countermeasure against facial spoofing attacks},
      journal = {ACCV 2012},
      year = {2012},
    }


2. Bob as the core framework used to run the experiments::

    @inproceedings{Anjos_ACMMM_2012,
      author = {A. Anjos AND L. El Shafey AND R. Wallace AND M. G\"unther AND C. McCool AND S. Marcel},
      title = {Bob: a free signal processing and machine learning toolbox for researchers},
      year = {2012},
      month = oct,
      booktitle = {20th ACM Conference on Multimedia Systems (ACMMM), Nara, Japan},
      publisher = {ACM Press},
    }


Raw Data
--------
 
The dataset used in the paper is REPLAY-ATTACK database and it is publicly available. It should be downloaded and
installed **prior** to using the programs described in this package. Visit
`the REPLAY-ATTACK database page <https://www.idiap.ch/dataset/replayattack>`_ for more information.


Installation
------------

.. note:: 

  If you are reading this page through our GitHub portal and not through PyPI,
  note **the development tip of the package may not be stable** or become
  unstable in a matter of moments.

  Go to `http://pypi.python.org/pypi/antispoofing.lbptop
  <http://pypi.python.org/pypi/antispoofing.lbptop>`_ to download the latest
  stable version of this package.

There are 2 options you can follow to get this package installed and
operational on your computer: you can use automatic installers like `pip
<http://pypi.python.org/pypi/pip/>`_ (or `easy_install
<http://pypi.python.org/pypi/setuptools>`_) or manually download, unpack and
use `zc.buildout <http://pypi.python.org/pypi/zc.buildout>`_ to create a
virtual work environment just for this package.

Using an automatic installer
============================

Using ``pip`` is the easiest (shell commands are marked with a ``$`` signal)::

  $ pip install antispoofing.lbptop

You can also do the same with ``easy_install``::

  $ easy_install antispoofing.lbptop

This will download and install this package plus any other required
dependencies. It will also verify if the version of Bob you have installed
is compatible.

This scheme works well with virtual environments by `virtualenv
<http://pypi.python.org/pypi/virtualenv>`_ or if you have root access to your
machine. Otherwise, we recommend you use the next option.

Using ``zc.buildout``
=====================

Download the latest version of this package from `PyPI
<http://pypi.python.org/pypi/antispoofing.lbptop>`_ and unpack it in your
working area. The installation of the toolkit itself uses `buildout
<http://www.buildout.org/>`_. You don't need to understand its inner workings
to use this package. Here is a recipe to get you started::
  
  $ python bootstrap.py 
  $ ./bin/buildout

These 2 commands should download and install all non-installed dependencies and
get you a fully operational test and development environment.

.. note::

  The python shell used in the first line of the previous command set
  determines the python interpreter that will be used for all scripts developed
  inside this package. Because this package makes use of `Bob
  <http://idiap.github.com/bob>`_, you must make sure that the ``bootstrap.py``
  script is called with the **same** interpreter used to build Bob, or
  unexpected problems might occur.

  If Bob is installed by the administrator of your system, it is safe to
  consider it uses the default python interpreter. In this case, the above 3
  command lines should work as expected. If you have Bob installed somewhere
  else on a private directory, edit the file ``buildout.cfg`` **before**
  running ``./bin/buildout``. Find the section named ``external`` and edit the
  line ``egg-directories`` to point to the ``lib`` directory of the Bob
  installation you want to use. For example::

    [external]
    recipe = xbob.buildout:external
    egg-directories=/Users/crazyfox/work/bob/build/lib

User Guide
----------

It is assumed you have followed the installation instructions for the package
and got this package installed and the REPLAY-ATTACK database downloaded and
uncompressed in a directory. You should have all required utilities sitting
inside a binary directory depending on your installation strategy (utilities
will be inside the ``bin`` if you used the buildout option). We expect that the
video files downloaded for the REPLAY-ATTACK database are installed in a
sub-directory called ``database`` at the root of the package.  You can use a
link to the location of the database files, if you don't want to have the
database installed on the root of this package::

  $ ln -s /path/where/you/installed/the/print-attack-database database

If you don't want to create a link, use the ``--input-dir`` flag to specify
the root directory containing the database files. That would be the directory
that *contains* the sub-directories ``train``, ``test``, ``devel`` and
``face-locations``.


Calculate the multiresolution and single resolution LBP-TOP features
====================================================================

The first stage of the process is calculating the feature vectors, which are essentially LBP-TOP histograms (XY, XT and YT directions) for each frame of the video.

The program to be used is `script/calclbptop_multiple_radius.py`.

The resulting histograms will be put in .hdf5 files in the default output directory `./lbp_features`.

.. code-block:: shell

  $ ./bin/calclbptop_multiple_radius.py replay


To gerate LBP-TOP features following the multiresolution strategy in time domain, it is necessary to set different values for Rt. For example, to generate a multiresolution description in time domain for Rt=[1-4] the code is the follows:

.. code-block:: shell

  $ ./bin/calclbptop_multiple_radius.py -rT 1 2 3 4 replay


To gerate a single resolution strategy in time domain, it is necessary to set only one value for Rt. For example, to generate a single resolution description in time domain for Rt=1 the code is the follows:

.. code-block:: shell

  $ ./bin/calclbptop_multiple_radius.py -rT 1 replay



To see all the options for the scripts `calclbptop_multiple_radius.py` just type
`--help` at the command line.



Classification using Chi-2 Distance
====================================================================

The clasification using Chi-2 distance consists of two steps. The first one is creating the histogram model (average LBP-TOP histogram for each plane and it combinations of all the real access videos in the training set). The second step is comparison of the features of development and test videos to the model histogram and writing the results.

The script to use for creating the histogram model is `script/mkhistmodel_lbptop.py`. It expects that the LBP-TOP features of the videos are stored in a folder `./lbp_features`. The model histogram will be written in the default output folder `./res`. You can change this default features by setting the input arguments. To execute this script, just run:

.. code-block:: shell

  $ ./bin/mkhistmodel_lbptop.py

The script for performing Chi-2 histogram comparison is `script/cmphistmodels_lbptop.py`, and it assumes that the model histogram has been already created. It makes use of the utility script `spoof/chi2.py` and `ml/perf.py` for writing the results in a file. The default input directory is `./lbp_features`, while the default input directory for the histogram model as well as default output directory is `./res`. To execute this script, just run: 

.. code-block:: shell

  $ ./bin/cmphistmodel_lbptop.py

The performance results will be calculated for each LBP-TOP planes and the combinations XT+YT and XY+XT+YT.

To see all the options for the scripts `mkhistmodel_lbptop.py` and `cmphistmodels_lbptop.py`, just type `--help` at the command line.



Classification with Linear Discriminant Analysis (LDA)
====================================================================

The classification with LDA is performed using the script `script/ldatrain_lbptop.py`. It makes use of the scripts `ml/lda.py`, `ml/pca.py` (if PCA reduction is performed on the data) and `ml/norm.py` (if the data need to be normalized). The default input and output directories are `./lbp_features` and `./res`. To execute the script with the default parameters, call:

.. code-block:: shell

  $ ./bin/ldatrain_lbptop.py

The performance results will be calculated for each LBP-TOP planes and the combinations XT+YT and XY+XT+YT.

To see all the options for this script, just type `--help` at the command line.


Classification with Support Vector Machine (SVM)
====================================================================

The classification with SVM is performed using the script `script/svmtrain_lbptop.py`. It makes use of the scripts `ml/pca.py` (if PCA reduction is performed on the data) and `ml/norm.py` (if the data need to be normalized). The default input and output directories are `./lbp_features` and `./res`. To execute the script with the default parameters, call:

.. code-block:: shell

  $ ./bin/svmtrain_lbptop.py

The performance results will be calculated for each LBP-TOP planes and the combinations XT+YT and XY+XT+YT.

To see all the options for this script, just type `--help` at the command line.

Generating paper results
====================================================================

The next code blocks are codes to generate the results from lines 4, 5, 6, 7, 8 of Table 1.

- **Line 4:**
.. code-block:: shell

  #Extracting the LBP-TOP features
  $ ./bin/calclbptop_multiple_radius.py --directory lbptop_features/ --input-dir database/ -rX 1 -rY 1 -rT 1 2 3 4 5 6 -cXY -cXT -cYT --lbptypeXY riu2 --lbptypeXT riu2 --lbptypeYT riu2 replay

  #Running the SVM machine
  $ ./bin/svmtrain_lbptop.py  -n --input-dir lbptop_features/ --output-dir res/ replay

  #Extracting the scores for each plane
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XY-plane.txt --normalization-file svm_normalization_XY-plane.txt --machine-type SVM --plane XY --output-dir res/scores/scores_XY replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XT-Plane.txt --normalization-file svm_normalization_XT-Plane.txt --machine-type SVM --plane XT --output-dir res/scores/scores_XT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_YT-Plane.txt --normalization-file svm_normalization_YT-Plane.txt --machine-type SVM --plane YT --output-dir res/scores/scores_YT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XT-YT-Plane.txt --normalization-file svm_normalization_XT-YT-Plane.txt --machine-type SVM --plane XT-YT --output-dir res/scores/scores_XT-YT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XY-XT-YT-plane.txt --normalization-file svm_normalization_XY-XT-YT-plane.txt --machine-type SVM --plane XY-XT-YT --output-dir res/scores/scores_XY-XT-YT replay

  #Result Analysis
  $./bin/result_analysis.py --scores-dir res/scores/ --output-dir res/results/ replay


- **Line 5:**
.. code-block:: shell

  #Extracting the LBP-TOP features
  $ ./bin/calclbptop_multiple_radius.py --directory lbptop_features/ --input-dir database/ -rX 1 -rY 1 -rT 1 2 3 4 5 6 -cXY -cXT -cYT -nXT 4 -nYT 4 replay

  #Running the SVM machine
  $ ./bin/svmtrain_lbptop.py  -n --input-dir lbptop_features/ --output-dir res/ replay

  #Extracting the scores for each plane
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XY-plane.txt --normalization-file svm_normalization_XY-plane.txt --machine-type SVM --plane XY --output-dir res/scores/scores_XY replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XT-Plane.txt --normalization-file svm_normalization_XT-Plane.txt --machine-type SVM --plane XT --output-dir res/scores/scores_XT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_YT-Plane.txt --normalization-file svm_normalization_YT-Plane.txt --machine-type SVM --plane YT --output-dir res/scores/scores_YT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XT-YT-Plane.txt --normalization-file svm_normalization_XT-YT-Plane.txt --machine-type SVM --plane XT-YT --output-dir res/scores/scores_XT-YT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XY-XT-YT-plane.txt --normalization-file svm_normalization_XY-XT-YT-plane.txt --machine-type SVM --plane XY-XT-YT --output-dir res/scores/scores_XY-XT-YT replay

  #Result Analysis
  $./bin/result_analysis.py --scores-dir res/scores/ --output-dir res/results/ replay



- **Line 6:**
.. code-block:: shell

  #Extracting the LBP-TOP features
  $ ./bin/calclbptop_multiple_radius.py --directory lbptop_features/ --input-dir database/ -rX 1 -rY 1 -rT 1 2 3 4 -cXY -cXT -cYT replay

  #Running the SVM machine
  $ ./bin/svmtrain_lbptop.py  -n --input-dir lbptop_features/ --output-dir res/ replay

  #Extracting the scores for each plane
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XY-plane.txt --normalization-file svm_normalization_XY-plane.txt --machine-type SVM --plane XY --output-dir res/scores/scores_XY replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XT-Plane.txt --normalization-file svm_normalization_XT-Plane.txt --machine-type SVM --plane XT --output-dir res/scores/scores_XT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_YT-Plane.txt --normalization-file svm_normalization_YT-Plane.txt --machine-type SVM --plane YT --output-dir res/scores/scores_YT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XT-YT-Plane.txt --normalization-file svm_normalization_XT-YT-Plane.txt --machine-type SVM --plane XT-YT --output-dir res/scores/scores_XT-YT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XY-XT-YT-plane.txt --normalization-file svm_normalization_XY-XT-YT-plane.txt --machine-type SVM --plane XY-XT-YT --output-dir res/scores/scores_XY-XT-YT replay

  #Result Analysis
  $./bin/result_analysis.py --scores-dir res/scores/ --output-dir res/results/ replay


- **Line 7:**
.. code-block:: shell

  #Extracting the LBP-TOP features
  $ ./bin/calclbptop_multiple_radius.py --directory lbptop_features/ --input-dir database/ -rX 1 -rY 1 -rT 1 2 -cXY -cXT -cYT --lbptypeXY regular --lbptypeXT regular --lbptypeYT regular replay

  #Running the SVM machine
  $ ./bin/svmtrain_lbptop.py  -n --input-dir lbptop_features/ --output-dir res/ replay

  #Extracting the scores for each plane
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XY-plane.txt --normalization-file svm_normalization_XY-plane.txt --machine-type SVM --plane XY --output-dir res/scores/scores_XY replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XT-Plane.txt --normalization-file svm_normalization_XT-Plane.txt --machine-type SVM --plane XT --output-dir res/scores/scores_XT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_YT-Plane.txt --normalization-file svm_normalization_YT-Plane.txt --machine-type SVM --plane YT --output-dir res/scores/scores_YT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XT-YT-Plane.txt --normalization-file svm_normalization_XT-YT-Plane.txt --machine-type SVM --plane XT-YT --output-dir res/scores/scores_XT-YT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XY-XT-YT-plane.txt --normalization-file svm_normalization_XY-XT-YT-plane.txt --machine-type SVM --plane XY-XT-YT --output-dir res/scores/scores_XY-XT-YT replay

  #Result Analysis
  $./bin/result_analysis.py --scores-dir res/scores/ --output-dir res/results/ replay


- **Line 8:**
.. code-block:: shell

  #Extracting the LBP-TOP features
  $ ./bin/calclbptop_multiple_radius.py --directory lbptop_features/ --input-dir database/ -rX 1 -rY 1 -rT 1 2 -cXY -cXT -cYT -nXT 16 -nYT 16 replay

  #Running the SVM machine
  $ ./bin/svmtrain_lbptop.py  -n --input-dir lbptop_features/ --output-dir res/ replay

  #Extracting the scores for each plane
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XY-plane.txt --normalization-file svm_normalization_XY-plane.txt --machine-type SVM --plane XY --output-dir res/scores/scores_XY replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XT-Plane.txt --normalization-file svm_normalization_XT-Plane.txt --machine-type SVM --plane XT --output-dir res/scores/scores_XT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_YT-Plane.txt --normalization-file svm_normalization_YT-Plane.txt --machine-type SVM --plane YT --output-dir res/scores/scores_YT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XT-YT-Plane.txt --normalization-file svm_normalization_XT-YT-Plane.txt --machine-type SVM --plane XT-YT --output-dir res/scores/scores_XT-YT replay
  $ ./bin/make_scores.py --features-dir lbptop_features --machine-file svm_machine_XY-XT-YT-plane.txt --normalization-file svm_normalization_XY-XT-YT-plane.txt --machine-type SVM --plane XY-XT-YT --output-dir res/scores/scores_XY-XT-YT replay

  #Result Analysis
  $./bin/result_analysis.py --scores-dir res/scores/ --output-dir res/results/ replay



After that, it's recommended to go out for a long coffee.

Problems
--------

In case of problems, please contact any of the authors of the paper.




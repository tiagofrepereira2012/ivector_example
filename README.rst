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

3. The original paper with the counter-measure explained in details::

    @inproceedings{Anjos_IJCB_2011,
      author = {Anjos, Andr{\'{e}} and Marcel, S{\'{e}}bastien},
      keywords = {Attack, Counter-Measures, Counter-Spoofing, Disguise, Dishonest Acts, Face Recognition, Face Verification, Forgery, Liveness Detection, Replay, Spoofing, Trick},
      month = oct,
      title = {Counter-Measures to Photo Attacks in Face Recognition: a public database and a baseline},
      journal = {International Joint Conference on Biometrics 2011},
      year = {2011},
      pdf = {http://publications.idiap.ch/downloads/papers/2011/Anjos_IJCB_2011.pdf}
    }




Raw Data
--------
 
The dataset used in the paper is REPLAY-ATTACK database and it is publicly available. It should be downloaded and
installed **prior** to using the programs described in this package. Visit
`the REPLAY-ATTACK database page <https://www.idiap.ch/dataset/replayattack>`_ for more information.


Installation
------------

To follow these instructions locally you will need a local copy of this
package. Start by cloning this project with something like (shell commands are marked with a
``$`` signal)::

  $ git clone --depth=1 https://github.com/bioidiap/antispoofing.lbptop.git
  $ cd antispoofing.lbptop
  $ rm -rf .git # you don't need the git directories...

Alternatively, you can use the github tarball API to download the package::

  $ wget --no-check-certificate https://github.com/bioidiap/antispoofing.lbptop/tarball/master -O- | tar xz 
  $ mv bioidiap-antispoofing-* antispoofing.lbptop

Installation of the toolkit uses the `buildout <http://www.buildout.org/>`_
build environment. You don't need to understand its inner workings to use this
package. Here is a recipe to get you started::
  
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
  else on a private directory, edit the file ``localbob.cfg`` and use that
  with ``buildout`` instead::

    $ python boostrap.py
    $ # edit localbob.cfg
    $ ./bin/buildout -c localbob.cfg

Usage
-----

Please refer to the documentation inside the ``doc`` directory of this package
for further instructions on the functionality available.


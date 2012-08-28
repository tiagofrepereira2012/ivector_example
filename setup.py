#!/usr/bin/env python
#Ivana Chingovska <tiagofrepereira@gmail.com>
#Sun Jul  8 20:35:55 CEST 2012

from setuptools import setup, find_packages

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    name='antispoofing.lbp',
    version='1.0',
    description='LBP-TOP based countermeasure against facial spoofing attacks',
    url='http://github.com/bioidiap/antispoofing.lbptop',
    license='LICENSE.txt',
    author_email='Tiago de Freitas Pereira <tiagofrepereira@gmail.com>',
    long_description=open('doc/howto.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),

    install_requires=[
        "bob",      # base signal proc./machine learning library
        "argparse", # better option parsing
        "xbob.db.replay", 
    ],

    entry_points={
      'console_scripts': [
        'calclbp.py = antispoofing.lbp.script.calclbp:main',
        'calclbptop.py = antispoofing.lbp.script.calclbptop:main',
        'calclbptop_videos.py = antispoofing.lbp.script.calclbptop_videos:main',
        'calclbptop_videos_all_bins.py = antispoofing.lbp.script.calclbptop_videos_all_bins:main',
        'calclbptop_histacum.py = antispoofing.lbp.script.calclbptop_histacum:main',
        'calcframelbp.py = antispoofing.lbp.script.calcframelbp:main',
        'calcframelbp_5quatities.py = antispoofing.lbp.script.calcframelbp_5quatities:main',
        'mkhistmodel.py = antispoofing.lbp.script.mkhistmodel:main',
        'mkhistmodel_lbptop.py = antispoofing.lbp.script.mkhistmodel_lbptop:main',
        'cmphistmodels.py = antispoofing.lbp.script.cmphistmodels:main',
        'cmphistmodels_lbptop.py = antispoofing.lbp.script.cmphistmodels_lbptop:main',
        'ldatrain_lbp.py = antispoofing.lbp.script.ldatrain_lbp:main',
        'ldatrain_lbptop.py = antispoofing.lbp.script.ldatrain_lbptop:main',
        'svmtrain_lbp.py = antispoofing.lbp.script.svmtrain_lbp:main',
        'svmtrain_lbptop.py = antispoofing.lbp.script.svmtrain_lbptop:main',
        'mlptrain_lbptop.py = antispoofing.lbp.script.mlptrain_lbptop:main',
        'calclbptop_multiple_radius.py = antispoofing.lbp.script.calclbptop_multiple_radius:main',
        ],
      },

)

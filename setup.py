#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Sun Jul  8 20:35:55 CEST 2012

from setuptools import setup, find_packages

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    name='antispoofing.lbptop',
    version='1.0.0a0',
    description='LBP-TOP based countermeasure against facial spoofing attacks',
    url='http://github.com/bioidiap/antispoofing.lbptop',
    license='LICENSE.txt',
    author='Tiago de Freitas Pereira',
    author_email='tiagofrepereira@gmail.com',
    long_description=open('doc/howto.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),

    install_requires=[
        "bob >= 1.1",      # base signal proc./machine learning library
        "argparse", # better option parsing
        "xbob.db.replay", #Replay database
    ],

    namespace_packages = [
      'antispooofing',
      ],

    entry_points={
      'console_scripts': [
        'calclbptop.py = antispoofing.lbp.script.calclbptop:main',
        'mkhistmodel_lbptop.py = antispoofing.lbp.script.mkhistmodel_lbptop:main',
        'cmphistmodels_lbptop.py = antispoofing.lbp.script.cmphistmodels_lbptop:main',
        'ldatrain_lbptop.py = antispoofing.lbp.script.ldatrain_lbptop:main',
        'svmtrain_lbptop.py = antispoofing.lbp.script.svmtrain_lbptop:main',
        'calclbptop_multiple_radius.py = antispoofing.lbp.script.calclbptop_multiple_radius:main',
        ],
      },

)

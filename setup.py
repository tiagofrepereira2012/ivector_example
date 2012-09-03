#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Sun Jul  8 20:35:55 CEST 2012

from setuptools import setup, find_packages

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    name='antispoofing.lbptop',
    version='1.1.0a0',
    description='LBP-TOP based countermeasure against facial spoofing attacks',
    url='http://pypi.python.org/pypi/antispoofing.lbptop',
    license='GPLv3',
    author='Tiago de Freitas Pereira',
    author_email='tiagofrepereira@gmail.com',
    long_description=open('README.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_packages_data = True,

    install_requires=[
        "bob >= 1.0",      # base signal proc./machine learning library
        "xbob.db.replay", #Replay database
    ],

    namespace_packages = [
      'antispoofing',
      ],

    entry_points={
      'console_scripts': [
        'mkhistmodel_lbptop.py = antispoofing.lbptop.script.mkhistmodel_lbptop:main',
        'cmphistmodels_lbptop.py = antispoofing.lbptop.script.cmphistmodels_lbptop:main',
        'ldatrain_lbptop.py = antispoofing.lbptop.script.ldatrain_lbptop:main',
        'svmtrain_lbptop.py = antispoofing.lbptop.script.svmtrain_lbptop:main',
        'calclbptop_multiple_radius.py = antispoofing.lbptop.script.calclbptop_multiple_radius:main',
        ],
      },

)

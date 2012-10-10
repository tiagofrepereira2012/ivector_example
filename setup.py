#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Sun Jul  8 20:35:55 CEST 2012

from setuptools import setup, find_packages

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    name='antispoofing.lbptop',
    version='1.0.0',
    description='LBP-TOP based countermeasure against facial spoofing attacks',
    url='http://pypi.python.org/pypi/antispoofing.lbptop',
    license='GPLv3',
    author='Tiago de Freitas Pereira',
    author_email='tiagofrepereira@gmail.com',
    long_description=open('README.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data = True,

    install_requires=[
        "bob >= 1.1.0",      # base signal proc./machine learning library
        "xbob.db.replay",
        "antispoofing.utils",  #Utils Package
    ],

    namespace_packages = [
      'antispoofing',
      ],

    entry_points={
      'console_scripts': [
        'lbptop_calculate_parameters.py = antispoofing.lbptop.script.lbptop_calculate_parameters:main',
        'lbptop_cmphistmodels.py = antispoofing.lbptop.script.lbptop_cmphistmodels:main',
        'lbptop_ldatrain.py = antispoofing.lbptop.script.lbptop_ldatrain:main',
        'lbptop_make_scores.py = antispoofing.lbptop.script.lbptop_make_scores:main',
        'lbptop_mkhistmodel.py = antispoofing.lbptop.script.lbptop_mkhistmodel:main',
        'lbptop_result_analysis.py = antispoofing.lbptop.script.lbptop_result_analysis:main',
        'lbptop_svmtrain.py  = antispoofing.lbptop.script.lbptop_svmtrain:main',
        ],
      },

    classifiers = [
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],


)

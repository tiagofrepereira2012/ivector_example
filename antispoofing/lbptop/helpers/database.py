#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Fri Oct 05 12:08:00 CEST 2012

import pkg_resources

"""
Utilitary functions to access the database resources
"""

def get_available_databases():
  """
  Return the available databases
  """
  databases = []
  for entrypoint in pkg_resources.iter_entry_points('antispoofing.utils.db'):
    databases.append(entrypoint.name)

  return databases


def new_database(databaseName,args=None):
  """
  Return an instance of antispoofing.utils.db.databases.Database subclasses given a name 
  """

  for entrypoint in pkg_resources.iter_entry_points('antispoofing.utils.db'):
    if(databaseName == entrypoint.name):
      plugin = entrypoint.load()
      return plugin(args)

  return None


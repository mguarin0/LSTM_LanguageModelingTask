'''
@author: Michael Guarino
desc: this file contains all constants that are shared throughout the
      project
'''

import os

#project paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PRJ_DIR = '/'.join(SRC_DIR.split('/')[:-1])
DATA_DIR = '{}/{}'.format(PRJ_DIR, 'data')

#project constants
SEQUENCE_LENGTH = 30
EPOCHS = 50

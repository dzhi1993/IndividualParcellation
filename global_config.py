#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set global variables and paths

Created on 12/7/2023 at 10:57 AM
Author: dzhi
"""
import torch as pt
from pathlib import Path

# pytorch cuda global flag: True - cuda; False - cpu
pt.cuda.is_available = lambda : False
if pt.cuda.is_available():
    DEVICE = 'cuda:1'
else:
    DEVICE = 'cpu'
pt.set_default_device(DEVICE)
pt.set_default_dtype(pt.float32)

# Find model directory to save model fitting results
MODEL_DIR = '/data/tge/dzhi/Indiv_par/Models'
home = str(Path.home())
if not Path(MODEL_DIR).exists():
    MODEL_DIR = '/srv/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel/Models'
if not Path(MODEL_DIR).exists():
    MODEL_DIR = '/cifs/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel/Models'
if not Path(MODEL_DIR).exists():
    MODEL_DIR = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Models'
if not Path(MODEL_DIR).exists():
    MODEL_DIR = str(Path(home, 'diedrichsen_data/data/Cerebellum/ProbabilisticParcellationModel/Models'))
if not Path(MODEL_DIR).exists():
    raise (NameError('Could not find model_dir'))

BASE_DIR = '/data/tge/Tian/UKBB_full/imaging'
if not Path(BASE_DIR).exists():
    BASE_DIR = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(BASE_DIR).exists():
    BASE_DIR = '/cifs/diedrichsen/data/FunctionalFusion'
if not Path(BASE_DIR).exists():
    BASE_DIR = 'Y:\data\FunctionalFusion'
if not Path(BASE_DIR).exists():
    BASE_DIR = '/Users/callithrix/Documents/Projects/Functional_Fusion/'
if not Path(BASE_DIR).exists():
    BASE_DIR = '/Users/jdiedrichsen/Data/FunctionalFusion/'
if not Path(BASE_DIR).exists():
    BASE_DIR = str(Path(home, 'diedrichsen_data/data/FunctionalFusion'))
if not Path(BASE_DIR).exists():
    raise (NameError('Could not find base_dir'))

ATLAS_DIR = BASE_DIR + '/Atlases'
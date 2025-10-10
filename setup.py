# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Copyright (C) [2025] Instituto Nacional de Estadística
#
# Este archivo forma parte del proyecto statcodgen.
#
# Licenciado bajo la Licencia Pública de la Unión Europea (EUPL) v.1.2.
# Puede obtener una copia de la licencia en la raiz de este proyecto o en:
# https://eupl.eu/1.2/es/
#
# A menos que se indique lo contrario, este software se distribuye
# "TAL CUAL", SIN GARANTÍAS NI CONDICIONES DE NINGÚN TIPO.
# Consulte la licencia para conocer los términos específicos.
# ------------------------------------------------------------------------------
# Copyright (C) [2025] National Institute of Statistics
#
# This file is part of the statcodgen project.
#
# Licensed under the European Union Public License (EUPL) v.1.2.
# You can obtain a copy of the license at the root of this project or at:
# https://eupl.eu/1.2/es/
#
# Unless otherwise indicated, this software is distributed
# "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the license for specific terms.
# ------------------------------------------------------------------------------
"""
Created on Tue Jul  9 14:04:30 2024

@author: git.metodologia@ine.es
"""

import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
# you should update the version of your library as you add new functionalities
VERSION = '1.0.0'
# It must match the name of the folder where the code is located
PACKAGE_NAME = 'codauto'
AUTHOR = 'Unidad de clasificaciones INE'
AUTHOR_EMAIL = 'git.metodologia@ine.es'
URL = 'https://gitlab.ine.es/prod-estadistica/metodologia/clasificaciones/cod-auto'
LICENSE = 'EUPL European Public License'  # License type
DESCRIPTION = "Library for training an autoencoder"  # Short description
# Reference to the README file with a more detailed description
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding='utf-8')
LONG_DESC_TYPE = "text/markdown"
# Required packages for the library. They will be installed automatically if
# not already present
INSTALL_REQUIRES = [
    'pandas',
    'numpy',
    'pyreadstat',
    'utils',
    'fasttext-wheel',
    'openpyxl',
    'matplotlib',
    'scikit-learn',
    'nltk',
    'xlrd',
    'torch',
    'transformers'
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True
)

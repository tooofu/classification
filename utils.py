# -*- coding: utf-8 -*-
"""
Dec 2018 by Haeryong Jeong
1024.ding@gmail.com
https://www.github.com/tooofu/classification
"""

from __future__ import print_function

import os
import shutil


def mkdir(paths, with_rm=True):
    for path in paths:
        created = False
        if not os.path.exists(path):
            os.makedirs(path)
            created = True

        if not created and with_rm:
            shutil.rmtree(path)
            os.makedirs(path)

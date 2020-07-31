import numpy as np
import pandas as pd
import sys
import os
import time
import hashlib
import pickle
# import struct
# import functools
import itertools
import subprocess
import json
import zipfile

HOST = 'nikhef-g2'
DEST = '/project/detrd/markv/Holographic-Projector/tmp'
SSH = f'ssh -T {HOST}'

# rsync nikhef-g2:/project/detrd/markv/Holographic-Projector/tmp/out.json tmp_local/out.json
# rsync -a nikhef-g2:/project/detrd/markv/Holographic-Projector/tmp/ tmp_local/


def cp_file(src='out.zip', target=None):
    if target is None:
        target = src
    subprocess.run(f'rsync {HOST}:{DEST}/{src} {target}',
                   shell=True, check=True)


def read_file(fn='tmp/out.json'):
    content = f"""
{SSH} << EOF
source ~/.profile
cat {fn}
EOF
"""
    # raw = subprocess.check_output(content, shell=True)
    raw = subprocess.run(content, shell=True, check=True,
                         capture_output=True).stdout
    return raw.decode('utf-8').split('\n')

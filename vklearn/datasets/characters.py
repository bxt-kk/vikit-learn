from glob import glob
import os

CHARACTERS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'characters/')

CHARACTERS_DICT = {
    os.path.basename(os.path.splitext(path)[0]): path
    for path in glob(os.path.join(CHARACTERS_DIR, '*.txt'))}

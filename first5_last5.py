import os
from distutils.dir_util import copy_tree

first5_dir = "first5"
last5_dir = "last5"
os.makedirs(first5_dir, exist_ok=True)
os.makedirs(last5_dir, exist_ok=True)

content = os.listdir()
for truc in content:
    if os.path.isfile(truc) or truc[:7] != "subject":
        continue
    num = int(truc[-2:])
    if num <= 4:
        copy_tree(truc, first5_dir)
    else:
        copy_tree(truc, last5_dir)










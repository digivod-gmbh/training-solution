import os
import sys

python_path = os.path.split(sys.executable) # e.g. ('C:\\Python37', 'python.exe')

command = 'python {}{}Tools{}i18n{}pygettext.py -d labelme ../labelme/*'.format(python_path[0], os.sep, os.sep, os.sep)

os.system(command)

pot_file = '../locale/labelme.pot'
if os.path.isfile(pot_file):
    os.remove(pot_file)
os.rename('labelme.pot', pot_file)
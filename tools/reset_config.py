import os

# Run application to reset config
os.chdir('..')
os.system('pip install -e .')
os.system('python labelme/main.py --debug-mode --reset-config')
import os

# Run application during development
os.chdir('..')
os.system('pip install -e .')
os.system('python labelme/main.py --debug-mode')
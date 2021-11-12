import os
import subprocess
import sys

from setuptools import find_packages, setup
from torch_utils import get_version

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def install_package(package):
    output = subprocess.check_output(
        [sys.executable, '-m', 'pip', 'install', package])
    print(output.decode())


def load_package(requirements_path='requirements.txt'):
    requirements = []
    with open(requirements_path, 'r') as f:
        for each in f.readlines():
            requirements.append(each.strip())
    return requirements


setup(name='torch_utils',
      version=get_version(),
      description='(WIP)(Unofficial) PyTorch Utils',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/seefun/TorchUtils',
      author='See Fun',
      author_email='seefun@outlook.com',
      packages=find_packages(),
      install_requires=load_package('./requirements.txt'),
      include_package_data=True)

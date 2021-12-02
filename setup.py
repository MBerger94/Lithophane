from setuptools import setup

setup(
    name='lithostl',
    version='0.1',
    description='Python Class to create Lithophanes from Images and export them as STL-File.',
    author='MBerger94',
    packages=['lithostl'],  #same as name
    install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
)
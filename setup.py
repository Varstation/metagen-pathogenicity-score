from setuptools import setup

setup(name='metagen_pathogenicity_score',
      version='0.0.1',
      description='nbNMF is an easy-to-use Python library for robust dimensionality reduction with count data',
      url='https://github.com/Varstation/metagen-pathogenicity-score',
      author='Pedro Sebe',
      packages=['metagen_pathogenicity_score'],
      install_requires=["numpy","pandas","xarray","sklearn","ete3"],
      zip_safe=False)

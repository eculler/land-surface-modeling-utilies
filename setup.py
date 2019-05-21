from setuptools import setup, find_packages

setup(name='lsmutils',
      version='0.1',
      description='Pre- and Post-processing for Land Surface Models',
      url='http://github.com/eculler/land-surface-modeling-framework',
      author='Elsa Culler',
      author_email='eculler@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'geopandas',
          'lxml',
          'netCDF4',
          'pydap',
          'pyyaml',
          'requests'
      ],
      include_package_data=True,
      zip_safe=False)

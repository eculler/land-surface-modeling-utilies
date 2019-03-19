from setuptools import setup

setup(name='lsmutils',
      version='0.1',
      description='Pre- and Post-processing for Land Surface Models',
      url='http://github.com/eculler/land-surface-modeling-framework',
      author='Elsa Culler',
      author_email='eculler@gmail.com',
      license='MIT',
      packages=['lsmutils'],
      install_requires=[
          'geopandas',
          'netCDF4',
          'pyyaml'
      ],
      include_package_data=True,
      zip_safe=False)

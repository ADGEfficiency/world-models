from setuptools import setup, find_packages


setup(
    name='worldmodels',
    version='0.0.1',
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_dir={'worldmodels': 'worldmodels'},
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)

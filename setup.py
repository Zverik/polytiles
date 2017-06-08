from setuptools import setup
from os import path

try:
    import mapnik
except ImportError:
    raise ImportError('Mapnik library and its python bindings are required to install Polytiles.')

here = path.abspath(path.dirname(__file__))

setup(
    name='polytiles',
    version='1.0.0',
    author='Ilya Zverev',
    author_email='ilya@zverev.info',
    packages=['polytiles'],
    install_requires=[
        'mapnik',
        'shapely',
    ],
    url='https://github.com/zverik/polytiles',
    license='WTFPL',
    description='A script to render tiles for an area with mapnik',
    long_description=open(path.join(here, 'README.rst')).read(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Information Technology',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Utilities',
        'License :: Public Domain',
        'Programming Language :: Python :: 2',
    ],
    entry_points={
        'console_scripts': ['polytiles = polytiles:main']
    },
)

from setuptools import setup

from realkd import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='realkd',
    version=__version__,
    packages=['realkd'],
    url='https://github.com/marioboley/realkd.py',
    license='MIT',
    author='Mario Boley',
    author_email='mario.boley@gmail.com',
    description='Methods for knowledge discovery and interpretable machine learning.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    install_requires=['bitarray==1.5.3',
                      'sortedcontainers>=2.1.0',
                      'pandas>=0.25',
                      'numpy>=1.16.1',
                      'matplotlib',
                      'sortednp>=0.3.0'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ]
)

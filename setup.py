from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='realkd',
    version='0.0.1.dev1',
    packages=['realkd'],
    url='https://github.com/marioboley/realkd.py',
    license='MIT',
    author='Mario Boley',
    author_email='mario.boley@gmail.com',
    description='Python implementation of knowledge discovery methods',
    long_description=long_description,
    python_requires='>=3.6',
    install_requires=['sortedcontainers>=2.1.0'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ]
)

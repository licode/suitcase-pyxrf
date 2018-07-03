from setuptools import setup
import versioneer


with open('requirements.txt') as f:
    requirements = f.read().split()


setup(
    name='suitcase-pyxrf',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="BSD (3-clause)",
    install_requires=requirements,
    packages=['suitcase.pyxrf'],
    classifiers=[
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
)

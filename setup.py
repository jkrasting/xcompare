""" setup script """
import setuptools

exec(open("xcompare/version.py").read())

setuptools.setup(version=__version__)

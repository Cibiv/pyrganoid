from setuptools import setup, find_packages
from codecs import open
from os import path
from os.path import splitext
from glob import glob

from Cython.Distutils import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import cython_gsl


here = path.abspath(path.dirname(__file__))

with open("pyrganoid/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.strip().split("=")[1].strip(" '\"")
            break
    else:
        version = "0.0.1"

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

ext_modules = []
for pyxfile in glob("pyrganoid/models/*.pyx"):
    ext = Extension(
        splitext(pyxfile)[0].replace("/", "."),
        [pyxfile],
        libraries=cython_gsl.get_libraries(),
        library_dirs=[cython_gsl.get_library_dir()],
        cython_include_dirs=[cython_gsl.get_cython_include_dir()],
        # define_macros=[("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1"), ("linetrace", "True")],
    )

    ext_modules.append(ext)
for pxdfile in glob("pyrganoid/models/*.pxd"):
    ext = Extension(
        splitext(pxdfile)[0].replace("/", "."),
        [pxdfile],
        # define_macros=[("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1"), ("linetrace", "True")],
    )

    ext_modules.append(ext)

setup(
    name="pyrganoid",
    version=version,
    description="simulating organoid growth",
    long_description=long_description,
    author="Simon Haendeler",
    author_email="simon@haend.de",
    packages=find_packages(exclude=["examples", "contrib", "docs", "tests"]),
    # This are the versions I tested with, but if you know what you do you can also change these for compatibility reasons
    install_requires=["bokeh", "pandas", "scipy", "cython", "numpy"],
    extras_require={"SVG export": ["selenium"]},
    ext_modules=cythonize(
        ext_modules, compiler_directives={"binding": True, "linetrace": True}
    ),
    include_dirs=[numpy.get_include(), cython_gsl.get_include()],
    cmdclass={"build_ext": build_ext},
    tests_require=["coverage", "pytest"],
)

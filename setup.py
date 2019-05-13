# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import os
from pathlib import Path

from setuptools import setup, find_packages


def requirements(filename="requirements.txt"):
    """
    Return requirements list from a text file.

    Parameters
    ----------
    filename: str
        Name of requirement file.

    Returns
    -------
        str-list
    """
    try:
        require = list()
        f = open(filename, "rb")
        for line in f.read().decode("utf-8").split("\n"):
            line = line.strip()
            if "#" in line:
                line = line[:line.find("#")].strip()
            if line:
                require.append(line)
    except IOError:
        print("'{}' not found!".format(filename))
        require = list()

    return require


if __name__ == "__main__":
    # Your package name
    PKG_NAME = 'cvm'

    # Your GitHub user name
    GITHUB_USERNAME = 'TsumiNa'

    # Short description will be the description on PyPI
    SHORT_DESCRIPTION = 'CVM is a python package for solution limit calculation using **Cluster Variation Method** (CVM).'

    # Long description will be the body of content on PyPI page
    with open('README.md') as f:
        LONG_DESCRIPTION = f.read()

    with open('LICENSE') as f:
        LICENSE = f.read()

    # Version number, VERY IMPORTANT!
    VERSION = '0.3.0'

    # Author and Maintainer
    AUTHOR = 'TsumiNa'

    # Author email
    AUTHOR_EMAIL = 'liu.chang.1865@gmail.com'

    MAINTAINER = 'TsumiNa'
    MAINTAINER_EMAIL = 'liu.chang.1865@gmail.com'

    PACKAGES, INCLUDE_PACKAGE_DATA, PACKAGE_DATA, PY_MODULES = (
        None,
        None,
        None,
        None,
    )

    # It's a directory style package
    if os.path.exists(__file__[:-8] + PKG_NAME):
        # Include all sub packages in package directory
        PACKAGES = find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests', 'docs'])

        # Include everything in package directory
        INCLUDE_PACKAGE_DATA = True
        PACKAGE_DATA = {
            "": ["*.*"],
        }

    # It's a single script style package
    elif os.path.exists(__file__[:-8] + PKG_NAME + ".py"):
        PY_MODULES = [
            PKG_NAME,
        ]

    # Project Url
    GITHUB_URL = "https://github.com/{0}/{1}".format(GITHUB_USERNAME, PKG_NAME)
    # Use todays date as GitHub release tag
    RELEASE_TAG = 'v' + VERSION
    # Source code download url
    DOWNLOAD_URL = "https://github.com/{0}/{1}/archive/{2}.tar.gz".format(
        GITHUB_USERNAME, PKG_NAME, RELEASE_TAG)

    PLATFORMS = [
        "Windows",
        "MacOS",
        "Unix",
    ]

    CLASSIFIERS = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ]

    # Read requirements.txt, ignore comments
    INSTALL_REQUIRES = requirements()
    SETUP_REQUIRES = ['pytest-runner', 'ruamel.yaml']
    TESTS_REQUIRE = ['pytest']
    setup(python_requires='~=3.6',
          name=PKG_NAME,
          description=SHORT_DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          version=VERSION,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          packages=PACKAGES,
          include_package_data=INCLUDE_PACKAGE_DATA,
          package_data=PACKAGE_DATA,
          py_modules=PY_MODULES,
          url=GITHUB_URL,
          download_url=DOWNLOAD_URL,
          classifiers=CLASSIFIERS,
          platforms=PLATFORMS,
          license=LICENSE,
          setup_requires=SETUP_REQUIRES,
          install_requires=INSTALL_REQUIRES,
          tests_require=TESTS_REQUIRE)
"""
Appendix
--------
classifiers: https://pypi.python.org/pypi?%3Aaction=list_classifiers
"""

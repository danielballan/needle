# This downloads and install setuptools if it is not installed.
from setuptools import setup

setup_parameters = dict(
    name = "trackwire",
    description = "wire-tracking toolkit",
    author = "Daniel Allan",
    author_email = "dallan@pha.jhu.edu",
    url = "https://github.com/danielballan/trackwire",
    packages = ['trackwire']
)

setup(**setup_parameters)

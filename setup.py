# This downloads and install setuptools if it is not installed.
from setuptools import setup

setup_parameters = dict(
    name = "needle",
    version = "0.0.1",
    description = "Track the orientation of elongated objects.",
    author = "Daniel Allan",
    author_email = "daniel.b.allan@jhu.edu",
    url = "https://github.com/danielballan/needle",
    packages = ['needle']
)

setup(**setup_parameters)

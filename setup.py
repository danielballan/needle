# This downloads and install setuptools if it is not installed.
from setuptools import setup
import versioneer


versioneer.VCS = 'git'
versioneer.versionfile_source = 'needle/_version.py'
versioneer.versionfile_build = 'needle/_version.py'
versioneer.tag_prefix = 'v'
versioneer.parentdir_prefix = '.'

setup_parameters = dict(
    name = "needle",
    version = versioneer.get_version(),
    cmdclass = versioneer.get_cmdclass(),
    description = "Track the orientation of elongated objects.",
    author = "Daniel Allan",
    author_email = "daniel.b.allan@jhu.edu",
    url = "https://github.com/danielballan/needle",
    packages = ['needle']
)

setup(**setup_parameters)

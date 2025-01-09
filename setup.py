from pathlib import Path
from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

VERSION = '0.0.1'
DESCRIPTION = 'Donwload data from IH or PdE Opendats.'
PACKAGE_NAME = 'opendats'
AUTHOR = 'IHCantabria - AGV'
EMAIL = 'gonzalezva@unican.es'
GITHUB_URL = 'https://github.com/alexgonzvalle'

setup(
    name=PACKAGE_NAME,
    packages=[PACKAGE_NAME],
    version=VERSION,
    license='',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=GITHUB_URL,
    keywords=[],
    install_requires=[
        'requests', 'xarray', 'xmltodict', 'urllib3', 'wget', 'numpy', 'datetime', 'tqdm', 'copernicusmarine'
    ],
    package_data={'': ['']},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)

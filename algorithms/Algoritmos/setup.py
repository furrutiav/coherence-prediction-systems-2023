import setuptools
from kidespell import (__version__, __url__, __author__, __email__,
                          __license__, __bugtrack_url__)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


KEYWORDS = ['spelling', 'checker', 'kid', 'spanish']

setuptools.setup(
    name="KidESpell",
    version = __version__,
    author = __author__,
    author_email = __email__,
    description="python Spanish spell checking oriented to child, based on KidSpell",
    long_description=long_description,
    license = __license__,
    keywords = ' '.join(KEYWORDS),
    url = __url__,
    download_url = __url__,
    project_urls={
        "Bug Tracker": __bugtrack_url__,
    },
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={'kidespell': ['resources/*']},
    include_package_data = True,
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
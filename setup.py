import os
import setuptools

from model import __version__ as project_version

here = os.getcwd()
req_path = os.path.join(here, "requirements.txt")
readme_path = os.path.join(here, "README.md")

with open(req_path, "r", encoding="utf-8") as fp:
    requirements = fp.read().splitlines()

with open(readme_path, "r", encoding="utf-8") as fp:
    readme = fp.read()
    

setuptools.setup(
    name="nlos-viewpoints",
    version=project_version,
    author="Salvador Rodriguez Sanz",
    author_email="salvador.rodriguez@unizar.es",
    url="https://github.com/srodsanz/nlos-viewpoints",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.8",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=requirements
)

import os

from setuptools import setup, find_packages
from pip._internal.network.session import PipSession
from pip._internal.req import parse_requirements


here = os.path.abspath(os.path.dirname(__file__))
session = PipSession()
process_requirements = parse_requirements(os.path.join(here, "requirements.txt"), session=session)
complete_reqs = list(process_requirements)

requirements = [str(ir.requirement) for ir in complete_reqs]

setup(
    name="nlos-viewpoints",
    version="0.1.0-SNAPSHOT",
    description="Utility library for running NLOS based on NERF",
    author="srodsanz@github.com",
    install_requires=requirements,
    packages=find_packages()
)
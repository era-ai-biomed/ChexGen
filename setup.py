from pathlib import Path

from setuptools import find_packages, setup


def read_requirements():
    requirements = Path(__file__).with_name("requirements.txt")
    return [
        line.strip()
        for line in requirements.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


setup(
    name='radiffuser',
    version='0.0.1',
    description='ChexGen chest radiography generation utilities',
    packages=find_packages(),
    install_requires=read_requirements(),
)

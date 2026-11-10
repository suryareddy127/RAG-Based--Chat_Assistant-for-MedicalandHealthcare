from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="RAG Medical assistatnt",
    description="RAG Medical healthcare services assistant",
    version="1.0",
    author="Surya_reddy",
    packages=find_packages(),
    install_requires = requirements,
)
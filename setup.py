"""
Setup script for Atlas.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="atlas-llm",
    version="0.1.0",
    author="Atlas Team",
    description="A from-scratch language model implementation with GGUF export",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juliuspleunes4/Atlas",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
)

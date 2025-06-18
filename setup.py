#!/usr/bin/env python3
"""
Setup script for Notion + Claude + Codebase Integration
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="notion-claude-codebase",
    version="1.0.0",
    author="Tech Bros",
    author_email="contact@techbros.dev",
    description="A LlamaIndex-based application that integrates Notion, Claude AI, and codebase analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/techbros/notion-claude-codebase",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ]
    },
    entry_points={
        "console_scripts": [
            "notion-claude-codebase=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 
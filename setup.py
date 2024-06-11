"""Setup script for Arabic NLP for Real Estate Documents."""

from pathlib import Path

from setuptools import find_packages, setup

this_dir = Path(__file__).parent

# Load long description from README
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

# Load requirements
requirements = [
    line.strip()
    for line in (this_dir / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="arabic-nlp-realestate",
    version="1.0.0",
    description=(
        "Production-grade Arabic NLP toolkit for processing real estate documents. "
        "Supports text classification, named entity recognition, and summarization."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="balaga raghuram",
    author_email="team@example.com",
    url="https://github.com/your-org/Arabic-NLP-for-Real-Estate-Documents",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Arabic",
    ],
    keywords="arabic nlp real-estate classification ner summarization",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "data"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.0",
        ],
        "ner": ["seqeval"],
        "summarization": ["rouge-score"],
    },
    entry_points={
        "console_scripts": [
            "arabic-nlp-realestate=main:main",
        ],
    },
    project_urls={
        "Source": "https://github.com/your-org/Arabic-NLP-for-Real-Estate-Documents",
        "Bug Tracker": "https://github.com/your-org/Arabic-NLP-for-Real-Estate-Documents/issues",
    },
)

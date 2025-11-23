"""Setup script for ML Model Deployment Toolkit."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ml-deployment-toolkit",
    version="1.0.0",
    author="UNC Google Developer Student Club",
    author_email="gdsc@unc.edu",
    description="Production-ready toolkit for deploying ML models to AWS Lambda, GCP Cloud Run, and Vercel",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UNC-GDSC/ML-Model-Deployment-Toolkit",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "pydantic>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "boto3>=1.28.0",
        "google-cloud-storage>=2.10.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "click>=8.1.0",
        "rich>=13.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "tensorflow": ["tensorflow>=2.14.0"],
        "pytorch": ["torch>=2.1.0"],
        "all": [
            "tensorflow>=2.14.0",
            "torch>=2.1.0",
            "onnxruntime>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-deploy=cli.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)

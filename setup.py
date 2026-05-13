from setuptools import setup, find_packages

setup(
    name="research-utils",
    version="0.1.0",
    description="Shared utility functions for PhD studies",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.5.2",
        "natsort>=8.4.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "pingouin>=0.5.3",
        "scikit-posthocs>=0.8.0",
        "scipy>=1.8.1",
        "seaborn>=0.13.2",
        "scikit-learn>=1.3.2",
        "spm1d>=0.4.20",
        "statsmodels>=0.14.0",
    ],
)

"""Setup script for NeuroFusionXAI package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neurofusionxai",
    version="1.0.0",
    author="NeuroFusionXAI Authors",
    author_email="author@example.com",
    description="Privacy-Preserving Cross-Modality Explainable Fusion Framework for Early Neurodegenerative Disease Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/NeuroFusionXAI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neurofusionxai-train=scripts.train:main",
            "neurofusionxai-evaluate=scripts.evaluate:main",
            "neurofusionxai-inference=scripts.inference:main",
        ],
    },
)

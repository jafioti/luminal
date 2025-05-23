from setuptools import setup, find_packages

setup(
    name="pyluminal",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
    ],
    author="Joe Fioti",
    author_email="jafioti@gmail.com",
    description="Deep Learning at the speed of light.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jafioti/luminal",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

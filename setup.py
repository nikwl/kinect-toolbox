import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="ktb", # Replace with your own username
    version="0.0.1",
    author="Nikolas lamb",
    author_email="nil518@lehigh.edu",
    description="A python wrapper for the kinect built on pylibfreenect2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nikwl/kinect-toolbox",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
		"Topic :: Multimedia :: Video :: Capture"
    ],
    python_requires='>=3.6',
)
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="audio-reactive-led-strip", # Replace with your own username
    version="1.0.0",
    author="James Wilson",
    author_email="james@drakeapps.com",
    description="Audio Reactive LED Strip Controller",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drakeapps/audio-reactive-led-strip",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
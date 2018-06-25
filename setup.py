import setuptools

with open("Readme.md","r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="icenumerics",
    version="0.1.0",
    author="Antonio Ortiz-Ambriz",
    author_email="aortiza@gmail.com",
    description="Simulations of Colloidal Ice",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aortiza/icenumerics",
    packages=setuptools.find_packages(),
    install_requires = ['numpy','scipy','pandas','matplotlib','ureg','jsonpickle','magcolloids']
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Windows, MacOS X(intel), Linux",
    ),
)
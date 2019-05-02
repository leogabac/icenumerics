# Ice Numerics

This is a numerics library for simulations and data processing of colloidal ice. It is very experimental, not very extensivelly tested and unfortunatelly not richly documented. I'll try to improve that over time.

## Getting Started

You can see a walktrhough [here](https://aortiza.github.io/icenumerics/IceNumericsUsage.html).

### Prerequisites
You will need numpy 1.11.3, scipy 0.18.1 and matplotlib 2.0.0. Many things are built to use Jupyter. You might be able to do them in matplotlib, but I haven't tested it. 

### Installing

There is no pip instaler or anything like that (yet). You need to download the files from the git repository and place them wherever you want to run your script.

The library runs many things in Python, but the brownian dynamics is done in a modified version of LAMMPS. I included LAMMPS compiled binaries for serial execution in Windows and Mac. If these don't work, or you want to use another system (for example, most linux distributions) you will need to compile the LAMMPS. 

## Contributing

I don't know. Contact me probably.

## Authors

* **Antonio Ortiz-Ambriz** 

## License

This project is licensed under the GPL License. In fact, most of the code is licenced under MIT licence, but LAMMPS is GPL. 

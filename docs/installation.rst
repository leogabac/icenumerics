=============
Installation
=============

Installing a released version using pip.
=========================================
The easiest way to install the package is to write in the command line::

	pip install icenumerics
	
If this works, then you should run a sample simulation, as described in the  :doc:`IceNumericsUsage`. If the command line shows an output like the following, your installation was successful.

.. image:: lammps_output.png

If you get an error, it means that your system can't run the lammps binaries included in the simulation.

.. todo:: write specific instructions lammps compilation and how to include the binaries on the package.

Installing the latest version from github.
==========================================

To clone the latest version from github write::
	
	git clone --recurse-submodules https://github.com/aortiza/icenumerics.git

This will download both the icenumerics package and the magcolloids subpackage. 
	
	
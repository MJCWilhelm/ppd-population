This repository contains the scripts to run a large number of protoplanetary disk models. The disk models use the VADER code (Krumholz & Forbes 2015) through an interface to the AMUSE framework.

This repository contains the following scripts:
- src/FRIED\_interp.py - an interpolator of the FRIED grid (Haworth et al. 2018).
- src/disk\_class.py - a class for a protoplanetary disk, used by other script.
- src/ppd\_population.py - a class that manages a large number of instances of the disk class.
- src/ppd\_population\_async.py - another class that manages a large number of instances of the disk class. This is a refactored and improved version of src/ppd\_population.py, and the current development version.

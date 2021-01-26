import numpy as np
import matplotlib.pyplot as plt

import pickle

from amuse.units import units, constants
from amuse.datamodel import Particles
from amuse.community.vader.interface import Vader

import disk_class
import FRIED_interp


viscous = Vader(mode='pedisk', redirection='none')

viscous.initialize_keplerian_grid(330, False,
                0.01|units.AU, 3000.|units.AU, 0.08 | units.MSun)

viscous.parameters.alpha = 1e-4
viscous.parameters.post_timestep_function = True
viscous.parameters.maximum_tolerated_change = 1E99
viscous.parameters.number_of_user_parameters = 7
viscous.parameters.inner_pressure_boundary_torque = 0. | units.g*units.cm**2./units.s**2.

            # Outer density in g/cm^2
viscous.set_parameter(2, 1E-12)
            # Mean particle mass in g
viscous.set_parameter(4, (2.33*1.008*constants.u).value_in(units.g))


disk = disk_class.Disk(200.*0.08**0.45|units.AU, 0.24*0.08**0.73|units.MSun, 
    0.08 | units.MSun, viscous.grid, 1e-4)
disk.outer_photoevap_rate = 4e-6 | units.MSun/units.yr
disk.viscous = viscous

disk.evolve_disk_for(10. | units.kyr)

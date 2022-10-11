import numpy as np
import matplotlib.pyplot as plt

import time

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles
from amuse.ic.plummer import new_plummer_model

from ppd_population_async import PPDPopulationAsync
from ppd_population import PPD_population


if __name__ == '__main__':

    start = time.time()
    ppd_pool  = PPD_population()
    end = time.time()
    print ("Starting pool with {a} workers took {b} s".format(a=len(ppd_pool.codes),
        b=end-start))

    start = time.time()
    ppd_async = PPDPopulationAsync(vader_mode='pedisk')
    end = time.time()
    print ("Starting async took {b} s".format(b=end-start))

    M = np.array([0.08, 0.3, 0.5, 1., 16.])|units.MSun

    converter = nbody_system.nbody_to_si(M.sum(), 1.|units.pc)
    stars = new_plummer_model(len(M), converter)
    stars.mass = M
    stars[-1].fuv_luminosity = 1e0 | units.LSun

    start = time.time()
    ppd_pool.add_star_particles(stars.copy())
    end = time.time()
    print ("Adding {a} stars ({b} disks) to pool took {c} s".format(
        a=len(stars), b=np.sum(M<1.9|units.MSun), c=end-start))

    start = time.time()
    ppd_async.add_star_particles(stars.copy())
    end = time.time()
    print ("Adding {a} stars ({b} disks) to async took {c} s".format(
        a=len(stars), b=np.sum(M<1.9|units.MSun), c=end-start))

    print (ppd_pool.star_particles.disk_gas_mass.value_in(units.MSun))
    print (ppd_async.star_particles.disk_gas_mass.value_in(units.MSun))

    ppd_pool.evolve_model(1.|units.kyr)
    ppd_async.evolve_model(1.|units.kyr)

    #plt.loglog(ppd_pool.disks[0].grid.r.value_in(units.au),
    #    ppd_pool.disks[0].grid.column_density.value_in(units.g/units.cm**2))
    #plt.show()

    print (ppd_pool.star_particles.accreted_mass.value_in(units.MSun),
        ppd_pool.star_particles.ipe_mass_loss.value_in(units.MSun),
        ppd_pool.star_particles.epe_mass_loss.value_in(units.MSun))
    print (ppd_async.star_particles.accreted_mass.value_in(units.MSun),
        ppd_async.star_particles.ipe_mass_loss.value_in(units.MSun),
        ppd_async.star_particles.epe_mass_loss.value_in(units.MSun))

    print (ppd_pool.star_particles.disk_gas_mass.value_in(units.MSun))
    print (ppd_async.star_particles.disk_gas_mass.value_in(units.MSun))

    ppd_pool.stop()
    ppd_async.stop()

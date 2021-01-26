import numpy as np
import matplotlib.pyplot as plt

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles
from amuse.community.vader.interface import Vader
from amuse.ic.brokenimf import MultiplePartIMF
from amuse.ic.plummer import new_plummer_model

import time

from ppd_population import PPD_population


kroupa_imf = MultiplePartIMF(mass_boundaries=[0.08, 0.5, 1.9]|units.MSun,
    alphas=[-1.3, -2.3])


def test_single (particles, N_workers, alpha, N_cells):

    ppds = PPD_population(alpha=alpha, number_of_cells=N_cells,
        number_of_workers=N_workers)

    ppds.add_star_particles(particles)


    start = time.time()

    ppds.evolve_model(1. | units.kyr)

    end = time.time()

    print ("{a} disks, {b} workers, {c} cells, alpha={d}".format(
        a=len(ppds.disked_stars), b=N_workers, c=N_cells, d=alpha))
    print ("Ran in {a} s".format(a=end-start))

    ppds.stop()

    return end-start


def test_no_workers (particles, N_workers, alpha=1e-3, N_cells=330):

    timer = np.zeros(len(N_workers))

    for i in range(len(N_workers)):
        timer[i] = test_single(particles, N_workers[i], alpha, N_cells)

    return timer


def test_no_cells (particles, N_cells, alpha=1e-3, N_workers=4):

    timer = np.zeros(len(N_cells))

    for i in range(len(N_cells)):
        timer[i] = test_single(particles, N_workers, alpha, N_cells[i])

    return timer


def test_alpha (particles, alpha, N_cells=330, N_workers=4):

    timer = np.zeros(len(alpha))

    for i in range(len(alpha)):
        timer[i] = test_single(particles, N_workers, alpha[i], N_cells)

    return timer


def make_ics (N, R):

    masses = kroupa_imf.next_mass(N)

    converter = nbody_system.nbody_to_si(masses.sum(), R)

    stars = new_plummer_model(N, converter)

    stars.mass = masses

    stars.add_particles(Particles(1,
        mass = 30. | units.MSun,
        x = 0. | units.pc,
        y = 0. | units.pc,
        z = 0. | units.pc,
        vx = 0. | units.kms,
        vy = 0. | units.kms,
        vz = 0. | units.kms,
    ))

    stars.fuv_luminosity = 0. | units.LSun
    stars[-1].fuv_luminosity = 1e3 | units.LSun

    return stars


if __name__ == '__main__':

    np.random.seed(917492387)

    N_stars = 30
    R_stars = 1. | units.pc

    stars = make_ics (N_stars, R_stars)


    N_workers = [1, 2, 3, 4, 5, 6, 7, 8]

    timer_workers = test_no_workers (stars, N_workers)

    plt.plot(N_workers, timer_workers)
    plt.xlabel('#Workers')
    plt.ylabel('s')


    N_cells = [50, 100, 200, 300, 500]

    timer_cells = test_no_cells (stars, N_cells)

    plt.figure()
    plt.plot(N_cells, timer_cells)
    plt.xlabel('#Cells')
    plt.ylabel('s')


    alpha = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    timer_alpha = test_alpha (stars, alpha)

    plt.figure()
    plt.plot(alpha, timer_alpha)
    plt.xlabel('$\\alpha$')
    plt.ylabel('s')
    plt.xscale('log')

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
#import yt

from amuse.io import read_set_from_file
from amuse.units import units

import time


def read_sets (indices, label='', filepath='./'):

    particles = {}

    for i in range(len(indices)):

        particles_temp = read_set_from_file(
            filepath+'viscous_particles_{b}i{a:05}.hdf5'.format(
            a=indices[i], b=label), 'hdf5')

        for j in range(len(particles_temp)):

            if particles_temp[j].disk_dispersed:
                continue

            key = particles_temp[j].key

            if key not in particles.keys():
                particles[key] = {}
                particles[key]['disk_radius'] = [] | units.AU
                particles[key]['disk_gas_mass'] = [] | units.MSun
                particles[key]['time'] = [] | units.Myr
                particles[key]['truncation_mass_loss'] = [] | units.MSun
                particles[key]['stellar_mass'] = [] | units.MSun
                particles[key]['disk_dispersed'] = []

            particles[key]['disk_radius'].append(particles_temp[j].disk_radius)
            particles[key]['disk_gas_mass'].append(particles_temp[j].disk_gas_mass)
            particles[key]['time'].append(particles_temp.get_timestamp())
            particles[key]['truncation_mass_loss'].append(particles_temp[j].truncation_mass_loss)
            particles[key]['stellar_mass'].append(particles_temp[j].mass)
            particles[key]['disk_dispersed'].append(particles_temp[j].disk_dispersed)

    return particles



if __name__ == '__main__':

    #filepath = '/home/martijn/scripts/torch_test4/data/'
    filepath = '/data2/wilhelm/sim_archive/gmc_star_formation/ColGMC_concha2021_test2/'

    start = time.time()
    disks = read_sets(np.arange(70,382), 
        filepath=filepath)
    end = time.time()
    print (end-start, 's', flush=True)

    N = len(disks)

    for key in disks:
        plt.plot(disks[key]['time'].value_in(units.Myr), 
            disks[key]['disk_radius'].value_in(units.AU), c='k', alpha=N**-0.5)

    plt.yscale('log')

    plt.xlabel('t [Myr]')
    plt.ylabel('R$_d$ [au]')

    plt.figure()

    for key in disks:
        plt.plot(disks[key]['time'].value_in(units.Myr), 
            disks[key]['disk_gas_mass'].value_in(units.MSun), c='k', alpha=N**-0.5)
        disk_gas_mass = disks[key]['disk_gas_mass'][ disks[key]['disk_dispersed'] ]

        N_zero = np.sum( (disk_gas_mass[:-1] - disk_gas_mass[1:]).value_in(units.MSun) > 0.)
        if N_zero > 0:
            print (N_zero, flush=True)
        #Mtrunc = disks[key]['truncation_mass_loss'][-1].value_in(units.MSun)
        #if Mtrunc > 1e-13:
        #    print (Mtrunc, 'MSun', flush=True)

    plt.yscale('log')

    plt.xlabel('t [Myr]')
    plt.ylabel('M$_d$ [M$_\\odot$]')



    for key in disks:
        if disks[key]['stellar_mass'][0] > 7. | units.MSun:
            print (disks[key]['time'][0].value_in(units.Myr), 'Myr', disks[key]['stellar_mass'][0].value_in(units.MSun), flush=True)

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
#import yt

from amuse.io import read_set_from_file
from amuse.units import units


def read_sets (indices, label='', filepath='./'):

    particles = {}

    for i in range(len(indices)):

        particles_temp = read_set_from_file(
            filepath+'viscous_particles_{b}i{a:05}.hdf5'.format(
            a=indices[i], b=label), 'hdf5')

        for j in range(len(particles_temp)):

            key = particles_temp[j].key

            if key not in particles.keys():
                particles[key] = {}
                particles[key]['disk_radius'] = [] | units.AU
                particles[key]['disk_gas_mass'] = [] | units.MSun
                particles[key]['time'] = [] | units.Myr
                particles[key]['truncation_mass_loss'] = [] | units.MSun
                particles[key]['stellar_mass'] = [] | units.MSun

            particles[key]['disk_radius'].append(particles_temp[j].disk_radius)
            particles[key]['disk_gas_mass'].append(particles_temp[j].disk_gas_mass)
            particles[key]['time'].append(particles_temp.get_timestamp())
            particles[key]['truncation_mass_loss'].append(particles_temp[j].truncation_mass_loss)
            particles[key]['stellar_mass'].append(particles_temp[j].mass)

    return particles



if __name__ == '__main__':

    filepath = '/home/martijn/scripts/torch_test4/data/'

    disks = read_sets(np.arange(152,261), label='plt_', 
        filepath=filepath)

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
        Mtrunc = disks[key]['truncation_mass_loss'][-1].value_in(units.MSun)
        if Mtrunc > 1e-13:
            print (Mtrunc, 'MSun', flush=True)

    plt.yscale('log')

    plt.xlabel('t [Myr]')
    plt.ylabel('M$_d$ [M$_\\odot$]')



    for key in disks:
        if disks[key]['stellar_mass'][0] > 7. | units.MSun:
            print (disks[key]['time'][0].value_in(units.Myr), 'Myr', disks[key]['stellar_mass'][0].value_in(units.MSun), flush=True)

    plt.show()

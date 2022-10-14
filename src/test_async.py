import numpy as np
import matplotlib.pyplot as plt

import time

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles
from amuse.ic.plummer import new_plummer_model
from amuse.community.ph4.interface import ph4
from amuse.community.vader.interface import Vader
from amuse.ic.brokenimf import MultiplePartIMF

from ppd_population_async import PPDPopulationAsync
from ppd_population import PPD_population
from test_pool import FUV_luminosity_from_mass


def compare_to_pooled (M):

    start = time.time()
    ppd_pool  = PPD_population()
    end = time.time()
    print ("Starting pool with {a} workers took {b} s".format(a=len(ppd_pool.codes),
        b=end-start))

    start = time.time()
    ppd_async = PPDPopulationAsync(vader_mode='pedisk')
    end = time.time()
    print ("Starting async took {b} s".format(b=end-start))

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

    #print (ppd_pool.star_particles.disk_gas_mass.value_in(units.MSun))
    #print (ppd_async.star_particles.disk_gas_mass.value_in(units.MSun))

    start = time.time()
    ppd_pool.evolve_model(10.|units.kyr)
    end = time.time()
    print ("Running {a} pooled disks took {b} s".format(a=np.sum(M<1.9|units.MSun),
        b=end-start))

    start = time.time()
    ppd_async.evolve_model(10.|units.kyr)
    end = time.time()
    print ("Running {a} async disks took {b} s".format(a=np.sum(M<1.9|units.MSun),
        b=end-start))

    print ((ppd_pool.star_particles.disk_gas_mass - ppd_async.star_particles.disk_gas_mass).value_in(units.MSun))
    print ((ppd_pool.star_particles.ipe_mass_loss - ppd_async.star_particles.ipe_mass_loss).value_in(units.MSun))
    print ((ppd_pool.star_particles.epe_mass_loss - ppd_async.star_particles.epe_mass_loss).value_in(units.MSun))
    print ((ppd_pool.star_particles.accreted_mass - ppd_async.star_particles.accreted_mass).value_in(units.MSun))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('symlog')

    for i in range(len(M)):
        if M[i] < 1.9 | units.MSun:
            r = ppd_pool.disks[i].grid.r.value_in(units.au)
            coldens_pool = ppd_pool.disks[i].grid.column_density
            coldens_async = ppd_async.disks[i].grid.column_density
            print (np.sum(coldens_async == coldens_pool))

            ax.plot(r, abs(coldens_async - coldens_pool)/coldens_pool)

    ppd_pool.stop()
    ppd_async.stop()


def test_scaling (N):

    kroupa_imf = MultiplePartIMF(mass_boundaries=[0.08, 0.5, 1.9]|units.MSun,
        alphas=[-1.3, -2.3])

    timer_add = np.zeros(len(N))
    timer_run = np.zeros(len(N))

    for i in range(len(N)):

        M = kroupa_imf.next_mass(N[i])
        M[-1] = 16. | units.MSun

        converter = nbody_system.nbody_to_si(M.sum(), 1.|units.pc)
        stars = new_plummer_model(len(M), converter)
        stars.mass = M
        stars[-1].fuv_luminosity = 1e0 | units.LSun

        #start = time.time()
        ppd_async = PPDPopulationAsync(vader_mode='pedisk')
        #end = time.time()
        #print ("Starting async took {b} s".format(b=end-start))

        start = time.time()
        ppd_async.add_star_particles(stars.copy())
        end = time.time()
        timer_add[i] = end-start
        print ("Adding {a} stars ({b} disks) to async took {c} s".format(
            a=len(stars), b=np.sum(M<1.9|units.MSun), c=end-start))

        start = time.time()
        ppd_async.evolve_model(1.|units.kyr)
        end = time.time()
        timer_run[i] = end-start
        print ("Running {a} async disks took {b} s".format(
            a=np.sum(M<1.9|units.MSun), b=end-start))


    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(N, timer_add, label='Add')
    ax.plot(N, timer_run, label='Run')

    ax.plot(N, timer_add[-1] * (N/N[-1]), c='k')
    ax.plot(N, timer_run[-1] * (N/N[-1]), c='k')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('N')
    ax.set_ylabel('Time [s]')

    ax.legend()


def test_population (N=10):

    from amuse.community.ph4.interface import ph4
    from amuse.community.huayno.interface import Huayno
    from amuse.units import nbody_system, units, constants
    from amuse.datamodel import Particles
    from amuse.ic.brokenimf import MultiplePartIMF
    from amuse.ic.plummer import new_plummer_model

    kroupa_imf = MultiplePartIMF(mass_boundaries=[0.08, 0.5, 1.9]|units.MSun,
        alphas=[-1.3, -2.3])

    #mass = kroupa_imf.next_mass(N)
    mass = np.ones(N) * 0.08 | units.MSun


    converter = nbody_system.nbody_to_si(mass.sum(), 1.|units.pc)
    stars = new_plummer_model(N, converter)
    stars.mass = mass

    stars.add_particles(Particles(1,
        mass=30.|units.MSun,
        x=0.|units.pc,
        y=0.|units.pc,
        z=0.|units.pc,
        vx=0.|units.kms,
        vy=0.|units.kms,
        vz=0.|units.kms,
        fuv_luminosity=FUV_luminosity_from_mass(100.|units.MSun)
    ))


    ppd_code = PPDPopulationAsync(max_frac=0.1)
    ppd_code.add_star_particles(stars)

    channel_ppd_to_star, channel_star_to_ppd = ppd_code.setup_self_channels(stars)

    dt = 1. | units.kyr
    t_end = 0.1 | units.Myr
    time = 0. | units.kyr

    counter = 0


    n = int(t_end/dt)

    disk_mass = np.zeros((n,N+1))
    disk_radius = np.zeros((n,N+1))


    for i in range(n):

        print (time.value_in(units.Myr), t_end.value_in(units.Myr),
            ppd_code.disked_stars.disk_gas_mass.min().value_in(units.MSun))

        ppd_code.evolve_model( time + dt )
        channel_ppd_to_star.copy()

        disk_mass[i] = ppd_code.star_particles.disk_gas_mass.value_in(
            units.MSun)
        disk_radius[i] = ppd_code.star_particles.disk_radius.value_in(
            units.AU)

        #counter += 1
        time += dt

    plt.loglog(disk_radius, disk_mass)

    plt.xscale('log')
    plt.yscale('log')


def test_truncation ():

    from amuse.community.ph4.interface import ph4
    from amuse.units import nbody_system, units, constants
    from amuse.datamodel import Particles

    stars = Particles(2, 
        mass = [0.08, 1.5] | units.MSun,
        x = [-1., 1.] | units.parsec,
        y = [-200., 200.] | units.AU,
        z = [0., 0.] | units.parsec,
        vx = [100., -100.] | units.parsec/units.Myr,
        vy = [0., 0.] | units.parsec/units.Myr,
        vz = [0., 0.] | units.parsec/units.Myr
    )
    stars.stellar_mass = stars.mass

    ppd_code = PPDPopulationAsync()
    ppd_code.add_star_particles(stars)


    converter = nbody_system.nbody_to_si(1.|units.MSun, 1.|units.parsec)
    gravity = ph4(converter)

    channel_ppd_to_star, channel_star_to_ppd = ppd_code.setup_self_channels(stars)
    channel_grav_to_star, channel_star_to_grav = ppd_code.setup_gravity_channels(
        stars, gravity)

    channels = {
        'ppd_to_star': channel_ppd_to_star,
        'star_to_ppd': channel_star_to_ppd,
        'grav_to_star': channel_grav_to_star,
        'star_to_grav': channel_star_to_grav
    }

    gravity.particles.add_particles(stars)


    collision_detector = gravity.stopping_conditions.collision_detection
    collision_detector.enable()
    ppd_code.collision_detector = collision_detector


    dt = 1. | units.kyr
    t_end = 10. | units.kyr
    time = 0. | units.kyr

    counter = 0

    while time < t_end:

        print ("Evolving until {a} kyr".format(a=(time+dt).value_in(units.kyr)))

        gravity.evolve_model( time + dt/2. )
        channels['grav_to_star'].copy()

        while collision_detector.is_set():

            print ("Encounter at t={a} Myr".format(
                a=gravity.model_time.value_in(units.Myr)))

            channels['star_to_ppd'].copy()
            ppd_code.resolve_encounters()
            channels['ppd_to_star'].copy()

            channels['star_to_grav'].copy()
            gravity.evolve_model( time + dt/2. )
            channels['grav_to_star'].copy()


        channels['star_to_ppd'].copy()
        ppd_code.evolve_model( time + dt )
        channels['ppd_to_star'].copy()


        channels['star_to_grav'].copy()
        gravity.evolve_model( time + dt )
        channels['grav_to_star'].copy()

        while collision_detector.is_set():

            print ("Encounter at t={a} Myr".format(
                a=gravity.model_time.value_in(units.Myr)))

            channels['star_to_ppd'].copy()
            ppd_code.resolve_encounters()
            channels['ppd_to_star'].copy()

            channels['star_to_grav'].copy()
            gravity.evolve_model( time + dt )
            channels['grav_to_star'].copy()

        counter += 1
        time += dt

    r_enc = (stars[1].position-stars[0].position).length().value_in(units.au)
    print ("Encounter distance: {a} au".format(a=r_enc))
    print ("Truncation radii: {a}, {b} au".format(
        a=r_enc/3. * (stars[0].mass/stars[1].mass)**0.32,
        b=r_enc/3. * (stars[1].mass/stars[0].mass)**0.32))

    print (ppd_code.star_particles.truncation_mass_loss.value_in(units.MSun))
    print (ppd_code.star_particles.disk_radius.value_in(units.AU))

    plt.loglog(ppd_code.disks[0].grid.r.value_in(units.au),
        ppd_code.disks[0].grid.column_density.value_in(units.g/units.cm**2))
    plt.loglog(ppd_code.disks[1].grid.r.value_in(units.au),
        ppd_code.disks[1].grid.column_density.value_in(units.g/units.cm**2))


def test_restartibility (N=10):

    from amuse.community.ph4.interface import ph4
    from amuse.community.huayno.interface import Huayno
    from amuse.units import nbody_system, units, constants
    from amuse.datamodel import Particles
    from amuse.ic.brokenimf import MultiplePartIMF
    from amuse.ic.plummer import new_plummer_model
    from ppd_population_async import restart_population

    kroupa_imf = MultiplePartIMF(mass_boundaries=[0.08, 0.5, 1.9]|units.MSun,
        alphas=[-1.3, -2.3])

    mass = kroupa_imf.next_mass(N)


    converter = nbody_system.nbody_to_si(mass.sum(), 1.|units.pc)
    stars = new_plummer_model(N, converter)
    stars.mass = mass
    stars.fuv_luminosity = 0. | units.LSun

    stars.add_particles(Particles(1,
        mass=30.|units.MSun,
        x=0.|units.pc,
        y=0.|units.pc,
        z=0.|units.pc,
        vx=0.|units.kms,
        vy=0.|units.kms,
        vz=0.|units.kms,
        fuv_luminosity=FUV_luminosity_from_mass(30.|units.MSun)
    ))


    ppds1 = PPDPopulationAsync()
    ppds1.add_star_particles(stars)

    ppds2 = PPDPopulationAsync()
    ppds2.add_star_particles(stars)


    for i in range(10):
        ppds1.evolve_model((i+1)|units.kyr)

        ppds2.evolve_model((i+1)|units.kyr)

    print (ppds1.model_time.value_in(units.Myr))
    for disk in ppds1.disks:
        if disk is not None:
            print (disk.viscous.model_time.value_in(units.Myr))
    print (ppds2.model_time.value_in(units.Myr))
    for disk in ppds2.disks:
        if disk is not None:
            print (disk.viscous.model_time.value_in(units.Myr))

    #ppds2.write_out()
    ppds2.write_particles()
    ppds2.write_grids()

    ppds3 = restart_population('./', 0, ppds2._params['alpha'],
        ppds2._params['mu'], ppds2._params['n_cells'], ppds2._params['r_min'],
        ppds2._params['r_max'], extra_attributes=['fuv_luminosity'],
        fried_folder='../data/')


    for i in range(10,20):
        ppds1.evolve_model((i+1)|units.kyr)

        ppds3.evolve_model((i+1)|units.kyr)


    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    fig2 = plt.figure()
    ax3 = fig2.add_subplot(211)
    ax4 = fig2.add_subplot(212)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax1.set_xlabel('R [AU]')
    ax1.set_ylabel('d$\\Sigma$')
    ax2.set_xlabel('R [AU]')
    ax2.set_ylabel('dP')

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax4.set_xscale('log')
    ax4.set_yscale('log')

    ax3.set_xlabel('R [AU]')
    ax3.set_ylabel('$\\Sigma$ [g cm$^{-2}$]]')
    ax4.set_xlabel('R [AU]')
    ax4.set_ylabel('P [g s$^{-2}$]')

    for i in range(N+1):
        if ppds1.star_particles[i].disk_key >= 0:
            ds = np.abs((ppds1.disks[i].grid.column_density - ppds3.disks[i].grid.column_density)/ppds1.disks[i].grid.column_density)
            ax1.plot(ppds1.disks[i].grid.r.value_in(units.AU), ds)

            dp = np.abs((ppds1.disks[i].grid.pressure - ppds3.disks[i].grid.pressure)/ppds1.disks[i].grid.pressure)
            ax2.plot(ppds1.disks[i].grid.r.value_in(units.AU), dp)

            ax3.plot(ppds1.disks[i].grid.r.value_in(units.AU), ppds1.disks[i].grid.column_density.value_in(units.g/units.cm**2))

            ax4.plot(ppds1.disks[i].grid.r.value_in(units.AU), ppds1.disks[i].grid.pressure.value_in(units.g/units.s**2))

    print ((ppds1.star_particles.disk_gas_mass - ppds3.star_particles.disk_gas_mass)/ppds1.star_particles.disk_gas_mass)
    print ((ppds1.star_particles.accreted_mass - ppds3.star_particles.accreted_mass)/ppds1.star_particles.accreted_mass)


if __name__ == '__main__':

    np.random.seed(940234457)

    #M = np.array([0.08, 0.5, 1., 1.8, 16.])|units.MSun
    #compare_to_pooled(M)

    #N = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64])
    #test_scaling(N+1)

    #test_radiation_field()

    #test_truncation()

    #test_restartibility(10)

    test_population(30)

    plt.show()

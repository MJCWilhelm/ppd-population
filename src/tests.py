import numpy as np
import matplotlib.pyplot as plt

from ppd_population import PPD_population


def FUV_luminosity_from_mass (M, folder='../data/'):
    '''
    Compute the FUV-luminosity from stellar mass according to the power
    law fit derived from the UVBLUE spectra (at z=0.0122, 
    Rodriguez-Merino 2005). The file 'ML_fit.txt' is needed in the
    same folder. Masses are put in in solar masses, and the 
    luminosities are put out in solar luminosities.
    '''

    from amuse.units import units

    A, B, mass = np.loadtxt(folder+'/ML_fit.txt', unpack=True)

    m = M.value_in(units.MSun)

    if m < mass[0]:
        return 10.**(A[0]*np.log10(m) + B[0]) | units.LSun
    elif m > mass[-1]:
        return 10.**(A[-1]*np.log10(mass[-1]) + B[-1]) | units.LSun
    else:
        i = np.argmax(mass > m)-1
        return 10.**(A[i]*np.log10(m) + B[i]) | units.LSun


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

    ppd_code = PPD_population(number_of_workers=2)
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
            ppd_code.resolve_encounters (collision_detector)
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
            ppd_code.resolve_encounters (collision_detector)
            channels['ppd_to_star'].copy()

            channels['star_to_grav'].copy()
            gravity.evolve_model( time + dt )
            channels['grav_to_star'].copy()

        counter += 1
        time += dt

    print (ppd_code.star_particles.truncation_mass_loss.value_in(units.MSun))
    print (ppd_code.star_particles.disk_radius.value_in(units.AU))


def test_radiation_field ():

    from amuse.community.ph4.interface import ph4
    from amuse.community.huayno.interface import Huayno
    from amuse.units import nbody_system, units, constants
    from amuse.datamodel import Particles

    import matplotlib.pyplot as plt

    stars = Particles(2, 
        mass = [30., 0.1] | units.MSun,
        x = [0., 1.] | units.parsec,
        y = [0., 0.] | units.AU,
        z = [0., 0.] | units.parsec,
        vx = [0., 0.] | units.parsec/units.Myr,
        vy = [0., 359.8] | units.ms,
        vz = [0., 0.] | units.parsec/units.Myr
    )

    ppd_code = PPD_population(number_of_workers=2)
    ppd_code.add_star_particles(stars)


    converter = nbody_system.nbody_to_si(1.|units.MSun, 1.|units.parsec)
    gravity = Huayno(converter)

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


    dt = 1. | units.kyr
    t_end = 15. | units.kyr
    time = 0. | units.kyr

    counter = 0

    while time < t_end:

        print ("Evolving until {a} kyr".format(a=(time+dt).value_in(units.kyr)))

        gravity.evolve_model( time + dt/2. )
        channels['grav_to_star'].copy()

        while collision_detector.is_set():

            print ("Encounter at t={a} Myr".format(a=gravity.model_time.value_in(units.Myr)))

            channels['star_to_ppd'].copy()
            ppd_code.resolve_encounters (collision_detector)
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

            print ("Encounter at t={a} Myr".format(a=gravity.model_time.value_in(units.Myr)))

            channels['star_to_ppd'].copy()
            ppd_code.resolve_encounters (collision_detector)
            channels['ppd_to_star'].copy()

            channels['star_to_grav'].copy()
            gravity.evolve_model( time + dt )
            channels['grav_to_star'].copy()

        plt.scatter(stars[0].x.value_in(units.pc), stars[0].y.value_in(units.pc), c='r')
        plt.scatter(stars[1].x.value_in(units.pc), stars[1].y.value_in(units.pc), c='b')

        print (ppd_code.star_particles[1].outer_photoevap_rate.value_in(units.MSun/units.yr), ppd_code.star_particles[1].disk_gas_mass.value_in(units.MSun), ppd_code.star_particles[1].disk_radius.value_in(units.AU))

        counter += 1
        time += dt

    plt.show()


def test_population (N=10):

    from amuse.community.ph4.interface import ph4
    from amuse.community.huayno.interface import Huayno
    from amuse.units import nbody_system, units, constants
    from amuse.datamodel import Particles
    from amuse.ic.brokenimf import MultiplePartIMF
    from amuse.ic.plummer import new_plummer_model

    kroupa_imf = MultiplePartIMF(mass_boundaries=[0.08, 0.5, 1.9]|units.MSun,
        alphas=[-1.3, -2.3])

    mass = kroupa_imf.next_mass(N)


    converter = nbody_system.nbody_to_si(mass.sum(), 3.|units.pc)
    stars = new_plummer_model(N, converter)
    stars.mass = mass

    stars.add_particles(Particles(1,
        mass=30.|units.MSun,
        x=0.|units.pc,
        y=0.|units.pc,
        z=0.|units.pc,
        vx=0.|units.kms,
        vy=0.|units.kms,
        vz=0.|units.kms
    ))


    ppd_code = PPD_population(number_of_workers=2)
    ppd_code.add_star_particles(stars)


    gravity = Huayno(converter)

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


    ppd_code.collision_detector = gravity.stopping_conditions.collision_detection
    ppd_code.collision_detector.enable()


    dt = 1. | units.kyr
    t_end = 10. | units.kyr
    time = 0. | units.kyr

    counter = 0


    n = int(t_end/dt)

    disk_mass = np.zeros((n,N+1))
    disk_radius = np.zeros((n,N+1))


    while time < t_end:

        print ("Evolving until {a} kyr".format(a=(time+dt).value_in(units.kyr)))

        gravity.evolve_model( time + dt/2. )
        channels['grav_to_star'].copy()

        while ppd_code.collision_detector.is_set():

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

        while ppd_code.collision_detector.is_set():

            print ("Encounter at t={a} Myr".format(
                a=gravity.model_time.value_in(units.Myr)))

            channels['star_to_ppd'].copy()
            ppd_code.resolve_encounters()
            channels['ppd_to_star'].copy()

            channels['star_to_grav'].copy()
            gravity.evolve_model( time + dt )
            channels['grav_to_star'].copy()

        disk_mass[counter] = ppd_code.star_particles.disk_gas_mass.value_in(
            units.MSun)
        disk_radius[counter] = ppd_code.star_particles.disk_radius.value_in(
            units.AU)

        counter += 1
        time += dt

    plt.plot(disk_radius, disk_mass)

    plt.xscale('log')
    plt.yscale('log')


def test_single_truncation (dm):

    from amuse.units import units, constants
    from amuse.community.vader.interface import Vader
    from disk_class import Disk

    viscous = Vader(mode='pedisk', redirection='none')

    viscous.initialize_keplerian_grid(300, False, 0.01|units.AU, 3000.|units.AU,
        1.|units.MSun)

    alpha = 1e-3

    viscous.parameters.alpha = alpha
    viscous.parameters.post_timestep_function = True
    viscous.parameters.maximum_tolerated_change = 1E99
    viscous.parameters.number_of_user_parameters = 7
    viscous.parameters.inner_pressure_boundary_torque = 0. | units.g*units.cm**2./units.s**2.

    viscous.set_parameter(2, 1E-12)
    viscous.set_parameter(4, (2.33*1.008*constants.u).value_in(units.g))


    disk = Disk(100.|units.AU, 0.1|units.MSun, 1.|units.MSun, viscous.grid, alpha)
    disk.viscous = viscous

    m1 = disk.disk_gas_mass.value_in(units.MSun)

    if dm > m1:
        dm = m1
    disk.evaporate_mass ( dm | units.MSun )

    m2 = disk.disk_gas_mass.value_in(units.MSun)

    print (m1-m2-dm)

    viscous.stop()


def test_restartibility (N=10):

    from amuse.community.ph4.interface import ph4
    from amuse.community.huayno.interface import Huayno
    from amuse.units import nbody_system, units, constants
    from amuse.datamodel import Particles
    from amuse.ic.brokenimf import MultiplePartIMF
    from amuse.ic.plummer import new_plummer_model
    from ppd_population import restart_population

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


    ppds1 = PPD_population(number_of_workers=1)
    ppds1.add_star_particles(stars)

    ppds2 = PPD_population(number_of_workers=1)
    ppds2.add_star_particles(stars)


    for i in range(10):
        ppds1.evolve_model((i+1)|units.kyr)

        ppds2.evolve_model((i+1)|units.kyr)

    print (ppds1.model_time.value_in(units.Myr))
    for code in ppds1.codes:
        print (code.model_time.value_in(units.Myr))
    print (ppds2.model_time.value_in(units.Myr))
    for code in ppds2.codes:
        print (code.model_time.value_in(units.Myr))

    #ppds2.write_out()
    ppds2.write_particles()
    ppds2.write_grids()

    ppds3 = restart_population('./', 0, ppds2._params['alpha'],
        ppds2._params['mu'], ppds2._params['n_cells'], ppds2._params['r_min'],
        ppds2._params['r_max'], extra_attributes=['fuv_luminosity'],
        number_of_workers=1)


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

    #test_truncation()

    #test_radiation_field()

    #test_population()

    #for dm in np.logspace(-3, -1):
    #    test_single_truncation(dm)

    test_restartibility()

    plt.show()

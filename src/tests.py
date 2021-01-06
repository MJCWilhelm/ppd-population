import numpy as np
import matplotlib.pyplot as plt

from ppd_population import PPD_population


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


    collision_detector = gravity.stopping_conditions.collision_detection
    collision_detector.enable()


    dt = 1. | units.kyr
    t_end = 100. | units.kyr
    time = 0. | units.kyr

    counter = 0


    n = int(t_end/dt)

    disk_mass = np.zeros((n,N+1))
    disk_radius = np.zeros((n,N+1))


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

        disk_mass[counter] = ppd_code.star_particles.disk_gas_mass.value_in(
            units.MSun)
        disk_radius[counter] = ppd_code.star_particles.disk_radius.value_in(
            units.AU)

        counter += 1
        time += dt

    plt.plot(disk_radius, disk_mass)

    plt.xscale('log')
    plt.yscale('log')


if __name__ == '__main__':

    #test_truncation()

    #test_radiation_field()

    test_population()

    plt.show()

import numpy as np
import matplotlib.pyplot as plt

from amuse.units import units, constants
from amuse.datamodel import Particles
from amuse.community.vader.interface import Vader

import disk_class
import FRIED_interp


G0 = 1.6e-3 * units.erg/units.s/units.cm**2
kappa = 2e-21 / 1.008 * units.cm**2/units.amu 


def initial_disk_masses (stellar_masses):

    return 0.1 * stellar_masses
    #return (0.24 | units.MSun) * (stellar_masses.value_in(units.MSun))**0.73


def initial_disk_radii (stellar_masses):

    return [400.]*len(stellar_masses) | units.AU
    #return (200. | units.AU) * (stellar_masses.value_in(units.MSun))**0.45


def FUV_luminosity_from_mass (M, folder='../data/'):
    '''
    Compute the FUV-luminosity from stellar mass according to the power
    law fit derived from the UVBLUE spectra (at z=0.0122, 
    Rodriguez-Merino 2005). The file 'ML_fit.txt' is needed in the
    same folder. Masses are put in in solar masses, and the 
    luminosities are put out in solar luminosities.
    '''

    A, B, mass = np.loadtxt(folder+'/ML_fit.txt', unpack=True)

    m = M.value_in(units.MSun)

    N = len(M)
    n = len(mass)

    L = np.zeros(N)

    for i in range(n-1):

        mask = (m >= mass[i])*(m < mass[i+1])

        L[ mask ] = 10.**(A[i]*np.log10(mass[i]) + B[i])

    mask = m >= mass[-1]

    L[ mask ] = 10.**(A[-1]*np.log10(mass[-1]) + B[-1])

    return L | units.LSun


class PPD_population:
    '''
    [IMPORTANT]: make a channel between this classes' star_particles and the main 
    set of star particles! This class requires their positions, and alters their
    masses [1], also during initialization! It adds the mass of the disk to the
    total, gravitational mass.

    [1] PPD masses can be a significant fraction of the stellar mass; therefore,
        it is important for gravity.
    '''

    def __init__ (self, alpha=1e-3, mu=2.33, number_of_cells=330,
            number_of_workers=4, r_min=0.01|units.AU, r_max=3000.|units.AU,
            fried_folder='./', hydro=None):
        '''
        alpha: dimensionless viscosity parameter of PPDs (float)
        number_of_cells: number of grid cells of VADER codes (int)
        number_of_workers: number of VADER codes to use [1] (int)
        r_min, r_max: inner and outer edges of VADER grid (scalar, units length)
        fried_folder: directory of FRIED grid (string)        
        hydro: SPH code

        [1] more is not necessarily faster; each is a new process
        '''

        self.codes = [ Vader(mode='pedisk', redirection='none') \
            for _ in range(number_of_workers) ]


        for i in range(number_of_workers):

            self.codes[i].initialize_code()
            self.codes[i].initialize_keplerian_grid(number_of_cells, False,
                r_min, r_max, 1. | units.MSun)

            self.codes[i].parameters.alpha = alpha
            self.codes[i].parameters.post_timestep_function = True
            self.codes[i].parameters.maximum_tolerated_change = 1E99
            self.codes[i].parameters.number_of_user_parameters = 7
            self.codes[i].parameters.inner_pressure_boundary_torque = 0. | units.g*units.cm**2./units.s**2.

            # Outer density in g/cm^2
            self.codes[i].set_parameter(2, 1E-12)
            # Mean particle mass in g
            self.codes[i].set_parameter(4, (mu*1.008*constants.u).value_in(units.g))


        self.hydro = hydro

        self.interpolator = FRIED_interp.FRIED_interpolator(verbosity=False,
            folder=fried_folder)

        self.model_time = 0. | units.Myr

        self.star_particles = Particles()
        self.disks = []

        #self.star_particles.add_calculated_attribute('total_mass', 
        #    lambda stellar_mass, disk_mass: stellar_mass + disk_mass )


        # disk properties available from particles, makes for easier saving

        # utility function, gets attribute from disk or returns an alternative
        # if star has no disk (typically, massive stars)
        check_attribute = lambda disk_key, attribute, alternative: \
            alternative if disk_key < 0 else \
            getattr(self.disks[disk_key], attribute)

        # disk attributes to save with particles
        disk_attributes = ['accreted_mass', 'disk_gas_mass', 'disk_dust_mass',
            'disk_radius', 'outer_photoevap_rate',
            'disk_active', 'disk_dispersed', 'disk_convergence_failure']

        # alternatives for massive stars
        alternatives = [0.|units.MSun, 0.|units.MSun, 0.|units.MSun, 
            0.|units.AU, 0.|units.MSun/units.yr, False, False, False]

        for attr, alt in zip(disk_attributes, alternatives):
            self.star_particles.add_calculated_attribute(attr,
                lambda disk_key: check_attribute (disk_key, attr, alt) )

        self.star_particles.add_calculated_attribute('total_mass',
            lambda mass, disk_mass, accreted_mass: mass + disk_mass + accreted_mass)


    def evolve_model (self, end_time):

        # stars typically brighten but lose mass, in contrast with fuv luminosity
        # trend. Instead, it is scaled by total luminosity, effectively assuming 
        # constant surface temperature
        self.star_particles.fuv_luminosity =             \
            self.star_particles.initial_fuv_luminosity * \
           (self.star_particles.luminosity /             \
            self.star_particles.initial_luminosity)

        active_disks = []

        for i in range(len(self.disks)):
            if self.disks[i] is not None and self.disks[i].disk_active:
                self.disks[i].outer_photoevap_rate = \
                    self.compute_epe_rate(self.disks[i])
                active_disks.append(self.disks[i])

        disk_class.run_disks(self.codes, active_disks, end_time - self.model_time)

        self.star_particles.radius = 0.02 | units.pc

        self.model_time = end_time


    def compute_epe_rate (self, disk):
        '''
        Computer external photoevaporation rate for a disk from the FRIED grid

        disk: Disk object to compute EPE rate for
        '''

        if len(self.radiative_stars) == 0:
            #print ("No radiative stars")
            return 1e-10 | units.MSun/units.yr

        F = 0. | G0

        for i in range(len(self.star_particles)):

            if self.star_particles[i].fuv_luminosity > 0. | units.LSun:

                host_star = self.star_particles[disk.host_star_id]

                R = (host_star.position - self.radiative_stars[i].position).length()

                R_disk = disk.disk_radius.value_in(units.cm)/1e14
                if R < 5e17/4.*R_disk**0.5 | units.cm:
                    return 2e-9 * (1.+1.5)**2/1.5*3 * R_disk | units.MSun/units.yr

                F += self.star_particles[i].fuv_luminosity/(4.*np.pi*R*R)

                if self.hydro is not None:
                    tau = optical_depth_between_points(self.hydro,
                        host_star.position, self.radiative_stars[i].position, kappa)
                    F *= np.exp(-tau)

        return self.interpolator.interp_amuse(disk.central_mass, F,
            disk.disk_gas_mass, disk.disk_radius)


    def _periastron_distance (self, stars):

        mu = constants.G * stars.mass.sum()

        r = stars[0].position - stars[1].position

        v = stars[0].velocity - stars[1].velocity

        E = 0.5 * v.length()**2 - mu / r.length()

        a = -0.5 * mu / E

        p = (np.cross(r.value_in(units.au), 
                      v.value_in(units.m/units.s)) | units.au * units.m / units.s
            ).length()**2 / mu

        e = np.sqrt(1. - p/a)

        return p / (1. + e)


    def resolve_encounters (self, collision_detector):

        for i in range(len(collision_detector.particles(0))):

            collided_particles = Particles(particles=[
                collision_detector.particles(0)[i], 
                collision_detector.particles(1)[i]])

            r_min = self._periastron_distance(collided_particles)

            for i in range(2):
                if collided_particles[i].key in self.disked_stars.key:

                    r_new = r_min/3. * \
                        (collided_particles[i].mass/ \
                         collided_particles[1-i].mass)**0.32

                    disk = self.disk_hash[collided_particles[i].key]

                    mask = disk.grid.r > r_new

                    disk.truncation_mass_loss += (disk.grid[mask].area * (disk.grid[mask].column_density - (1e-12 | units.g/units.cm**2))).sum()

                    disk.grid[mask].column_density = 1e-12 | units.g/units.cm**2

                    const = constants.kB / (disk.mu*1.008*constants.u)
                    T = disk.Tm/np.sqrt(disk.grid[mask].r.value_in(units.AU))
                    disk.grid[mask].pressure = (1e-12 | units.g/units.cm**2) * T * const

                    self.star_particles[disk.host_star_id].radius = 0.49 * r_min


    '''
    @property
    def disk_particles (self):
        return Particles(len(self.disked_stars),
            disk_radius =    [ disk.disk_radius.value_in(units.AU)      for disk in self.disks ] | units.AU,
            disk_gas_mass =  [ disk.disk_gas_mass.value_in(units.MSun)  for disk in self.disks ] | units.MSun,
            disk_dust_mass = [ disk.disk_dust_mass.value_in(units.MSun) for disk in self.disks ] | units.MSun,
            central_mass =   [ disk.central_mass.value_in(units.MSun)   for disk in self.disks ] | units.MSun,
            accreted_mass =  [ disk.accreted_mass.value_in(units.MSun)  for disk in self.disks ] | units.MSun,
            disk_active =    [ disk.disk_active for disk in self.disks ],
            disk_dispersed = [ disk.disk_dispersed for disk in self.disks ],
            disk_convergence_failure = [ disk.disk_convergence_failure for disk in self.disks ],
            outer_photoevap_rate = [ disk.outer_photoevap_rate[0].value_in(units.MSun/units.yr) for disk in self.disks ] | units.MSun/units.yr,
            host_star = [ self.disked_stars[disk.host_star_id].key for disk in self.disks ],
        )
    '''


    #@property
    #def star_particles (self):
    #    return self.disked_stars[:].union(self.radiative_stars)


    @property
    def disked_stars (self):
        return self.star_particles.select_array(lambda disk_key: disk_key >= 0,
            ['disk_key'])


    @property
    def radiative_stars (self):
        return self.star_particles.select_array(lambda disk_key: disk_key < 0,
            ['disk_key'])


    def add_star_particles (self, new_star_particles):

        mask_disked = new_star_particles.mass < 1.9 | units.MSun

        start = len(self.star_particles)

        self.star_particles.add_particles(new_star_particles)

        for i in range(len(new_star_particles)):
            self.star_particles[start+i].radius = 0.02 | units.pc
            self.star_particles[start+i].initial_luminosity = \
                self.star_particles[start+i].luminosity

            if mask_disked[i]:
                new_disk = disk_class.Disk(
                    initial_disk_radii(new_star_particles.mass[i]), 
                    initial_disk_masses(new_star_particles.mass[i]),
                    new_star_particles[i].mass, self.codes[0].grid,
                    self.codes[0].parameters.alpha)

                new_disk.host_star_id = start+i
                new_disk.truncation_mass_loss = 0. | units.MSun

                self.disks.append(new_disk)

                self.star_particles[start+i].disk_key = start+i

            else:

                self.disks.append(None)

                self.star_particles[start+i].initial_fuv_luminosity = \
                    FUV_luminosity_from_mass(self.star_particles[start+i].mass)

                self.star_particles[start+i].disk_key = -1


    def stop (self):

        disk_class.stop_codes(self.codes)


def test_truncation ():

    from amuse.community.ph4.interface import ph4
    from amuse.units import nbody_system

    stars = Particles(2, 
        mass = 2*[1.] | units.MSun,
        x = [-1., 1.] | units.parsec,
        y = [-400., 400.] | units.AU,
        z = [0., 0.] | units.parsec,
        vx = [100., -100.] | units.parsec/units.Myr,
        vy = [0., 0.] | units.parsec/units.Myr,
        vz = [0., 0.] | units.parsec/units.Myr
    )

    ppd_code = PPD_population(number_of_workers=2, fried_folder='../data/')
    ppd_code.add_star_particles(stars)


    converter = nbody_system.nbody_to_si(1.|units.MSun, 1.|units.parsec)
    gravity = ph4(converter)

    channels = {
        'ppd_to_star': ppd_code.star_particles.new_channel_to(stars),
        'star_to_ppd': stars.new_channel_to(ppd_code.star_particles),
        'grav_to_star': gravity.particles.new_channel_to(stars),
        'star_to_grav': stars.new_channel_to(gravity.particles)
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

        counter += 1
        time += dt


def test_radiation_field ():

    from amuse.community.ph4.interface import ph4
    from amuse.community.huayno.interface import Huayno
    from amuse.units import nbody_system

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

    channels = {
        'ppd_to_star': ppd_code.star_particles.new_channel_to(stars),
        'star_to_ppd': stars.new_channel_to(ppd_code.star_particles),
        'grav_to_star': gravity.particles.new_channel_to(stars),
        'star_to_grav': stars.new_channel_to(gravity.particles)
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

        counter += 1
        time += dt

    plt.show()


if __name__ == '__main__':

    #test_truncation ()

    test_radiation_field ()

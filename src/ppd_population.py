import numpy as np
import matplotlib.pyplot as plt

from amuse.units import units, constants
from amuse.datamodel import Particles
from amuse.community.vader.interface import Vader

import disk_class
import FRIED_interp


G0 = 1.6e-3 * units.erg/units.s/units.cm**2
kappa = 2e-21 / 1.008 * units.cm**2/units.amu


def initial_disk_mass (stellar_mass):

    #return 0.1 * stellar_mass
    return (0.24 | units.MSun) * (stellar_mass.value_in(units.MSun))**0.73


def initial_disk_radius (stellar_mass):

    #return 400. | units.AU
    return (200. | units.AU) * (stellar_mass.value_in(units.MSun))**0.45


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

    if m < mass[0]:
        return 10.**(A[0]*np.log10(m) + B[0]) | units.LSun
    elif m > mass[-1]:
        return 10.**(A[-1]*np.log10(mass[-1]) + B[-1]) | units.LSun
    else:
        i = np.argmax(mass > m)-1
        return 10.**(A[i]*np.log10(m) + B[i]) | units.LSun


class Temp:
    disk_active = False


class PPD_population:
    '''
    [IMPORTANT]: when coupling with gravity or stellar evolution, be careful with
    channels! Because of the different mass components, this is non-trivial. See
    below for convenience functions to set these up.
    '''

    def __init__ (self, alpha=1e-3, mu=2.33, number_of_cells=330,
            number_of_workers=4, r_min=0.01|units.AU, r_max=3000.|units.AU,
            fried_folder='../data/', sph_hydro=None, grid_hydro=None,
            stellar_evolution=False):
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


        if grid_hydro is not None:
            self.compute_epe_rate = self.compute_epe_rate_from_radtrans
            if sph_hydro is not None:
                print ("Both grid and SPH is unphysical, please pick one.")
                print ("Defaulting to grid.")
        else:
            self.compute_epe_rate = self.compute_epe_rate_from_stars

        self.sph_hydro = sph_hydro
        self.grid_hydro = grid_hydro

        self.stellar_evolution = stellar_evolution

        self.interpolator = FRIED_interp.FRIED_interpolator(verbosity=False,
            folder=fried_folder)

        self.model_time = 0. | units.Myr

        self.star_particles = Particles()
        self.disks = []

        # disk attributes to save with particles
        self.disk_attributes = ['accreted_mass', 'disk_gas_mass', 'disk_dust_mass',
            'disk_mass', 'truncation_mass_loss', 'disk_radius',
            'outer_photoevap_rate',
            'disk_active', 'disk_dispersed', 'disk_convergence_failure']

        # alternatives for massive stars
        self.alternatives = [0.|units.MSun, 0.|units.MSun, 0.|units.MSun, 
            0.|units.MSun, 0.|units.MSun, 0.|units.AU, 0.|units.MSun/units.yr,
            False, False, False]

        self.star_particles.add_calculated_attribute('total_mass',
            lambda mass, disk_mass, accreted_mass: mass + disk_mass + accreted_mass)


    def evolve_model (self, end_time):

        # stars typically brighten but lose mass, in contrast with fuv luminosity
        # trend. Instead, it is scaled by total luminosity, effectively assuming 
        # constant surface temperature
        if self.grid_hydro is None and self.stellar_evolution:
            self.star_particles.fuv_luminosity =             \
                self.star_particles.initial_fuv_luminosity * \
               (self.star_particles.luminosity /             \
                self.star_particles.initial_luminosity)
        else:
            self.star_particles.fuv_luminosity = \
                self.star_particles.initial_fuv_luminosity

        active_disks = []

        for disk in self.disks:
            if disk is not None and disk.disk_active:
                disk.outer_photoevap_rate = self.compute_epe_rate(disk)
                disk.central_mass = self.star_particles[disk.host_star_id].mass
                active_disks.append(disk)

        disk_class.run_disks(self.codes, active_disks, end_time - self.model_time)

        self.star_particles.radius = 0.02 | units.pc

        self.model_time = end_time

        self.copy_from_disks()


    def compute_epe_rate_from_stars (self, disk):
        '''
        Compute external photoevaporation rate for a disk from the FRIED grid
        Use radiation from bright stars (>1.9 MSun)
        Radiative transfer assumes geometric attenuation (1/r^2), with potential
        manual integration through density field (if sph_hydro is not None)

        disk: Disk object to compute EPE rate for
        '''

        if len(self.radiative_stars) == 0:
            #print ("No radiative stars")
            return 1e-10 | units.MSun/units.yr

        F = 0. | G0

        for i in range(len(self.star_particles)):

            if self.star_particles[i].fuv_luminosity > 0. | units.LSun:

                host_star = self.star_particles[disk.host_star_id]

                R = (host_star.position - self.star_particles[i].position).length()

                R_disk = disk.disk_radius.value_in(units.cm)/1e14
                if R < 5e17/4.*R_disk**0.5 | units.cm:
                    return 2e-9 * (1.+1.5)**2/1.5*3 * R_disk | units.MSun/units.yr

                F += self.star_particles[i].fuv_luminosity/(4.*np.pi*R*R)

                if self.sph_hydro is not None:
                    tau = optical_depth_between_points(self.sph_hydro,
                        host_star.position, self.star_particles[i].position, kappa)
                    F *= np.exp(-tau)

        return self.interpolator.interp_amuse(disk.central_mass, F,
            disk.disk_gas_mass, disk.disk_radius)


    def compute_epe_rate_from_radtrans (self, disk):
        '''
        Compute external photoevaporation rate for a disk from the FRIED grid
        Use radiation field from radiation-hydrodynamical grid code

        disk: Disk object to compute EPE rate for
        '''

        host_star = self.star_particles[disk.host_star_id]

        i,j,k = self.grid_hydro.get_index_of_position(host_star.x, host_star.y,
            host_star.z)

        F = self.grid_hydro.get_grid_flux_photoelectric(i,j,k)

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

                disk_id = np.argmax(
                    collided_particles[i].key == self.star_particles.key)
                star_id = np.argmax(
                    collided_particles[1-i].key == self.star_particles.key)

                if self.star_particles[disk_id].disk_key >= 0:

                    r_new = r_min/3. * \
                        (self.star_particles[disk_id].mass/ \
                         self.star_particles[star_id].mass)**0.32

                    disk = self.disks[disk_id]

                    mask = disk.grid.r > r_new

                    disk.truncation_mass_loss += (disk.grid[mask].area * (disk.grid[mask].column_density - (1e-12 | units.g/units.cm**2))).sum()

                    disk.grid[mask].column_density = 1e-12 | units.g/units.cm**2

                    const = constants.kB / (disk.mu*1.008*constants.u)
                    T = disk.Tm/np.sqrt(disk.grid[mask].r.value_in(units.AU))
                    disk.grid[mask].pressure = (1e-12 | units.g/units.cm**2) * T * const

                    self.star_particles[disk.host_star_id].radius = 0.49 * r_min

                else:

                    self.star_particles[disk_id].radius = 0.49 * r_min

        self.copy_from_disks()


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

        new_star_particles.stellar_mass = new_star_particles.mass

        mask_disked = new_star_particles.mass < 1.9 | units.MSun

        start = len(self.star_particles)

        self.star_particles.add_particles(new_star_particles)

        for i in range(len(new_star_particles)):
            self.star_particles[start+i].fuv_ambient_flux = -1. | G0
            self.star_particles[start+i].radius = 0.02 | units.pc
            if self.stellar_evolution:
                self.star_particles[start+i].initial_luminosity = \
                    self.star_particles[start+i].luminosity

            if mask_disked[i]:
                new_disk = disk_class.Disk(
                    initial_disk_radius(new_star_particles.mass[i]), 
                    initial_disk_mass(new_star_particles.mass[i]),
                    new_star_particles[i].mass, self.codes[0].grid,
                    self.codes[0].parameters.alpha)

                new_disk.host_star_id = start+i
                new_disk.truncation_mass_loss = 0. | units.MSun

                self.disks.append(new_disk)

                self.star_particles[start+i].disk_key = start+i
                self.star_particles[start+i].initial_fuv_luminosity = 0.| units.LSun

            else:

                self.disks.append(Temp())

                self.star_particles[start+i].initial_fuv_luminosity = \
                    FUV_luminosity_from_mass(self.star_particles[start+i].mass)

                self.star_particles[start+i].disk_key = -1

        self.copy_from_disks()


    def copy_from_disks (self):
        for attr, alt in zip(self.disk_attributes, self.alternatives):
            for i in range(len(self.star_particles)):
                if self.star_particles[i].disk_key < 0:
                    val = alt
                else:
                    val = getattr(self.disks[self.star_particles[i].disk_key], attr)
                setattr(self.star_particles[i], attr, val)


    '''
    Convenience functions to set up channels between an external particle set,
    this code's particle set, and coupled gravity and stellar evolution codes.
    The gravity code needs the total mass, the stellar code the host star mass, etc
    '''

    def setup_self_channels (self, particles):
        '''
        Set up channels between own star_particles and external set
        Handles the separation between mass components

        particle: external particle set to set up channels with
        '''

        channel_from_self = self.star_particles.new_channel_to(particles, 
            attributes=['total_mass', 'radius'], 
            target_names=['total_mass', 'radius'])
        if self.stellar_evolution:
            channel_to_self = particles.new_channel_to(self.star_particles,
                attributes=['stellar_mass', 'x', 'y', 'z', 'luminosity'],
                target_names=['mass', 'x', 'y', 'z', 'luminosity'])
        else:
            channel_to_self = particles.new_channel_to(self.star_particles,
                attributes=['stellar_mass', 'x', 'y', 'z'],
                target_names=['mass', 'x', 'y', 'z'])

        return channel_from_self, channel_to_self


    def setup_gravity_channels (self, particles, gravity):
        '''
        Set up channels between gravity code particles and external set
        Handles the separation between mass components

        particle: external particle set to set up channels with
        gravity: gravity code to set up channels with
        '''

        channel_from_gravity = gravity.particles.new_channel_to(particles,
            attributes=['x', 'y', 'z', 'vx', 'vy', 'vz'],
            target_names=['x', 'y', 'z', 'vx', 'vy', 'vz'])
        channel_to_gravity = particles.new_channel_to(gravity.particles,
            attributes=['total_mass', 'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz'],
            target_names=['mass', 'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

        return channel_from_gravity, channel_to_gravity


    def setup_stellar_channel (self, particles, stellar, extra_attributes=[]):
        '''
        Set up channel between stellar code particles and external set
        Handles the separation between mass components

        particle: external particle set to set up channel with
        stellar: stellar evolution code to set up channel with
        '''

        channel_from_stellar = stellar.particles.new_channel_to(particles,
            attributes=['mass', 'luminosity', 'radius'].extend(extra_attributes),
            target_names=['stellar_mass', 'luminosity', 'stellar_radius'].extend(
                extra_attributes))

        return channel_from_stellar


    def stop (self):

        disk_class.stop_codes(self.codes)

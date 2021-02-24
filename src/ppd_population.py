import numpy as np
import matplotlib.pyplot as plt

import pickle
from os import path

from amuse.units import units, constants
from amuse.datamodel import Particles, new_regular_grid
from amuse.community.vader.interface import Vader
from amuse.io import write_set_to_file, read_set_from_file

import disk_class
import FRIED_interp


G0 = 1.6e-3 * units.erg/units.s/units.cm**2
kappa = 2e-21 / 1.008 * units.cm**2/units.amu


def initial_disk_mass (stellar_mass):

    try:
        N = len(stellar_mass)
    except:
        N = 0

    if N > 0:
        mask = stellar_mass < 1.9 | units.MSun

        disk_mass = np.zeros(N) | units.MSun

        disk_mass[ mask ] = (0.24 | units.MSun) * \
            (stellar_mass[ mask ].value_in(units.MSun))**0.73

        return disk_mass

    else:
        if stellar_mass < 1.9 | units.MSun:
            return (0.24 | units.MSun) * (stellar_mass.value_in(units.MSun))**0.73
        else:
            return 0. | units.MSun


def initial_disk_radius (stellar_mass):

    #return 400. | units.AU
    return (200. | units.AU) * (stellar_mass.value_in(units.MSun))**0.45

'''
def FUV_luminosity_from_mass (M, folder='../data/'):
    ''
    Compute the FUV-luminosity from stellar mass according to the power
    law fit derived from the UVBLUE spectra (at z=0.0122, 
    Rodriguez-Merino 2005). The file 'ML_fit.txt' is needed in the
    same folder. Masses are put in in solar masses, and the 
    luminosities are put out in solar luminosities.
    ''

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
'''


class PPD_population:
    '''
    [IMPORTANT]: when coupling with gravity or stellar evolution, be careful with
    channels! Because of the different mass components, this is non-trivial. See
    below for convenience functions to set these up.
    '''

    def __init__ (self, alpha=1e-3, mu=2.33, number_of_cells=330,
            number_of_workers=4, r_min=0.01|units.AU, r_max=3000.|units.AU,
            begin_time=0.|units.Myr, fried_folder=None, 
            sph_hydro=None, grid_hydro=None):
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

        self._params = {
            'alpha': alpha,
            'mu': mu,
            'r_min': r_min,
            'r_max': r_max,
            'n_cells': number_of_cells,
            'fried_folder': fried_folder,
        }

        self.output_counter = 0


        for i in range(number_of_workers):

            self.codes[i].initialize_code()
            self.codes[i].initialize_keplerian_grid(number_of_cells, False,
                r_min, r_max, 1. | units.MSun)

            self.codes[i].parameters.alpha = alpha
            self.codes[i].parameters.post_timestep_function = True
            self.codes[i].parameters.maximum_tolerated_change = 1E99
            self.codes[i].parameters.number_of_user_parameters = 7
            self.codes[i].parameters.inner_pressure_boundary_torque = \
                0. | units.g*units.cm**2./units.s**2.

            # Outer density in g/cm^2
            self.codes[i].set_parameter(2, 1E-12)
            # Mean particle mass in g
            self.codes[i].set_parameter(4,
                (mu*1.008*constants.u).value_in(units.g))


        if grid_hydro is not None:
            self.compute_rad_field = self.compute_rad_field_from_radtrans
            if sph_hydro is not None:
                print ("[PPD] Both grid and SPH is unphysical, please pick one.")
                print ("[PPD] Defaulting to grid.")
        else:
            self.compute_rad_field = self.compute_rad_field_from_stars

        self.sph_hydro = sph_hydro
        self.grid_hydro = grid_hydro

        #self.interpolator = FRIED_interp.FRIED_interpolator(verbosity=False,
        #    folder=fried_folder)

        self.model_time = begin_time

        self.star_particles = Particles()
        self.disks = []

        self.collision_detector = None

        # disk attributes to save with particles
        self.disk_attributes = ['accreted_mass', 'disk_gas_mass', 'disk_dust_mass',
            'disk_mass', 'truncation_mass_loss', 'disk_radius',
            'outer_photoevap_rate', 't_viscous', 'fuv_ambient_flux',
            'disk_active', 'disk_dispersed', 'disk_convergence_failure']

        # alternatives for massive stars
        self.alternatives = [0.|units.MSun, 0.|units.MSun, 0.|units.MSun, 
            0.|units.MSun, 0.|units.MSun, 0.|units.AU, 0.|units.MSun/units.yr,
            0.|units.yr, 0.|G0, False, False, False]

        self.star_particles.add_calculated_attribute('gravity_mass',
            lambda mass, disk_mass, accreted_mass: mass + disk_mass + accreted_mass)


    def evolve_model (self, end_time):

        active_disks = []

        for disk in self.disks:
            if disk is not None and disk.disk_active:
                #disk.outer_photoevap_rate = self.compute_epe_rate(disk)
                disk.fuv_ambient_flux = self.compute_rad_field(disk)
                disk.central_mass = self.star_particles[disk.host_star_id].mass
                active_disks.append(disk)

        print ("[PPD] Running {a}/{b} disks".format(a=len(active_disks),
            b=len(self.disked_stars)), flush=True)

        if len(active_disks) > 0:
            disk_class.run_disks(self.codes, active_disks, 
                end_time - self.model_time)
            self.copy_from_disks()

        if len(self.star_particles) > 0:
            self.star_particles.radius = 0.02 | units.pc

        self.model_time = end_time


    def compute_rad_field_from_stars (self, disk):

        host_star = self.star_particles[disk.host_star_id]

        F = 0. | G0

        if len(self.radiative_stars) == 0 or \
                self.star_particles[disk.host_star_id].star_ejected:
            return F


        for i in range(len(self.radiative_stars)):

            if self.radiative_stars[i].fuv_luminosity > 0. | units.LSun and \
                    not self.radiative_stars[i].star_ejected:

                R= (host_star.position - self.radiative_stars[i].position).length()

                dF = self.radiative_stars[i].fuv_luminosity/(4.*np.pi*R*R)

                if self.sph_hydro is not None:
                    tau = optical_depth_between_points(self.sph_hydro,
                        host_star.position, self.radiative_stars[i].position, 
                        kappa)
                    dF *= np.exp(-tau)

                F += dF

        return F


    def compute_rad_field_from_radtrans (self, disk):

        F = 0. | G0

        if self.star_particles[disk.host_star_id].star_ejected:
            return F

        host_star = self.star_particles[disk.host_star_id]

        i,j,k,m,n = self.grid_hydro.get_index_of_position(host_star.x, host_star.y,
            host_star.z)

        F = self.grid_hydro.get_grid_flux_photoelectric(i,j,k,m,n)

        return F


    '''
    def compute_epe_rate_from_stars (self, disk):
        ''
        Compute external photoevaporation rate for a disk from the FRIED grid
        Use radiation from bright stars (>1.9 MSun)
        Radiative transfer assumes geometric attenuation (1/r^2), with potential
        manual integration through density field (if sph_hydro is not None)

        disk: Disk object to compute EPE rate for
        ''

        host_star = self.star_particles[disk.host_star_id]

        if len(self.radiative_stars) == 0 or disk.disk_ejected:
            #print ("No radiative stars")
            host_star.fuv_ambient_flux = 0. | G0
            return 1e-10 | units.MSun/units.yr

        F = 0. | G0

        for i in range(len(self.star_particles)):

            if self.star_particles[i].fuv_luminosity > 0. | units.LSun:

                R = (host_star.position - self.star_particles[i].position).length()

                R_disk = disk.disk_radius.value_in(units.cm)/1e14
                if R < 5e17/4.*R_disk**0.5 | units.cm:
                    return 2e-9 * (1.+1.5)**2/1.5*3 * R_disk | units.MSun/units.yr

                dF = self.star_particles[i].fuv_luminosity/(4.*np.pi*R*R)

                if self.sph_hydro is not None:
                    tau = optical_depth_between_points(self.sph_hydro,
                        host_star.position, self.star_particles[i].position, kappa)
                    dF *= np.exp(-tau)

                F += dF

        host_star.fuv_ambient_flux = F

        if F <= 0.|G0:
            return 1e-10 | units.MSun/units.yr

        return self.interpolator.interp_amuse(disk.central_mass, F,
            disk.disk_gas_mass, disk.disk_radius)


    def compute_epe_rate_from_radtrans (self, disk):
        ''
        Compute external photoevaporation rate for a disk from the FRIED grid
        Use radiation field from radiation-hydrodynamical grid code

        disk: Disk object to compute EPE rate for
        ''

        if disk.disk_ejected:
            host_star.fuv_ambient_flux = 0. | G0
            return 1e-10 | units.MSun/units.yr


        host_star = self.star_particles[disk.host_star_id]

        i,j,k,m,n = self.grid_hydro.get_index_of_position(host_star.x, host_star.y,
            host_star.z)

        F = self.grid_hydro.get_grid_flux_photoelectric(i,j,k,m,n)

        host_star.fuv_ambient_flux = F

        #print('Rad field:', F.value_in(G0), host_star.position.value_in(units.pc), i,j,k,m,n, flush=True)

        if F <= 0.|G0:
            return 1e-10 | units.MSun/units.yr

        return self.interpolator.interp_amuse(disk.central_mass, F,
            disk.disk_gas_mass, disk.disk_radius)
    '''


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


    def resolve_encounters (self):

        if self.collision_detector is None:
            print ("[PPD] No collision detector set!")
            return

        for i in range(len(self.collision_detector.particles(0))):

            collided_particles = Particles(particles=[
                self.collision_detector.particles(0)[i], 
                self.collision_detector.particles(1)[i]])

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


    @property
    def disked_stars (self):
        if len(self.star_particles):
            return self.star_particles.select_array(lambda disk_key: disk_key >= 0,
                ['disk_key'])
        else:
            return self.star_particles


    @property
    def radiative_stars (self):
        if len(self.star_particles):
            return self.star_particles.select_array(lambda disk_key: disk_key < 0,
                ['disk_key'])
        else:
            return self.star_particles


    def add_star_particles (self, new_star_particles):

        mask_disked = new_star_particles.mass < 1.9 | units.MSun

        start = len(self.star_particles)

        self.star_particles.add_particles(new_star_particles)
        self.star_particles[start:].initial_mass = new_star_particles.mass

        for i in range(len(new_star_particles)):
            self.star_particles[start+i].fuv_ambient_flux = 0. | G0
            self.star_particles[start+i].radius = 0.02 | units.pc
            #self.star_particles[start+i].fuv_luminosity = 0. | units.LSun

            if mask_disked[i]:
                new_disk = disk_class.Disk(
                    initial_disk_radius(new_star_particles.mass[i]), 
                    initial_disk_mass(new_star_particles.mass[i]),
                    new_star_particles[i].mass, self.codes[0].grid,
                    self.codes[0].parameters.alpha,
                    fried_folder=self._params['fried_folder'])

                new_disk.host_star_id = start+i
                new_disk.truncation_mass_loss = 0. | units.MSun

                self.disks.append(new_disk)

                self.star_particles[start+i].disk_key = start+i
                self.star_particles[start+i].star_ejected = False
                

            else:

                self.disks.append(None)

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
            attributes=['gravity_mass', 'radius'], 
            target_names=['gravity_mass', 'radius'])
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
            attributes=['gravity_mass', 'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz'],
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


    def write_particles (self, filepath='./', label='', overwrite=True):

        if len(self.star_particles) > 0:
            write_set_to_file(self.star_particles, 
                filepath+'/viscous_particles_{b}i{a:05}.hdf5'.format(
                    a=self.output_counter, b=label), 'hdf5',
                    timestamp=self.model_time, overwrite_file=overwrite)
        else:
            print ("[PPD] No particles to write!")


    def write_grids (self, filepath='./', label='', overwrite=True):

        N_disks = len(self.disked_stars)

        if N_disks == 0:
            print ("[PPD] No disks to write!")
            return

        N_grid = len(self.codes[0].grid.r)

        grids = new_regular_grid((N_disks, N_grid), [1., 1.], axes_names='nm')

        for i in range(N_disks):
            disk = self.disks[self.disked_stars[i].disk_key]

            grids[i].r = disk.grid.r
            grids[i].column_density = disk.grid.column_density
            grids[i].pressure = disk.grid.pressure

        write_set_to_file(grids, filepath+'/viscous_grids_{b}i{a:05}.hdf5'.format(
            a=self.output_counter, b=label), 'hdf5', timestamp=self.model_time, 
            overwrite_file=overwrite)


    def write_parameters (self, filepath='./', overwrite=True):

        if not path.isfile(filepath+'/viscous_params.pickle'):
            param_dump = open(filepath+'/viscous_params.pickle', 'wb')
            pickle.dump(self._params, param_dump)
            param_dump.close()



def restart_population (filepath, input_counter, alpha, mu, n_cells, r_min, r_max, 
        fried_folder=None, number_of_workers=4, sph_hydro=None, grid_hydro=None, 
        label='', extra_attributes=[]):

    if path.isfile(filepath+'/viscous_grids_{b}i{a:05}.hdf5'.format(
            a=input_counter, b=label)):
        grids = read_set_from_file(
            filepath+'/viscous_grids_{b}i{a:05}.hdf5'.format(a=input_counter,
            b=label), 'hdf5')
    else:
        grids = None

    if path.isfile(filepath+'/viscous_particles_{b}i{a:05}.hdf5'.format(
            a=input_counter, b=label)):
        star_particles = read_set_from_file(
            filepath+'/viscous_particles_{b}i{a:05}.hdf5'.format(a=input_counter,
            b=label), 'hdf5')
    else:
        star_particles = None


    ppd_code = PPD_population(alpha=alpha, mu=mu, number_of_cells=n_cells,
            number_of_workers=number_of_workers, 
            r_min=r_min, r_max=r_max,
            fried_folder=fried_folder, sph_hydro=sph_hydro, 
            grid_hydro=grid_hydro)


    if star_particles is None:
        return ppd_code


    ppd_code.model_time = star_particles.get_timestamp()


    if grids is None:
        ppd_code.add_star_particles(star_particles)
        ppd_code.star_particles.initial_mass = star_particles.initial_mass
        return ppd_code


    N_stars = len(star_particles)
    N_disks = grids.shape[0]
    N_cells = grids.shape[1]


    non_disk_attributes = ['key', 'disk_key', 'mass', 'position', 'velocity',
        'radius', 'initial_mass', 'star_ejected']

    ppd_code.star_particles.add_particles(Particles(N_stars))

    for attr in non_disk_attributes:
        setattr(ppd_code.star_particles, attr, getattr(star_particles, attr))


    for attribute in extra_attributes:
        if hasattr(star_particles, attribute):
            setattr(ppd_code.star_particles, attribute, 
                getattr(star_particles, attribute))


    disk_counter = 0
    for i in range(N_stars):

        if star_particles.disk_key[i] < 0:
            ppd_code.disks.append(None)

        else:
            disk = disk_class.Disk(100.|units.AU, 0.1|units.MSun,
                star_particles.initial_mass[i], ppd_code.codes[0].grid,
                alpha, mu=mu, fried_folder=fried_folder)

            host_star = star_particles[i]

            disk.model_time = grids.get_timestamp()
            disk.disk_dispersed = host_star.disk_dispersed
            disk.disk_convergence_failure = host_star.disk_convergence_failure
            disk.disk_dust_mass = host_star.disk_dust_mass
            disk.accreted_mass = host_star.accreted_mass
            disk.t_viscous = host_star.t_viscous
            disk.host_star_id = i
            disk.truncation_mass_loss = host_star.truncation_mass_loss

            disk.grid.column_density = grids[disk_counter].column_density
            disk.grid.pressure = grids[disk_counter].pressure

            ppd_code.disks.append(disk)

            disk_counter += 1

    ppd_code.copy_from_disks()

    ppd_code.output_counter = input_counter + 1

    return ppd_code

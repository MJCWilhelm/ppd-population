import numpy as np
import matplotlib.pyplot as plt

import pickle
import time
from os import path

from amuse.community.vader.interface import Vader
from amuse.units import units, constants
from amuse.datamodel import Particles, new_regular_grid
from amuse.io import write_set_to_file, read_set_from_file

import disk_class
import FRIED_interp
from ppd_population import G0, initial_disk_mass, initial_disk_radius


class PPDPopulationAsync:

    def __init__ (self, alpha=1e-3, mu=2.33, number_of_cells=330, 
            number_of_workers=4, r_min=0.01|units.AU, r_max=3000.|units.AU,
            begin_time=0.|units.Myr, max_frac=1., eta=0.3, fried_folder='../data/', 
            sph_hydro=None, grid_hydro=None, dust_model='Haworth2018',
            vader_mode='pedisk_nataccr', dust_params=None):

        self.codes = [ Vader(mode=vader_mode, redirection='none') \
            for _ in range(number_of_workers) ]

        for i in range(number_of_workers):

            self.codes[i].initialize_code()
            self.codes[i].initialize_keplerian_grid(number_of_cells, False,
                r_min, r_max, 1. | units.MSun)

            self.codes[i].parameters.alpha = alpha
            self.codes[i].parameters.post_timestep_function = True
            self.codes[i].parameters.maximum_tolerated_change = 1E99
            self.codes[i].parameters.inner_pressure_boundary_type = 1
            self.codes[i].parameters.inner_boundary_function = True
            self.codes[i].parameters.initial_timestep = 1.|units.yr

            if vader_mode == 'pedisk_dusty':
                self.codes[i].parameters.number_of_user_parameters = 14
                self.codes[i].parameters.number_of_user_outputs = 3
            else:
                self.codes[i].parameters.number_of_user_parameters = 7
                self.codes[i].parameters.number_of_user_outputs = 0

            self.codes[i].recommit_parameters()

            # Outer density in g/cm^2
            self.codes[i].set_parameter(2, 1E-12)
            # Mean particle mass in g
            self.codes[i].set_parameter(4,
                (mu*1.008*constants.u).value_in(units.g))

            if vader_mode == 'pedisk_dusty':
                self.codes[i].set_parameter(3, dust_params['Tsubl'].value_in(units.K))
                self.codes[i].set_parameter(5, alpha)
                self.codes[i].set_parameter(7,
                    dust_params['uf'].value_in(units.cm/units.s))
                self.codes[i].set_parameter(8,
                    dust_params['rho_s'].value_in(units.g/units.cm**3))
                self.codes[i].set_parameter(9, dust_params['a0'].value_in(units.cm))
                self.codes[i].set_parameter(10, dust_params['CFL'])
                self.codes[i].set_parameter(11, 0.)
                self.codes[i].set_parameter(12, 0.)
                self.codes[i].set_parameter(13, 0.)

            elif vader_mode == 'pedisk_nataccr':
                self.codes[i].set_parameter(5, alpha)

        self._params = {
            'alpha': alpha,
            'mu': mu,
            'r_min': r_min,
            'r_max': r_max,
            'n_cells': number_of_cells,
            'fried_folder': fried_folder,
            'vader_mode': vader_mode,
            'max_frac': max_frac,
        }

        self.dust_params = dust_params

        self.max_frac = max_frac
        self.eta = eta

        self.output_counter = 0

        if grid_hydro is not None:
            self.compute_rad_field = self.compute_rad_field_from_radtrans
            if sph_hydro is not None:
                print ("[PPDA] Both grid and SPH is unphysical, please pick one.")
                print ("[PPDA] Defaulting to grid.", flush=True)
        else:
            self.compute_rad_field = self.compute_rad_field_from_stars

        self.sph_hydro = sph_hydro
        self.grid_hydro = grid_hydro

        self.interpolator = FRIED_interp.FRIED_interpolator(verbosity=False,
            folder=fried_folder)

        self.model_time = begin_time

        self.star_particles = Particles()
        self.disks = []

        self.collision_detector = None

        # disk attributes to save with particles
        self.disk_attributes = ['accreted_mass', 'disk_gas_mass', 'disk_dust_mass',
            'disk_mass', 'truncation_mass_loss', 'disk_radius',
            'outer_photoevap_rate', 't_viscous', 'age',
            'disk_active', 'disk_dispersed', 'disk_convergence_failure',
            'ipe_mass_loss', 'epe_mass_loss']

        # alternatives for massive stars
        self.alternatives = [0.|units.MSun, 0.|units.MSun, 0.|units.MSun, 
            0.|units.MSun, 0.|units.MSun, 0.|units.AU, 0.|units.MSun/units.yr,
            0.|units.yr, 0.|units.yr, False, False, False, 
            0.|units.MSun, 0.|units.MSun]

        self.star_particles.add_calculated_attribute('gravity_mass',
            lambda mass, disk_mass, accreted_mass: mass + disk_mass + accreted_mass)

        self.dust_model = dust_model


    def compute_epe_rate (self, disk_ids):

        host_stars = self.star_particles[disk_ids]
        mask_not_ejected = host_stars.star_ejected == False

        epe_rate = np.ones(len(disk_ids)) * 1e-10 | units.MSun/units.yr

        host_stars[mask_not_ejected == False].fuv_ambient_flux = 0. | G0

        F = self.compute_rad_field(disk_ids[mask_not_ejected])

        host_stars[mask_not_ejected].fuv_ambient_flux = F

        mask_nonzero_F = F > 0.|G0

        inds_epe = np.arange(len(disk_ids))[mask_not_ejected][mask_nonzero_F]

        epe_rate[inds_epe] = self.interpolator.interp_amuse(
            host_stars[inds_epe].mass, F[mask_nonzero_F],
            [ self.disks[disk_id].disk_gas_mass.value_in(units.MSun) \
                for disk_id in disk_ids[inds_epe] ] | units.MSun,
            [ self.disks[disk_id].disk_radius.value_in(units.au) \
                for disk_id in disk_ids[inds_epe] ] | units.au)

        return epe_rate


    def compute_rad_field_from_radtrans (self, disk_ids):

        host_stars = self.star_particles[disk_ids]

        i,j,k,m,n = self.grid_hydro.get_index_of_position(host_stars.x,
            host_stars.y, host_stars.z)

        F = self.grid_hydro.get_grid_flux_photoelectric(i,j,k,m,n)

        return F


    def compute_rad_field_from_stars (self, disk_ids):

        host_stars = self.star_particles[disk_ids]

        F = np.zeros(len(disk_ids)) | G0

        if len(self.radiative_stars) > 0:

            for i in range(len(self.radiative_stars)):

                if self.radiative_stars[i].fuv_luminosity > 0. | units.LSun and \
                        not self.radiative_stars[i].star_ejected:

                    R = (host_stars.position - self.radiative_stars[i].position) \
                        .lengths()

                    dF = self.radiative_stars[i].fuv_luminosity/(4.*np.pi*R*R)

                    if self.sph_hydro is not None:
                        print ("[PPDA] Extinction in SPH hydro is currently not implemented", flush=True)
                        '''
                        tau = optical_depth_between_points(self.sph_hydro,
                            host_star.position, self.radiative_stars[i].position, 
                            kappa)
                        dF *= np.exp(-tau)
                        '''

                    F += dF

        return F


    def evolve_model (self, end_time):

        mask_active = [ disk is not None and disk.disk_active == True \
            for disk in self.disks ]

        if len(mask_active):
            disk_ids = np.arange(len(self.disks))[mask_active]

            epe_rate = self.compute_epe_rate(disk_ids)
            for i in range(len(disk_ids)):
                self.disks[disk_ids[i]].outer_photoevap_rate = epe_rate[i]
                self.disks[disk_ids[i]].central_mass = \
                    self.star_particles[disk_ids[i]].mass

            self.evolve_model_adaptive(disk_ids, end_time - self.model_time)

            self.copy_from_disks()

        if len(self.star_particles):
            self.star_particles.radius = 0.02 | units.pc

        self.model_time = end_time


    def evolve_model_adaptive (self, disk_ids, dt):

        if len(disk_ids) == 0:
            return

        disk_mass = [ self.disks[disk_id].disk_gas_mass.value_in(units.MSun) \
            for disk_id in disk_ids ] | units.MSun
        epe_rate = [ self.disks[disk_id].outer_photoevap_rate.value_in(
            units.MSun/units.yr) for disk_id in disk_ids ] | units.MSun/units.yr

        tau = self.eta * disk_mass / epe_rate

        mask_subcycle = tau < dt

        if np.sum(mask_subcycle):
            # Subcycle rapidly evaporating disks
            self.evolve_model_adaptive(disk_ids[ mask_subcycle ], dt/2.)

            # For disks that are still active, recompute EPE rate
            mask_still_active = [ self.disks[disk_id].disk_active == True \
                for disk_id in disk_ids[ mask_subcycle ] ]
            still_active_disk_ids = disk_ids[ mask_subcycle ][ mask_still_active ]

            if len(still_active_disk_ids):
                subcycle_epe_rate = self.compute_epe_rate(still_active_disk_ids)
                for i in range(len(still_active_disk_ids)):
                    self.disks[still_active_disk_ids[i]].outer_photoevap_rate = \
                        subcycle_epe_rate[i]
                self.evolve_model_adaptive(still_active_disk_ids, dt/2.)

        # Evolve disks that are not subcycled, and are still active
        # We select for active disks in the top call, but we can be in a subcycle!
        mask_active = (mask_subcycle == False)*np.array([
            self.disks[disk_id].disk_active == True for disk_id in disk_ids])


        if self._params['vader_mode'] == 'pedisk':
            # Compatibility with old model, should be bitwise equal to that!
            # Not any more now that 2nd call updates accretion rate, but it was that
            # or using parameters from different disks...
            self.evolve_disks(disk_ids[ mask_active ], dt/2.)

            self.evolve_Haworth2018_dust_model(disk_ids[ mask_active ], dt)

            self.evolve_disks(disk_ids[ mask_active ], dt/2.)

        else:
            # Leap-frog like operator splitting for Haworth dust model
            # Makes more sense than leap-frogging VADER
            if self.dust_model == 'Haworth2018':
                self.evolve_Haworth2018_dust_model(disk_ids[ mask_active ], dt/2.)

            self.evolve_disks(disk_ids[ mask_active ], dt)

            if self.dust_model == 'Haworth2018':
                self.evolve_Haworth2018_dust_model(disk_ids[ mask_active ], dt/2.)


    def evolve_disks (self, disk_ids, dt):

        print ("[PPDA] Evolving {a}/{b} disks for {c} kyr".format(
            a=len(disk_ids), b=len(self.disked_stars), c=dt.value_in(units.kyr)),
            flush=True)

        active_disk_ids = list(disk_ids)

        # Treat disk ids to evolve as a stack
        while len(active_disk_ids):

            evolve_disk_ids, accreted_mass_before = [], []
            ipe_before, epe_before = [], []

            channels_to_codes, channels_from_codes = [], []

            counter = 0
            # Pop off stack a number of disks equal to the number of codes, or the
            # remaining disks if smaller
            while len(evolve_disk_ids) < len(self.codes) and len(active_disk_ids):

                evolve_disk_ids.append(active_disk_ids.pop(0))

                # Make channels to copy data between disks/codes
                channels_from_codes.append([self.codes[counter].grid.new_channel_to(
                    self.disks[evolve_disk_ids[counter]].grid)])
                channels_to_codes.append([
                    self.disks[evolve_disk_ids[counter]].grid.new_channel_to(
                        self.codes[counter].grid)])

                # Make additional channels for the user grid, if present
                if self.codes[0].parameters.number_of_user_outputs:
                    channels_from_codes[-1].append(
                      self.codes[counter].grid_user.new_channel_to(
                        self.disks[evolve_disk_ids[counter]].grid_user))
                    channels_to_codes[-1].append(
                      self.disks[evolve_disk_ids[counter]].grid_user.new_channel_to(
                        self.codes[counter].grid_user))

                # Get mass lost in code before now, to compute mass lost in step
                accreted_mass_before.append(
                    -self.codes[counter].inner_boundary_mass_out)
                ipe_before.append(self.codes[counter].get_parameter(12))
                epe_before.append(self.codes[counter].get_parameter(13))

                counter += 1

            for i in range(len(evolve_disk_ids)):
                # Prepare codes for evolution by implanting disk
                disk = self.disks[evolve_disk_ids[i]]

                self.codes[i].update_keplerian_grid(disk.central_mass)

                self.codes[i].set_parameter(0, disk.internal_photoevap_flag * \
                    disk.inner_photoevap_rate.value_in(units.g/units.s))
                self.codes[i].set_parameter(1, disk.external_photoevap_flag * \
                    disk.outer_photoevap_rate.value_in(units.g/units.s))
                if self._params['vader_mode'] != 'pedisk_dusty':
                    self.codes[i].set_parameter(3, disk.Tm.value_in(units.K))
                if self._params['vader_mode'] == 'pedisk':
                    self.codes[i].set_parameter(5, disk.accretion_rate.value_in(
                        units.g/units.s))
                self.codes[i].set_parameter(6,
                    disk.central_mass.value_in(units.MSun))

                for channel in channels_to_codes[i]:
                    channel.copy()

                if self._params['vader_mode'] != 'pedisk_dusty':
                    disk.ipe_mass_loss += disk.internal_photoevap_flag * \
                        disk.inner_photoevap_rate * dt
                    disk.epe_mass_loss += disk.external_photoevap_flag * \
                        disk.outer_photoevap_rate * dt

            pool = []
            for i in range(len(evolve_disk_ids)):
                # Add jobs to pool
                pool.append( self.codes[i].evolve_model.asynchronous(
                    self.codes[i].model_time + dt) )

            for i in range(len(evolve_disk_ids)):
                # This errors if the code crashes, if so continue the simulation
                # but turn off this disk's evolution
                try:
                    pool[i].wait()
                except:
                    print ("[PPDA] Absolute convergence failure of disk {b} at {a} Myr".format(
                        a=self.disks[evolve_disk_ids[i]].model_time.value_in(
                            units.Myr), b=evolve_disk_ids[i]), flush=True)
                    self.disks[evolve_disk_ids[i]].disk_convergence_failure = True
                # Also take errors of dust evolution into account
                if self._params['vader_mode'] == 'pedisk_dusty' and \
                        self.codes[i].get_parameter(11) != 0.:
                    print ("[PPDA] Absolute convergence failure of disk {b} at {a} Myr, in dust".format(
                        a=self.disks[evolve_disk_ids[i]].model_time.value_in(
                            units.Myr), b=evolve_disk_ids[i]), flush=True)
                    self.disks[evolve_disk_ids[i]].disk_convergence_failure = True

            for i in range(len(evolve_disk_ids)):
                # Extract updated information from codes
                disk = self.disks[evolve_disk_ids[i]]

                disk.model_time += dt

                accreted_mass = -self.codes[i].inner_boundary_mass_out - \
                    accreted_mass_before[i]
                disk.accreted_mass += accreted_mass
                ''' May still be available for planet formation through pebble accretion
                if self.dust_model == 'Haworth2018':
                    accreted_dust = min(disk._disk_dust_mass,
                        disk.delta*accreted_mass)
                    disk.accreted_mass  += accreted_dust
                    disk._disk_dust_mass -= accreted_dust
                '''

                if self._params['vader_mode'] == 'pedisk_dusty':
                    disk.ipe_mass_loss += (self.codes[i].get_parameter(12) - \
                        ipe_before[i]) | units.g
                    disk.epe_mass_loss += (self.codes[i].get_parameter(13) - \
                        epe_before[i]) | units.g

                for channel in channels_from_codes[i]:
                    channel.copy()

                if disk.disk_gas_mass < 0.00008 | units.MSun:
                    # Dispersed if below 1% of a 10% disk of a 0.08 MSun star
                    disk.disk_dispersed = True
                    print ('[PPDA] Disk {b} dispersal at {a} Myr'.format(
                        a=disk.model_time.value_in(units.Myr), 
                        b=evolve_disk_ids[i]), flush=True)


    def evolve_Haworth2018_dust_model (self, disk_ids, dt):
        # Remove dust following the prescription of Haworth et al. 2018 (MNRAS 475)

      for i in range(len(disk_ids)):
        disk = self.disks[disk_ids[i]]

        # Thermal speed of particles
        v_th =(8.*constants.kB*disk.Tm/np.sqrt(disk.disk_radius.value_in(units.AU))\
            /(np.pi * disk.mu*1.008*constants.u))**(1./2.)
        # Disk scale height at disk edge
        Hd =(constants.kB*disk.Tm*(1.|units.AU)**(1./2.)*disk.disk_radius**(5./2.)/\
            (disk.mu*1.008*constants.u*disk.central_mass*constants.G))**(1./2.)
        # Disk filling factor of sphere at disk edge
        F = Hd/(Hd**2 + disk.disk_radius**2)**(1./2.)

        disk.dust_photoevap_rate = disk.external_photoevap_flag * disk.delta * \
            disk.outer_photoevap_rate**(3./2.) * (v_th/( 4.*np.pi * F * \
            constants.G*disk.central_mass * disk.rho_g*disk.a_min ))**(1./2.) * \
            np.exp( -disk.delta*(constants.G*disk.central_mass)**(1./2.) * \
                disk.model_time/(2.*disk.disk_radius**(3./2.)) )

        # Can't entrain more dust than is available
        if  disk.dust_photoevap_rate > disk.delta*disk.outer_photoevap_rate:
            disk.dust_photoevap_rate = disk.delta*disk.outer_photoevap_rate

        # Eulerian integration
        dM_dust = disk.dust_photoevap_rate * dt
        if disk.disk_dispersed: # If disk is dispersed, do only half a step
            dM_dust /= 2.
        disk._disk_dust_mass -= dM_dust

        # Can't have negative mass
        if  disk._disk_dust_mass < 0. | units.MSun:
            disk._disk_dust_mass = 0. | units.MSun


    def _periastron_distance (self, stars):
        '''
        Compute the periastron of two stars' two-body orbit
        '''

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
        '''
        Resolve all dynamic truncations
        '''

        if self.collision_detector is None:
            print ("[PPDA] No collision detector set!", flush=True)
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

                    # Also truncate dust, including resetting dust sizes
                    if self._params['vader_mode'] == 'pedisk_dusty':
                        disk.grid_user[0,mask].value = 1e-14
                        disk.grid_user[1,mask].value = \
                            self.dust_params['a0'].value_in(units.cm)

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

                host_star_id = start + i

                new_disk = disk_class.Disk(
                    initial_disk_radius(new_star_particles.mass[i], 
                        max_frac=self.max_frac), 
                    initial_disk_mass(new_star_particles.mass[i],
                        max_frac=self.max_frac),
                    new_star_particles[i].mass, self.codes[0].grid,
                    self._params['alpha'])

                if self._params['vader_mode'] == 'pedisk_dusty':
                    new_disk.grid_user = self.codes[0].grid_user.copy()
                    new_disk.grid_user[0].value = 0.01 * \
                        new_disk.grid.column_density.value_in(units.g/units.cm**2)
                    new_disk.grid_user[1].value = \
                        self.dust_params['a0'].value_in(units.cm)
                    new_disk.grid_user[2].value = new_disk.Tm.value_in(units.K) * \
                        new_disk.grid.r.value_in(units.au)**-0.5
                    # A sublimated innermost cell and unsublimated 2nd cell
                    # gives weird edge effects (tries to maintain constant
                    # dust-to-gas ratio, so pulls in dust, which is then
                    # sublimated, etc). Fine if both are sublimated, so just
                    # ignore sublimation if only in the innermost cell
                    if new_disk.grid_user[2,0].value|units.K >= self.dust_params['Tsubl'] and \
                       new_disk.grid_user[2,1].value|units.K <  self.dust_params['Tsubl']:
                        new_disk.grid_user[2,0].value = new_disk.grid_user[2,1].value

                new_disk.host_star_id = host_star_id
                new_disk.truncation_mass_loss = 0. | units.MSun

                self.disks.append(new_disk)

                self.star_particles[host_star_id].disk_key = host_star_id
                self.star_particles[host_star_id].star_ejected = False


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

        for code in self.codes:
            code.stop()


    def write_particles (self, filepath='./', label='', overwrite=True):

        if len(self.star_particles) > 0:
            write_set_to_file(self.star_particles, 
                filepath+'/viscous_particles_{b}i{a:05}.hdf5'.format(
                    a=self.output_counter, b=label), 'hdf5',
                    timestamp=self.model_time, overwrite_file=overwrite)
        else:
            print ("[PPDA] No particles to write!", flush=True)


    def write_grids (self, filepath='./', label='', overwrite=True):

        N_disks = len(self.disked_stars)

        if N_disks == 0:
            print ("[PPDA] No disks to write!", flush=True)
            return

        N_grid = len(self.disks[self.disked_stars[0].disk_key].grid.r)

        grids = new_regular_grid((N_disks, N_grid), [1., 1.], axes_names='nm')

        with_grid_user = self.codes[0].parameters.number_of_user_outputs > 0
        if with_grid_user:
            grids_user = new_regular_grid(
                (N_disks, self.codes[0].parameters.number_of_user_outputs, N_grid),
                [1., 1., 1.], axes_names='nml')

        for i in range(N_disks):
            disk = self.disks[self.disked_stars[i].disk_key]

            grids[i].r = disk.grid.r
            grids[i].column_density = disk.grid.column_density
            grids[i].pressure = disk.grid.pressure

            if with_grid_user:
                grids_user[i].value = disk.grid_user.value

        write_set_to_file(grids, filepath+'/viscous_grids_{b}i{a:05}.hdf5'.format(
            a=self.output_counter, b=label), 'hdf5', timestamp=self.model_time, 
            overwrite_file=overwrite)

        if with_grid_user:
            write_set_to_file(grids_user, 
                filepath+'viscous_grids_user_{b}i{a:05}.hdf5'.format(
                a=self.output_counter, b=label), 'hdf5', timestamp=self.model_time, 
                overwrite_file=overwrite)


    def write_parameters (self, filepath='./', overwrite=True):

        self._params['dust_params'] = self.dust_params
        if not path.isfile(filepath+'/viscous_params.pickle'):
            param_dump = open(filepath+'/viscous_params.pickle', 'wb')
            pickle.dump(self._params, param_dump)
            param_dump.close()


def restart_population (filepath, input_counter, alpha, mu, n_cells, r_min, r_max, 
        number_of_workers=4, fried_folder='./', sph_hydro=None, grid_hydro=None, 
        label='', extra_attributes=[], vader_mode='pedisk_nataccr', 
        max_frac=1., dust_params=None, dust_model='Haworth2018'):

    if path.isfile(filepath+'/viscous_grids_{b}i{a:05}.hdf5'.format(
            a=input_counter, b=label)):
        grids = read_set_from_file(
            filepath+'/viscous_grids_{b}i{a:05}.hdf5'.format(a=input_counter,
            b=label), 'hdf5')
    else:
        grids = None

    if path.isfile(filepath+'/viscous_grids_user_{b}i{a:05}.hdf5'.format(
            a=input_counter, b=label)):
        grids_user = read_set_from_file(
            filepath+'/viscous_grids_user_{b}i{a:05}.hdf5'.format(a=input_counter,
            b=label), 'hdf5')
    else:
        grids_user = None

    if path.isfile(filepath+'/viscous_particles_{b}i{a:05}.hdf5'.format(
            a=input_counter, b=label)):
        star_particles = read_set_from_file(
            filepath+'/viscous_particles_{b}i{a:05}.hdf5'.format(a=input_counter,
            b=label), 'hdf5')
    else:
        star_particles = None


    ppd_code = PPDPopulationAsync(alpha=alpha, mu=mu, number_of_cells=n_cells,
            number_of_workers=number_of_workers, r_min=r_min, r_max=r_max,
            fried_folder=fried_folder, sph_hydro=sph_hydro, grid_hydro=grid_hydro,
            vader_mode=vader_mode, max_frac=max_frac, dust_params=dust_params,
            dust_model=dust_model)


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
                alpha, mu=mu)

            host_star = star_particles[i]

            disk.central_mass = host_star.mass
            disk.disk_dispersed = host_star.disk_dispersed
            disk.disk_convergence_failure = host_star.disk_convergence_failure
            disk._disk_dust_mass = host_star.disk_dust_mass
            disk.accreted_mass = host_star.accreted_mass
            disk.t_viscous = host_star.t_viscous
            if hasattr(host_star, 'age'):
                disk.age = host_star.age
            disk.host_star_id = i
            disk.truncation_mass_loss = host_star.truncation_mass_loss
            disk.ipe_mass_loss = host_star.ipe_mass_loss
            disk.epe_mass_loss = host_star.epe_mass_loss

            disk.grid.column_density = grids[disk_counter].column_density
            disk.grid.pressure = grids[disk_counter].pressure

            if grids_user is not None:
                disk.grid_user = ppd_code.codes[0].grid_user.copy()
                disk.grid_user.value = grids_user[disk_counter].value

            ppd_code.disks.append(disk)

            disk_counter += 1

    ppd_code.copy_from_disks()

    ppd_code.output_counter = input_counter + 1

    return ppd_code

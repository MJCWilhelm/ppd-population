import numpy as np
import matplotlib.pyplot as plt

import pickle
from os import path

from amuse.community.vader.interface import Vader
from amuse.units import units, constants
from amuse.datamodel import Particles, new_regular_grid
from amuse.io import write_set_to_file, read_set_from_file
#from amuse.rfi.async_request import AsyncRequestsPool

import disk_class
import FRIED_interp
from ppd_population import G0, initial_disk_mass, initial_disk_radius


class PPDPopulationAsync:

    def __init__ (self, alpha=1e-3, mu=2.33, number_of_cells=330,
            r_min=0.01|units.AU, r_max=3000.|units.AU,
            begin_time=0.|units.Myr, max_frac=1., eta=1., fried_folder='../data/', 
            sph_hydro=None, grid_hydro=None, dust_model='Haworth2018',
            vader_mode='pedisk_nataccr'):

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

        self.max_frac = max_frac
        self.eta = eta

        self.output_counter = 0

        if grid_hydro is not None:
            self.compute_rad_field = self.compute_rad_field_from_radtrans
            if sph_hydro is not None:
                print ("[PPDA] Both grid and SPH is unphysical, please pick one.")
                print ("[PPDA] Defaulting to grid.")
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
            'outer_photoevap_rate', 't_viscous', 'age', 'fuv_ambient_flux',
            'disk_active', 'disk_dispersed', 'disk_convergence_failure',
            'ipe_mass_loss', 'epe_mass_loss']

        # alternatives for massive stars
        self.alternatives = [0.|units.MSun, 0.|units.MSun, 0.|units.MSun, 
            0.|units.MSun, 0.|units.MSun, 0.|units.AU, 0.|units.MSun/units.yr,
            0.|units.yr, 0.|units.yr, 0.|G0, False, False, False, 
            0.|units.MSun, 0.|units.MSun]

        self.star_particles.add_calculated_attribute('gravity_mass',
            lambda mass, disk_mass, accreted_mass: mass + disk_mass + accreted_mass)

        self.dust_model = dust_model


    def compute_epe_rate (self, disk):

        host_star = self.star_particles[disk.host_star_id]

        if host_star.star_ejected:

            epe_rate = 1e-10 | units.MSun/units.yr

        else:

            F = self.compute_rad_field(disk)

            disk.fuv_ambient_flux = F

            if F <= 0. | G0:
                epe_rate = 1e-10 | units.MSun/units.yr
            else:
                epe_rate = self.interpolator.interp_amuse(
                        disk.central_mass, disk.fuv_ambient_flux, 
                        disk.disk_gas_mass, disk.disk_radius)

        return epe_rate


    def compute_rad_field_from_radtrans (self, disk):

        host_star = self.star_particles[disk.host_star_id]

        i,j,k,m,n = self.grid_hydro.get_index_of_position(host_star.x, host_star.y,
            host_star.z)

        F = self.grid_hydro.get_grid_flux_photoelectric(i,j,k,m,n)

        return F


    def compute_rad_field_from_stars (self, disk):

        host_star = self.star_particles[disk.host_star_id]

        F = 0. | G0

        if len(self.radiative_stars) > 0:

            for i in range(len(self.radiative_stars)):

                if self.radiative_stars[i].fuv_luminosity > 0. | units.LSun and \
                        not self.radiative_stars[i].star_ejected:

                    R = (host_star.position - self.radiative_stars[i].position) \
                        .length()

                    dF = self.radiative_stars[i].fuv_luminosity/(4.*np.pi*R*R)

                    if self.sph_hydro is not None:
                        tau = optical_depth_between_points(self.sph_hydro,
                            host_star.position, self.radiative_stars[i].position, 
                            kappa)
                        dF *= np.exp(-tau)

                    F += dF

        return F


    def evolve_model (self, end_time):

        mask_active = [ disk is not None and disk.disk_active == True \
            for disk in self.disks ]
        disk_ids = np.arange(len(self.disks))[mask_active]

        epe_rate = [ self.compute_epe_rate(self.disks[disk_id])[0].\
            value_in(units.MSun/units.yr) \
            for disk_id in disk_ids ] | units.MSun/units.yr

        self.evolve_model_adaptive(disk_ids, end_time - self.model_time,
            epe_rate)

        self.copy_from_disks()
        self.star_particles.radius = 0.02 | units.pc

        self.model_time = end_time

        '''
        dt = end_time - self.model_time

        active_disks = []

        for disk in self.disks:
            if disk is not None and disk.disk_active:

                disk.outer_photoevap_rate = self.compute_epe_rate(disk)
                # Internal photoevaporation rate        
                disk.viscous.set_parameter(0, 
                    disk.internal_photoevap_flag * \
                    disk.inner_photoevap_rate.value_in(units.g/units.s))
                # External photoevaporation rate
                disk.viscous.set_parameter(1, 
                    disk.external_photoevap_flag * \
                    disk.outer_photoevap_rate.value_in(units.g/units.s))

                disk.ipe_mass_loss += disk.internal_photoevap_flag * \
                    disk.inner_photoevap_rate * dt
                disk.epe_mass_loss += disk.external_photoevap_flag * \
                    disk.outer_photoevap_rate * dt

                disk.viscous.update_keplerian_grid(
                    #self.star_particles[disk.host_star_id].mass)
                    disk.central_mass)
                disk.channel_to_code.copy()
                active_disks.append(disk)

        print ("[PPDA] Running {a}/{b} disks".format(a=len(active_disks),
            b=len(self.disked_stars)))

        if len(active_disks) == 0:
            self.model_time = end_time
            return

        pool = []

        for disk in active_disks:
            pool.append( disk.viscous.evolve_model.asynchronous(
                disk.viscous.model_time + dt/2.) )


        for i in range(len(pool)):
            final_mass = active_disks[i].disk_gas_mass - active_disks[i].outer_photoevap_rate*dt/2.
            tau = active_disks[i].disk_gas_mass/active_disks[i].outer_photoevap_rate
            try:
                pool[i].wait()
            except:
                print("[PPDA] Absolute convergence failure at {a} Myr".format(
                    a=active_disks[i].model_time.value_in(units.Myr)), flush=True)
                print (final_mass.value_in(units.MSun), tau.value_in(units.kyr))
                plt.loglog(
                active_disks[i].viscous.grid.r.value_in(units.au),
                active_disks[i].viscous.grid.column_density.value_in(units.g/units.cm**2)
                )
                #plt.show()
                active_disks[i].disk_convergence_failure = True



        for disk in active_disks:

            disk.model_time += dt/2.
            disk.channel_from_code.copy()

            if disk.disk_gas_mass < 0.00008 | units.MSun:
                disk.disk_dispersed = True
                print ('[PPDA] Disk dispersal at {a} Myr'.format(
                    a=disk.model_time.value_in(units.Myr)))

            disk.accreted_mass = -disk.viscous.inner_boundary_mass_out


            if self.dust_model == 'Haworth2018':
                # Remove dust in a leapfrog-like integration
                # Follows the prescription of Haworth et al. 2018 (MNRAS 475)

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
                disk.disk_dust_mass -= dM_dust

                # Can't have negative mass
                if  disk.disk_dust_mass < 0. | units.MSun:
                    disk.disk_dust_mass = 0. | units.MSun



        active_disks = []

        for disk in self.disks:
            if disk is not None and disk.disk_active:
                active_disks.append(disk)


        pool = []

        for disk in active_disks:
            if disk.disk_active:
                pool.append( disk.viscous.evolve_model.asynchronous(
                    disk.viscous.model_time + dt/2.) )


        for i in range(len(pool)):
            final_mass = active_disks[i].disk_gas_mass - active_disks[i].outer_photoevap_rate*dt/2.
            tau = active_disks[i].disk_gas_mass/active_disks[i].outer_photoevap_rate
            try:
                pool[i].wait()
            except:
                print("[PPDA] Absolute convergence failure at {a} Myr".format(
                    a=active_disks[i].model_time.value_in(units.Myr)), flush=True)
                print (final_mass.value_in(units.MSun), tau.value_in(units.kyr))
                plt.loglog(
                active_disks[i].viscous.grid.r.value_in(units.au),
                active_disks[i].viscous.grid.column_density.value_in(units.g/units.cm**2)
                )
                #plt.show()
                active_disks[i].disk_convergence_failure = True


        for disk in active_disks:

            disk.model_time += dt/2.
            disk.channel_from_code.copy()

            if disk.disk_gas_mass < 0.00008 | units.MSun:
                disk.disk_dispersed = True
                print ('[PPDA] Disk dispersal at {a} Myr'.format(
                    a=disk.model_time.value_in(units.Myr)))

            disk.accreted_mass = -disk.viscous.inner_boundary_mass_out

        self.copy_from_disks()
        self.star_particles.radius = 0.02 | units.pc

        self.model_time = end_time
        '''


    def evolve_model_adaptive (self, disk_ids, dt, epe_rate):

        print ("[PPDA] Evolving {a}/{b} disks for {c} kyr".format(
            a=len(disk_ids), b=len(self.disked_stars), c=dt.value_in(units.kyr)))

        if len(disk_ids) == 0:
            return

        disk_mass = [ self.disks[disk_id].disk_gas_mass.value_in(units.MSun) \
            for disk_id in disk_ids ] | units.MSun

        tau = self.eta * disk_mass / epe_rate

        mask_subcycle = tau < dt

        if np.sum(mask_subcycle):
            # Subcycle rapidly evaporating disks
            self.evolve_model_adaptive(disk_ids[ mask_subcycle ], dt/2., 
                epe_rate[ mask_subcycle ])

            mask_still_active = [ self.disks[disk_id].disk_active == True \
                for disk_id in disk_ids[mask_subcycle] ]
            subcycle_epe_rate = [ self.compute_epe_rate(self.disks[disk_id])[0].\
                value_in(units.MSun/units.yr) \
                for disk_id in disk_ids[mask_subcycle][mask_still_active] ] \
                | units.MSun/units.yr
            self.evolve_model_adaptive(
                disk_ids[ mask_subcycle ][mask_still_active], dt/2., 
                subcycle_epe_rate)


        for i in range(len(disk_ids)):
          if not mask_subcycle[i]:

            disk = self.disks[disk_ids[i]]

            disk.outer_photoevap_rate = epe_rate[i]
            # Internal photoevaporation rate        
            disk.viscous.set_parameter(0, 
                disk.internal_photoevap_flag * \
                disk.inner_photoevap_rate.value_in(units.g/units.s))
            # External photoevaporation rate
            disk.viscous.set_parameter(1, 
                disk.external_photoevap_flag * \
                disk.outer_photoevap_rate.value_in(units.g/units.s))

            disk.ipe_mass_loss += disk.internal_photoevap_flag * \
                disk.inner_photoevap_rate * dt
            disk.epe_mass_loss += disk.external_photoevap_flag * \
                disk.outer_photoevap_rate * dt

            disk.viscous.update_keplerian_grid(
                disk.central_mass)
            disk.channel_to_code.copy()


        pool = []; in_pool = [];
        for i in range(len(disk_ids)):
          disk = self.disks[disk_ids[i]]
          if not mask_subcycle[i] and disk.disk_active:
            pool.append( disk.viscous.evolve_model.asynchronous(
                disk.viscous.model_time + dt/2.) )
            in_pool.append(i)

        for i in range(len(pool)):
            try:
                pool[i].wait()
            except:
                print ("[PPDA] Absolute convergence failure at {a} Myr".format(
                    a=self.disks[disk_ids[in_pool[i]]].viscous.model_time.value_in(
                        units.Myr)), flush=True)
                self.disks[disk_ids[in_pool[i]]].disk_convergence_failure = True


        for i in range(len(disk_ids)):
          disk = self.disks[disk_ids[i]]
          if not mask_subcycle[i] and disk.disk_active:
            disk.model_time += dt/2.
            disk.channel_from_code.copy()

            if disk.disk_gas_mass < 0.00008 | units.MSun:
                disk.disk_dispersed = True
                print ('[PPDA] Disk dispersal at {a} Myr'.format(
                    a=disk.model_time.value_in(units.Myr)))

            disk.accreted_mass = -disk.viscous.inner_boundary_mass_out


            if self.dust_model == 'Haworth2018':
                self.evolve_Haworth2018_dust_model(disk, dt)


        pool = []; in_pool = [];
        for i in range(len(disk_ids)):
          disk = self.disks[disk_ids[i]]
          if not mask_subcycle[i] and disk.disk_active:
            pool.append( disk.viscous.evolve_model.asynchronous(
                disk.viscous.model_time + dt/2.) )
            in_pool.append(i)

        for i in range(len(pool)):
            try:
                pool[i].wait()
            except:
                print ("[PPDA] Absolute convergence failure at {a} Myr".format(
                    a=self.disks[disk_ids[in_pool[i]]].viscous.model_time.value_in(
                        units.Myr)), flush=True)
                self.disks[disk_ids[in_pool[i]]].disk_convergence_failure = True

        for i in range(len(disk_ids)):
          disk = self.disks[disk_ids[i]]
          if not mask_subcycle[i] and disk.disk_active:
            disk.model_time += dt/2.
            disk.channel_from_code.copy()

            if disk.disk_gas_mass < 0.00008 | units.MSun:
                disk.disk_dispersed = True
                print ('[PPDA] Disk dispersal at {a} Myr'.format(
                    a=disk.model_time.value_in(units.Myr)))

            disk.accreted_mass = -disk.viscous.inner_boundary_mass_out


    def evolve_Haworth2018_dust_model (self, disk, dt):
        # Remove dust in a leapfrog-like integration
        # Follows the prescription of Haworth et al. 2018 (MNRAS 475)

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
        disk.disk_dust_mass -= dM_dust

        # Can't have negative mass
        if  disk.disk_dust_mass < 0. | units.MSun:
            disk.disk_dust_mass = 0. | units.MSun


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
            print ("[PPDA] No collision detector set!")
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

                host_star_id = start + i

                viscous = self.setup_viscous_code(host_star_id)

                new_disk = disk_class.Disk(
                    initial_disk_radius(new_star_particles.mass[i], 
                        max_frac=self.max_frac), 
                    initial_disk_mass(new_star_particles.mass[i],
                        max_frac=self.max_frac),
                    new_star_particles[i].mass, viscous.grid,
                    self._params['alpha'])

                new_disk.host_star_id = host_star_id
                new_disk.truncation_mass_loss = 0. | units.MSun

                # Outer density in g/cm^2
                viscous.set_parameter(2, 1E-12)
                # Disk midplane temperature at 1 AU in K
                viscous.set_parameter(3, new_disk.Tm.value_in(units.K))
                # Mean particle mass in g
                viscous.set_parameter(4,
                    (self._params['mu']*1.008*constants.u).value_in(units.g))
                # Disk accretion rate
                if self._params['vader_mode'] == 'pedisk':
                    viscous.set_parameter(5,
                        new_disk.accretion_rate.value_in(units.g/units.s))
                    viscous.parameters.inner_pressure_boundary_type = 1
                    viscous.parameters.inner_boundary_function = True
                    viscous.parameters.inner_pressure_boundary_torque = \
                        0. | units.g*units.cm**2./units.s**2.
                elif self._params['vader_mode'] == 'pedisk_nataccr':
                    # Disk alpha viscosity
                    viscous.set_parameter(5, self._params['alpha'])
                # Stellar mass in MSun
                viscous.set_parameter(6,
                    self.star_particles[host_star_id].mass.value_in(units.MSun))

                new_disk.viscous = viscous

                new_disk.channel_from_code = viscous.grid.new_channel_to(
                    new_disk.grid)
                new_disk.channel_to_code = new_disk.grid.new_channel_to(
                    viscous.grid)

                self.disks.append(new_disk)

                self.star_particles[host_star_id].disk_key = host_star_id
                self.star_particles[host_star_id].star_ejected = False
                

            else:

                self.disks.append(None)

                self.star_particles[start+i].disk_key = -1

        self.copy_from_disks()


    def setup_viscous_code (self, host_star_id):

        viscous = Vader(mode=self._params['vader_mode'], redirection='none')
        viscous.initialize_keplerian_grid(self._params['n_cells'], False,
            self._params['r_min'], self._params['r_max'], 
            self.star_particles[host_star_id].mass)

        viscous.parameters.alpha = self._params['alpha']
        viscous.parameters.post_timestep_function = True
        viscous.parameters.maximum_tolerated_change = 1E99
        viscous.parameters.number_of_user_parameters = 7

        return viscous


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

        disk_class.stop_codes(
            [ disk.viscous for disk in self.disks if disk is not None ])


    def write_particles (self, filepath='./', label='', overwrite=True):

        if len(self.star_particles) > 0:
            write_set_to_file(self.star_particles, 
                filepath+'/viscous_particles_{b}i{a:05}.hdf5'.format(
                    a=self.output_counter, b=label), 'hdf5',
                    timestamp=self.model_time, overwrite_file=overwrite)
        else:
            print ("[PPDA] No particles to write!")


    def write_grids (self, filepath='./', label='', overwrite=True):

        N_disks = len(self.disked_stars)

        if N_disks == 0:
            print ("[PPDA] No disks to write!")
            return

        N_grid = len(self.disks[self.disked_stars[0].disk_key].grid.r)

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
        fried_folder='./', sph_hydro=None, grid_hydro=None, 
        label='', extra_attributes=[], vader_mode='pedisk_nataccr'):

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


    ppd_code = PPDPopulationAsync(alpha=alpha, mu=mu, number_of_cells=n_cells,
            r_min=r_min, r_max=r_max, fried_folder=fried_folder, 
            sph_hydro=sph_hydro, grid_hydro=grid_hydro, vader_mode=vader_mode)


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

            viscous = ppd_code.setup_viscous_code(i)

            disk = disk_class.Disk(100.|units.AU, 0.1|units.MSun,
                star_particles.initial_mass[i], viscous.grid,
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

            disk.channel_from_code = viscous.grid.new_channel_to(disk.grid)
            disk.channel_to_code =   disk.grid.new_channel_to(viscous.grid)

            # Outer density in g/cm^2
            viscous.set_parameter(2, 1E-12)
            # Disk midplane temperature at 1 AU in K
            viscous.set_parameter(3, disk.Tm.value_in(units.K))
            # Mean particle mass in g
            viscous.set_parameter(4, (mu*1.008*constants.u).value_in(units.g))
            # Disk accretion rate
            if vader_mode == 'pedisk':
                viscous.set_parameter(5,
                    new_disk.accretion_rate.value_in(units.g/units.s))
                viscous.parameters.inner_pressure_boundary_type = 1
                viscous.parameters.inner_boundary_function = True
                viscous.parameters.inner_pressure_boundary_torque = \
                    0. | units.g*units.cm**2./units.s**2.
            elif vader_mode == 'pedisk_nataccr':
                # Disk alpha viscosity
                viscous.set_parameter(5, alpha)
            # Stellar mass in MSun
            viscous.set_parameter(6,
                star_particles[i].mass.value_in(units.MSun))

            disk.viscous = viscous

            ppd_code.disks.append(disk)

            disk_counter += 1

    ppd_code.copy_from_disks()

    ppd_code.output_counter = input_counter + 1

    return ppd_code

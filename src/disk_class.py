import numpy as np

import queue
import threading
import multiprocessing
import time

from amuse.community.vader.interface import Vader
from amuse.units import units, constants

import FRIED_interp


code_queue = queue.Queue()

G0 = 1.6e-3 * units.erg / units.s / units.cm**2

class Disk:

    def __init__(self, disk_radius, disk_gas_mass, central_mass, 
                       grid, alpha,
                       mu=2.33, Tm=None, delta=1e-2, rho_g=1.|units.g/units.cm**3, a_min=1e-8|units.m,
                       internal_photoevap_flag=True, external_photoevap_flag=True, critical_radius=None):

        self.model_time = 0. | units.Myr

        self.disk_dispersed           = False   # Disk is dispersed if the mass is lower than some value (1% of 0.08 MSun)
        self.disk_convergence_failure = False   # Viscous code can fail to converge, catch and do not involve further
        self.disk_active              = True    # Disk is only evolved if it is not dispersed or failed to evolve

        self.internal_photoevap_flag = internal_photoevap_flag
        self.external_photoevap_flag = external_photoevap_flag

        if Tm is None:
            Tm = (100. | units.K) * (central_mass.value_in(units.MSun))**(1./4.)    # Tm ~ L^1/4 ~ (M^3)^1/4
        self.Tm = Tm                                # Midplane temperature at 1 AU
        self.mu = mu                                # Mean molecular mass, in hydrogen masses
        T = Tm/np.sqrt(grid.r.value_in(units.AU))   # Disk midplane temperature at 1 AU is Tm, T(r)~r^-1/2

        # Compute viscous timescale, used for modulating accretion rate
        if critical_radius is None:
            R1 = disk_radius
        else:
            R1 = critical_radius
        nu = alpha * constants.kB/constants.u * self.Tm/np.sqrt(R1.value_in(units.AU)) * (R1**3/(constants.G*central_mass))**0.5
        self.t_viscous = R1*R1/(3.*nu)

        self.grid = grid.copy()
        self.grid.column_density = self.column_density(disk_radius, disk_gas_mass, rc=critical_radius)
        self.grid.pressure = self.grid.column_density * constants.kB*T / (mu*1.008*constants.u) # Ideal gas law

        self.central_mass = central_mass        # Mass of host star
        self.accreted_mass = 0. | units.MSun    # Mass accreted from disk (still important for gravity)

        self.delta = delta                      # (dust mass)/(gas mass)
        self.rho_g = rho_g                      # density of individual dust grains
        self.a_min = a_min                      # minimum dust grain size

        self.disk_dust_mass = self.delta * self.disk_gas_mass

        self.outer_photoevap_rate = 0. | units.MSun/units.yr


    def evolve_disk_for(self, dt):
        '''
        Evolve a protoplanetary disk for a time step. Gas evolution is done through VADER, dust evaporation through
        the proscription of Haworth et al. 2018 (MNRAS 475). Note that before calling this function, a VADER code must
        be assigned to the 'viscous' property.

        dt: time step to evolve the disk for (scalar, units of time)
        '''

        # Adjust rotation curves to current central mass
        self.viscous.update_keplerian_grid(self.central_mass)

        # Specified mass flux, using VADER function
        self.viscous.parameters.inner_pressure_boundary_type = 1
        self.viscous.parameters.inner_boundary_function = True

        self.viscous.set_parameter(0, 
            self.internal_photoevap_flag * self.inner_photoevap_rate.value_in(units.g/units.s)) # Internal photoevaporation rate        
        self.viscous.set_parameter(1, 
            self.external_photoevap_flag * self.outer_photoevap_rate.value_in(units.g/units.s)) # External photoevaporation rate
        self.viscous.set_parameter(3, self.Tm.value_in(units.K))                                # Disk midplane temperature at 1 AU in K
        self.viscous.set_parameter(5, self.accretion_rate.value_in(units.g/units.s))            # Nominal accretion rate
        self.viscous.set_parameter(6, self.central_mass.value_in(units.MSun))                   # Stellar mass in MSun

        target_gas_mass = self.disk_gas_mass - ( self.internal_photoevap_flag * self.inner_photoevap_rate + self.external_photoevap_flag * self.outer_photoevap_rate + self.accretion_rate )*dt

        initial_accreted_mass = -self.viscous.inner_boundary_mass_out   # As codes are re-used, need to remember initial state

        # Channels to efficiently transfer data to and from code
        ch_fram_to_visc = self.grid.new_channel_to(self.viscous.grid) # class to code
        ch_visc_to_fram = self.viscous.grid.new_channel_to(self.grid) # code to class

        # Copy disk data to code
        ch_fram_to_visc.copy()





        # Gas and dust evaporation are coupled in a leapfrog-like method
        # (half step gas, full step dust, half step gas)
        try:
            self.viscous.evolve_model( self.viscous.model_time + dt/2. )

        except:
            print ("Partial convergence failure at {a} Myr".format(a=self.model_time.value_in(units.Myr)), flush=True)
            # Failure is often due to excessive accretion, so switch to zero-torque and restart
            self.viscous.parameters.inner_pressure_boundary_type = 3
            self.viscous.parameters.inner_boundary_function = False

            initial_accreted_mass = -self.viscous.inner_boundary_mass_out

            ch_fram_to_visc.copy()

            try:
                self.viscous.evolve_model( self.viscous.model_time + dt/2. )

            except:
                # If still fails, give up hope
                print ("Absolute convergence failure at {a} Myr".format(a=self.model_time.value_in(units.Myr)), flush=True)
                self.disk_convergence_failure = True


        self.model_time += dt/2.

        # Copy disk data to class 
        ch_visc_to_fram.copy()

        # Lower limit of disk masses is 1% of the mass of a 10% mass ratio disk around a 0.08 MSun star
        # About 27 MEarth, and 0.08 MJupiter
        if self.disk_gas_mass < 0.00008 | units.MSun:
            self.disk_dispersed = True
            print ('Disk dispersal at {a} Myr'.format(a=self.model_time.value_in(units.Myr)))

        # Keep track of mass accreted from the disk, as in the code this is the sum of all past mass accretions (including from other disks)
        if self.disk_convergence_failure == False:
            self.accreted_mass += -self.viscous.inner_boundary_mass_out - initial_accreted_mass
            initial_accreted_mass = -self.viscous.inner_boundary_mass_out

        # Flag to decide whether or not to evolve the disk
        self.disk_active = (not self.disk_dispersed)*(not self.disk_convergence_failure)





        # Remove dust in a leapfrog-like integration
        # Follows the prescription of Haworth et al. 2018 (MNRAS 475)

        # Thermal speed of particles
        v_th = (8.*constants.kB*self.Tm/np.sqrt(self.disk_radius.value_in(units.AU))/(np.pi * self.mu*1.008*constants.u))**(1./2.)
        # Disk scale height at disk edge
        Hd = (constants.kB*self.Tm*(1.|units.AU)**(1./2.)*self.disk_radius**(5./2.)/(self.mu*1.008*constants.u*self.central_mass*constants.G))**(1./2.)
        # Disk filling factor of sphere at disk edge
        F = Hd/(Hd**2 + self.disk_radius**2)**(1./2.)

        self.dust_photoevap_rate = self.external_photoevap_flag * self.delta * self.outer_photoevap_rate**(3./2.) * \
            (v_th/( 4.*np.pi * F * constants.G*self.central_mass * self.rho_g*self.a_min ))**(1./2.) * \
            np.exp( -self.delta*(constants.G*self.central_mass)**(1./2.)*self.model_time/(2.*self.disk_radius**(3./2.)) )

        # Can't entrain more dust than is available
        if  self.dust_photoevap_rate > self.delta*self.outer_photoevap_rate:
            self.dust_photoevap_rate = self.delta*self.outer_photoevap_rate

        # Eulerian integration
        dM_dust = self.dust_photoevap_rate * dt
        if self.disk_dispersed: # If disk is dispersed, do only half a step
            dM_dust /= 2.
        self.disk_dust_mass -= dM_dust

        # Can't have negative mass
        if  self.disk_dust_mass < 0. | units.MSun:
            self.disk_dust_mass = 0. | units.MSun

        if not self.disk_active:
            return





        # Back to fixed accretion rate, has potentially switched above
        self.viscous.parameters.inner_pressure_boundary_type = 1
        self.viscous.parameters.inner_boundary_function = True

        try:
            self.viscous.evolve_model( self.viscous.model_time + dt/2. )

        except:
            print ("Partial convergence failure at {a} Myr".format(a=self.model_time.value_in(units.Myr)), flush=True)
            self.viscous.parameters.inner_pressure_boundary_type = 3
            self.viscous.parameters.inner_boundary_function = False

            initial_accreted_mass = -self.viscous.inner_boundary_mass_out

            ch_fram_to_visc.copy()

            try: 
                self.viscous.evolve_model( self.viscous.model_time + dt/2. )

            except:
                print ("Absolute convergence failure at {a} Myr".format(a=self.model_time.value_in(units.Myr)), flush=True)
                self.disk_convergence_failure = True


        self.model_time += dt/2.

        ch_visc_to_fram.copy()

        if self.disk_gas_mass < 0.00008 | units.MSun:
            self.disk_dispersed = True
            print ('Disk dispersal at {a} Myr'.format(a=self.model_time.value_in(units.Myr)))

        if self.disk_convergence_failure == False:
            self.accreted_mass += -self.viscous.inner_boundary_mass_out - initial_accreted_mass

        self.disk_active = (not self.disk_dispersed)*(not self.disk_convergence_failure)

        # Relative error in disk mass after step, compared to prescribed change rates
        # Causes of error can be choked accretion (potentially big) and numerical errors in 
        # internal photoevaporation (~1% or less)
        self.mass_error = np.abs((self.disk_gas_mass - target_gas_mass)/self.disk_gas_mass)


    def column_density(self,
                       rd,
                       disk_gas_mass,
                       lower_density=1E-12 | units.g / units.cm**2,
                       rc=None):
        '''
        Sets up a Lynden-Bell & Pringle 1974 disk profile

        rc: Scale length of disk profile                                                (scalar, units of length)
        disk_gass_mass: target disk gas mass                                            (scalar, units of mass)
        lower_density: minimum surface density of disk, as VADER can't handle 0 density (scalar, units of mass per surface)
        rd: Disk cutoff length                                                          (scalar, units of length)

        returns the surface density at positions defined on the grid                    (vector, units of mass per surface)
        '''

        # If no cutoff is specified, the scale length is used
        # Following Anderson et al. 2013
        if rc is None:
            rc = rd

        r = self.grid.r.copy()

        Sigma_0 = disk_gas_mass/( 2.*np.pi * rc**2 * (1. - np.exp(-rd/rc)))
        Sigma = Sigma_0 * (rc/r) * np.exp(-r/rc) * (r <= rd) + lower_density

        return Sigma


    @property
    def accretion_rate(self):
        '''
        Mass-dependent accretion rate of T-Tauri stars according to Alcala et al. 2014
        Modulated by viscous timescale (following Lynden-Bell & Pringle 1974)
        '''
        return 10.**( 1.81*np.log10(self.central_mass.value_in(units.MSun)) - 8.25 ) \
            * (1. + self.model_time/self.t_viscous)**(-3./2.) | units.MSun / units.yr
        #return 6.4e-11 | units.MSun / units.yr

    @property
    def inner_photoevap_rate(self):
        '''
        Internal photoevaporation rate of protoplanetary disks from Picogna et al. 2019, with mass scaling following Owen et al. 2012
        '''
        Lx = self.xray_luminosity.value_in( units.erg / units.s )
        return 10.**( -2.7326*np.exp( -( np.log(np.log10( Lx )) - 3.3307 )**2/2.9868e-3 ) - 7.2580) \
                  * (self.central_mass/(0.7 | units.MSun))**-0.068 | units.MSun / units.yr

    @property
    def xray_luminosity(self):
        '''
        Mass-dependent X-ray luminosity of classical T-Tauri stars according to Flaccomio et al. 2012 (typical luminosities)
        '''
        return 10.**( 1.7*np.log10(self.central_mass.value_in(units.MSun)) + 30.  ) | units.erg / units.s

    @property
    def disk_radius(self, f=0.999):
        '''
        Gas radius of the disk, defined as the radius within which a fraction f of the total mass is contained

        f: fraction of mass within disk radius  (float)

        returns the disk radius                 (scalar, units of length)
        '''

        Mtot = (self.grid.area*self.grid.column_density).sum()
        Mcum = 0. | units.MSun

        edge = -1

        for i in range(len(self.grid.r)):

            Mcum += self.grid.area[i]*self.grid.column_density[i]
            
            if Mcum >= Mtot*f:
                edge = i
                break

        return self.grid.r[edge]

    @property
    def disk_gas_mass(self):
        '''
        Gas mass of disk (defined as total mass on VADER grid)
        '''
        return (self.grid.area*self.grid.column_density).sum()

    @property
    def disk_mass(self):
        '''
        Total mass of disk
        '''
        return self.disk_dust_mass + self.disk_gas_mass

    @property
    def disk_sigma_edge(self):
        '''
        Column density at disk radius
        '''
        return self.grid.column_density[np.argmax( self.disk_radius == self.grid.r )]


def setup_disks_and_codes(disk_radii, disk_masses, stellar_masses, number_of_vaders, 
        number_of_cells, r_min, r_max, alpha, critical_radii=None,
        mu=2.33, Tm=None, IPE=True, EPE=True):
    '''
    Setup up a number of disk objects and VADER integrators
    This is done in the same function as they need to exactly share grids

    disk_radii:       initial gas radius of each protoplanetary disk (vector shape (N), units of length)
    disk_masses:      initial gas masses of each protoplanetary disk (vector shape (N), units of mass)
    stellar_masses:   initial mass of host star of each protoplanetary disk (vector shape (N), units of mass)
    number_of_vaders: number of VADER integrators to initialize (integer)
    number_of_cells:  number of cells in grids of VADER integrators (integer)
    r_min:            inner edge of VADER grids (scalar, units of length)
    r_max:            inner edge of VADER grids (scalar, units of length)
    alpha:            dimensionless viscosity parameter (float)
    mu:               mean molecular mass of disk gas (float)
    Tm:               Disk midplane temperature at 1 AU (scalar, units of temperature)
    IPE:              (I)nternal (P)hoto(E)vaporation flag (bool)
    EPE:              (E)xternal (P)hoto(E)vaporation flag (bool)    
    '''

    viscous_codes = [ Vader(mode='pedisk', redirection='none') for _ in range(number_of_vaders) ]

    for i in range(number_of_vaders):

        viscous_codes[i].initialize_code()
        viscous_codes[i].initialize_keplerian_grid(number_of_cells, False, r_min, r_max, 1. | units.MSun)

        viscous_codes[i].parameters.alpha = alpha
        viscous_codes[i].parameters.post_timestep_function = True
        viscous_codes[i].parameters.maximum_tolerated_change = 1E99
        viscous_codes[i].parameters.number_of_user_parameters = 7
        viscous_codes[i].parameters.inner_pressure_boundary_torque = 0. | units.g*units.cm**2./units.s**2.

        viscous_codes[i].set_parameter(2, 1E-12)                                        # Outer density in g/cm^2
        viscous_codes[i].set_parameter(4, (mu*1.008*constants.u).value_in(units.g))     # Mean particle mass in g


    number_of_disks = len(stellar_masses)

    if critical_radii is None:
        critical_radii = [None]*number_of_disks
    if Tm is None:
        Tm = [None]*number_of_disks

    disks = [ Disk(disk_radii[i], disk_masses[i], stellar_masses[i], viscous_codes[0].grid, alpha, mu=mu, Tm=Tm[i], 
                   internal_photoevap_flag=IPE, external_photoevap_flag=EPE, critical_radius=critical_radii[i]) \
                        for i in range(number_of_disks) ]

    return viscous_codes, disks


def run_disks(viscous_codes, disks, dt):
    '''
    Evolve a set of disks for a time step, using a set of viscous codes

    viscous_codes: list of VADER codes
    disks: list of disk objects
    dt: time step to evolve for (scalar, units of time)
    '''

    # Distribute the disks evenly over the available codes
    disks_pooled = pool_disks(disks, len(viscous_codes))

    # Assign each group of disks to a process and start each process
    for i in range(len(viscous_codes)):

        code_queue.put({'disks': disks_pooled[i], 'viscous': viscous_codes[i], 'dt': dt})
        viscous_thread = threading.Thread(target=remote_worker_code)
        viscous_thread.daemon = True

        try:
	        viscous_thread.start()
        except:
	        print ("Thread could not be started; currently {a} threads are active".format(a=threading.active_count()), flush=True)

    # Wait for all threads to finish
    code_queue.join()


def pool_disks(disks, N_cores):
    '''
    Distribute a set of disks among a number of processes

    disks: list of disk objects to distribute
    N_cores: number of cores to distribute disks over (int)

    returns a list of list of disks, such that the disks are divided as evenly as possible
    '''

    N = len(disks)//N_cores
    n = len(disks)% N_cores

    disks_pooled = []

    MIN = 0
    counter = 0

    for i in range(N_cores):

        if counter < n:
            DIFF = N + 1
            counter += 1
        else:
            DIFF = N

        disks_pooled.append([])
        disks_pooled[i].extend( disks[MIN:MIN+DIFF] )

        MIN += DIFF

    return disks_pooled


def stop_codes(codes):
    '''
    Stop all codes in a list of codes
    '''

    for code in codes:
        code.stop()


def remote_worker_code():
    '''
    Worker function of each thread
    Receives a number of disks, a viscous code, and a time step, and evolves every disk for the time step using the code
    '''

    package = code_queue.get()

    disks = package['disks']
    code  = package['viscous']
    dt    = package['dt']

    for disk in disks:
        disk.viscous = code
        disk.evolve_disk_for(dt)

    code_queue.task_done()


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib.colors as clr
    import matplotlib.cm as cm

    import time




    #'''
    number_of_vaders = 4
    number_of_cells  = 330
    number_of_disks  = 8

    stellar_masses = [0.08]*8 | units.MSun
    disk_masses    = ([0.15*0.08]*4 + [0.35*0.08]*4) | units.MSun
    disk_radii     = [150.]*8 | units.AU
    critical_radii = [50.]*8 | units.AU
    Tm = [50.]*8 | units.K

    fluxes = [1e1, 1e2, 1e3, 1e4]*2 | G0


    r_min = 0.5  | units.AU
    r_max = 300. | units.AU

    alpha = 1E-3

    codes, disks = setup_disks_and_codes(disk_radii, disk_masses, stellar_masses, number_of_vaders, 
        number_of_cells, r_min, r_max, alpha, critical_radii=critical_radii, Tm=Tm)

    t     = 0.  | units.Myr
    t_end = 10.  | units.Myr
    dt    = 1.  | units.kyr

    N_steps = int(t_end/dt)

    interpolator = FRIED_interp.Haworth2018_interpolator(verbosity=False, folder='forMartijn/forMartijn/')


    disk_radii = np.zeros((N_steps, 8))
    dust_masses = np.zeros((N_steps, 8))
    gas_masses  = np.zeros((N_steps, 8))

    Mdot_gas = np.zeros((N_steps, 8))
    Mdot_dust = np.zeros((N_steps, 8))


    PLOT = True

    start2 = time.time()

    for i in range(N_steps):

        active_disks = []

        for j in range(number_of_disks):
            if disks[j].disk_active:
                disks[j].outer_photoevap_rate = interpolator.interp_amuse(fluxes[j], disks[j].disk_sigma_edge, disks[j].disk_radius)
                active_disks.append(disks[j])


        if len(active_disks) == 0:
            print ("No active disks. Terminating simulation.", flush=True)
            break


        #if (i+1)%20 == 0:
        start = time.time()

        run_disks(codes, active_disks, dt)

        t += dt

        #if (i+1)%20 == 0:
        end = time.time()
        print ("Running {c} disks for a step took {a} s, time is {b} Myr".format(a=end-start, b=t.value_in(units.Myr),
            c=len(active_disks)), flush=True)

        dust_masses[i] = [ disk.disk_dust_mass.value_in(units.MEarth) for disk in disks ]
        gas_masses[i]  = [ disk.disk_gas_mass.value_in(units.MEarth)  for disk in disks ]
        disk_radii[i]  = [ disk.disk_radius.value_in(units.AU)        for disk in disks ]
        Mdot_gas[i]    = [ disk.outer_photoevap_rate.value_in(units.MSun/units.yr) for disk in disks ]
        Mdot_dust[i]   = [ disk.dust_photoevap_rate.value_in(units.MSun/units.yr) for disk in disks ]



    end2 = time.time()

    print ('Evolved {a} disks for {b} Myr in {c} s'.format(a=number_of_disks, b=t_end.value_in(units.Myr), c=end2-start2))
    print ('Dispersed disks:', np.sum([ disk.disk_dispersed for disk in disks ]))
    print ('Convergence failures:', np.sum([ disk.disk_convergence_failure for disk in disks ]))


    stop_codes(codes)




    t_array = (np.arange(N_steps)+1)*dt.value_in(units.kyr)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(t_array, dust_masses[:,0:4])
    ax.plot(t_array, gas_masses[:,0:4], linestyle='--')

    ax.set_title('15% Stellar')
    ax.set_yscale('log')
    ax.set_xlabel('t [kyr]')
    ax.set_ylabel('Disk Mass [MEarth]')


    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(t_array, dust_masses[:,4:])
    ax.plot(t_array, gas_masses[:,4:], linestyle='--')

    ax.set_title('35% Stellar')
    ax.set_yscale('log')
    ax.set_xlabel('t [kyr]')
    ax.set_ylabel('Disk Mass [MEarth]')




    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(t_array, Mdot_gas[:,0], c='r')
    ax.plot(t_array, Mdot_dust[:,0], c='g')

    ax.set_title('10 G0')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('t [kyr]')
    ax.set_ylabel('Disk Mass Loss [MSun/yr]')



    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(t_array, Mdot_gas[:,3], c='r')
    ax.plot(t_array, Mdot_dust[:,3], c='g')

    ax.set_title('10^4 G0')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('t [kyr]')
    ax.set_ylabel('Disk Mass Loss [MSun/yr]')



    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(t_array, disk_radii[:,0], label='1e1 G0')
    ax.plot(t_array, disk_radii[:,3], label='1e4 G0')
    ax.set_title('Disk Radii')
    ax.set_xscale('log')
    ax.set_xlabel('t [kyr]')
    ax.set_ylabel('Disk Radius [AU]')



    plt.show()

    stop_codes(codes)
    #'''



    '''
    number_of_vaders = 2
    number_of_cells  = 330
    number_of_disks  = 2

    stellar_masses = np.ones(number_of_disks)*1. | units.MSun#np.logspace(np.log10(1.9), np.log10(1.9), num=number_of_disks) | units.MSun
    disk_masses    = 0.1 * stellar_masses / stellar_masses.value_in(units.MSun)**0.5
    disk_radii     = np.ones(number_of_disks)*150. | units.AU

    fluxes = [1000.]*10 | G0#[30., 100., 300., 1000., 3000.]*10 | G0


    r_min = 0.05  | units.AU
    r_max = 5000. | units.AU

    alpha = 5E-3

    mu = 2.3

    codes, disks = setup_disks_and_codes(disk_radii, disk_masses, stellar_masses, number_of_vaders, 
        number_of_cells, r_min, r_max, alpha, Tm=[120., 300.] | units.K)


    t     = 0.  | units.Myr
    t_end = 1000.  | units.kyr
    dt    = 1.  | units.kyr

    N_steps = int(t_end/dt)


    interpolator = FRIED_interp.FRIED_interpolator(verbosity=False)

    dust_masses = np.zeros((N_steps, number_of_disks))
    gas_masses  = np.zeros((N_steps, number_of_disks))
    accreted_mass = np.zeros((N_steps, number_of_disks))

    disk_radii  = np.zeros((N_steps, number_of_disks))

    mass_error = np.zeros((N_steps, number_of_disks))

    disk_profiles = np.zeros((number_of_cells, N_steps//20))
    counter = 0


    start2 = time.time()

    for i in range(N_steps):

        active_disks = []

        for j in range(number_of_disks):
            if disks[j].disk_active:
                disks[j].outer_photoevap_rate = interpolator.interp_amuse(disks[j].central_mass, fluxes[j], disks[j].disk_gas_mass, disks[j].disk_radius)
                active_disks.append(disks[j])


        if len(active_disks) == 0:
            print ("No active disks. Terminating simulation.", flush=True)
            break


        if (i+1)%20 == 0:
            start = time.time()

        run_disks(codes, active_disks, dt)

        t += dt

        if (i+1)%20 == 0:
            end = time.time()
            print ("Running {c} disks for a step took {a} s, time is {b} Myr".format(a=end-start, b=t.value_in(units.Myr),
                c=len(active_disks)), flush=True)
            disk_profiles[:,counter] = disks[0].grid.column_density.value_in(units.g/units.cm**2)
            counter += 1

        dust_masses[i] = [ disk.disk_dust_mass.value_in(units.MEarth) for disk in disks ]
        gas_masses[i]  = [ disk.disk_gas_mass.value_in(units.MEarth)  for disk in disks ]
        disk_radii[i]  = [ disk.disk_radius.value_in(units.AU)        for disk in disks ]
        accreted_mass[i] = [ disk.accreted_mass.value_in(units.MSun)  for disk in disks ]
        mass_error[i] =  [ disk.mass_error                            for disk in disks ]

    end2 = time.time()

    print ('Evolved {a} disks for {b} Myr in {c} s'.format(a=number_of_disks, b=t_end.value_in(units.Myr), c=end2-start2))
    print ('Dispersed disks:', np.sum([ disk.disk_dispersed for disk in disks ]))
    print ('Convergence failures:', np.sum([ disk.disk_convergence_failure for disk in disks ]))


    stop_codes(codes)



    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_yscale('log')

    ax.set_xlabel('t [kyr]')
    ax.set_ylabel('Mass [MEarth]')

    ax.set_title('Photoevaporation of Protoplanetary Disks (Solid is gas, Dotted is dust, color is Host Mass (0.08-1.9 MSun))')

    t_array = (np.arange(N_steps)+1)*dt.value_in(units.kyr)
    cmap = plt.get_cmap('jet')
    cmap_norm = clr.Normalize(vmin=0, vmax=number_of_disks)
    scalarmap = cm.ScalarMappable(norm=cmap_norm, cmap=cmap)

    for i in range(number_of_disks):

        ax.plot(t_array, gas_masses[:,i], linewidth=1, color=scalarmap.to_rgba(i))
        ax.plot(t_array, dust_masses[:,i], linewidth=1, linestyle=':', color=scalarmap.to_rgba(i))


    ax.axhline( (0.08*0.1*0.01 | units.MSun).value_in(units.MEarth), c='k' )
    ax.axhline( (0.08*0.1*0.01*0.01 | units.MSun).value_in(units.MEarth), c='k', linestyle=':')



    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('R [AU]')
    ax.set_ylabel('Sigma [g/cm$^{-2}$]')

    ax.set_title('Photoevaporated Disk Profiles after {a} Myr (Color is Host Mass (0.08-1.9 MSun))'.format(a=t_end.value_in(units.Myr)))

    for i in range(number_of_disks):

        ax.plot(disks[i].grid.r.value_in(units.AU), disks[i].grid.column_density.value_in(units.g/units.cm**2), color=scalarmap.to_rgba(i))



    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_yscale('log')

    ax.set_xlabel('t [kyr]')
    ax.set_ylabel('Gas Radius [AU]')

    ax.set_title('Disk Radius Evolution of Protoplanetary Disks (Color is Host Mass (0.08-1.9 MSun))')

    t_array = (np.arange(N_steps)+1)*dt.value_in(units.kyr)
    cmap = plt.get_cmap('jet')
    cmap_norm = clr.Normalize(vmin=0, vmax=number_of_disks)
    scalarmap = cm.ScalarMappable(norm=cmap_norm, cmap=cmap)

    for i in range(number_of_disks):

        ax.plot(t_array, disk_radii[:,i], linewidth=1, color=scalarmap.to_rgba(i))




    fig = plt.figure()
    ax = fig.add_subplot(111)

    #ax.set_yscale('log')

    ax.set_xlabel('t [kyr]')
    ax.set_ylabel('Accreted Mass [MSun]')

    ax.set_title('Cumulative Accreted Mass of Protoplanetary Disks (Color is Host Mass (0.08-1.9 MSun))')

    t_array = (np.arange(N_steps)+1)*dt.value_in(units.kyr)
    cmap = plt.get_cmap('jet')
    cmap_norm = clr.Normalize(vmin=0, vmax=number_of_disks)
    scalarmap = cm.ScalarMappable(norm=cmap_norm, cmap=cmap)

    for i in range(number_of_disks):

        ax.plot(t_array, accreted_mass[:,i], linewidth=1, color=scalarmap.to_rgba(i))




    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_yscale('log')

    ax.set_xlabel('t [kyr]')
    ax.set_ylabel('Relative Mass Error')

    ax.set_title('Relative Mass Error in Step (Color is Host Mass (0.08-1.9 MSun))')

    t_array = (np.arange(N_steps)+1)*dt.value_in(units.kyr)
    cmap = plt.get_cmap('jet')
    cmap_norm = clr.Normalize(vmin=0, vmax=number_of_disks)
    scalarmap = cm.ScalarMappable(norm=cmap_norm, cmap=cmap)

    for i in range(number_of_disks):

        ax.plot(t_array, mass_error[:,i], linewidth=1, color=scalarmap.to_rgba(i))


    print ([ disk.model_time.value_in(units.Myr) for disk in disks ])



    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('R [AU]')
    ax.set_ylabel('Sigma [g/cm$^{-2}$]')

    ax.set_title('Photoevaporated Disk Profiles after {a} Myr (Color is Host Mass (0.08-1.9 MSun))'.format(a=t_end.value_in(units.Myr)))

    cmap = plt.get_cmap('jet')
    cmap_norm = clr.Normalize(vmin=0, vmax=N_steps//20)
    scalarmap = cm.ScalarMappable(norm=cmap_norm, cmap=cmap)

    for i in range(N_steps//20):

        ax.plot(disks[0].grid.r.value_in(units.AU), disk_profiles[:,i], color=scalarmap.to_rgba(i))



    plt.show()

    stop_codes(codes)
    '''

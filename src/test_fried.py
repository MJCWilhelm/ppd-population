import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as cm

from amuse.units import units
from FRIED_interp import FRIED_interpolator


G0 = 1.6e-3 * units.erg/units.s/units.cm**2


if __name__ == '__main__':

    interp = FRIED_interpolator(folder='../data')


    Mstar_grid = 10.**np.unique(interp._grid[:,0])
    F_grid = 10.**np.unique(interp._grid[:,1])
    Rdisk_grid = 10.**np.unique(interp._grid[:,3])


    for i in range(len(Mstar_grid)):
        for j in range(len(Rdisk_grid)):
            mask = (10.**interp._grid[:,0] == Mstar_grid[i])*(10.**interp._grid[:,3] == Rdisk_grid[j])
            Mdisk_grid = 10.**interp._grid[mask,2]


            cmap = plt.get_cmap('plasma')
            cmap_norm = clr.Normalize(vmin=0, vmax=len(Mdisk_grid))
            scalarmap = cm.ScalarMappable(norm=cmap_norm, cmap=cmap)


            plt.figure()
            for k in range(len(Mdisk_grid)):
                x = np.ones(len(F_grid))
                plt.scatter(F_grid, interp.interp(Mstar_grid[i]*x, F_grid, Mdisk_grid[k]*x, Rdisk_grid[j]*x))#, c=scalarmap.to_rgba(k))


            plt.xscale('log')
            plt.yscale('log')

    plt.show()

    '''
    Mstar = 1. | units.MSun
    Rdisk = 100. | units.AU
    Mdisk = 10. | units.MJupiter

    F = [10., 100., 1000., 10000.] | G0
    N = len(F)

    Mstar = [1.]*N | units.MSun
    Rdisk = [50.]*N | units.AU
    Mdisk = [10.]*N | units.MJupiter


    Mdot = interp.interp_amuse(Mstar, F, Mdisk, Rdisk)


    plt.scatter(F.value_in(G0), Mdot.value_in(units.MSun/units.yr))

    plt.xscale('log')
    plt.yscale('log')

    plt.show()
    '''

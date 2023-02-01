import zeus
from scipy.stats import norm
import sklearn.gaussian_process as skg
import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import dask.dataframe as dd
import time
import cosmoprimo
from numba import njit, jit, numba
import math
from joblib import Parallel, delayed
import multiprocessing
import sys
import lhsmdu
import os
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from dask import dataframe as dd 
from pycorr import TwoPointCorrelationFunction, project_to_multipoles, project_to_wp
import fitsio


@njit(fastmath=True)
def HMQ(log10_Mh, Ac, Mc, sig_M, gamma, Q, pmax, *args):
    """
    --- HMQ HOD model from arXiv:1910.05095 (S. Alam 2019) FAUX add max 
    """
    phi_x = 1 / (np.sqrt(2 * np.pi) * sig_M) * np.exp(
            - (log10_Mh - Mc)**2 / (2 * sig_M**2))
    PHI_gamma_x = 0.5 * (1 + math.erf(gamma * (log10_Mh - Mc)
                                   / (sig_M*np.sqrt(2))))
    A = (pmax - 1/Q) / (2 * phi_x * PHI_gamma_x)
    return Ac * (2 * A * phi_x * PHI_gamma_x + 0.5 / Q * (1 + math.erf(
            (log10_Mh - Mc) / 0.01)))

@njit(fastmath=True)
def HMQ_Sandy(log10_Mh, Ac, Mc, sig_M, gamma, Q, pmax, *args):
    """
    --- HMQ HOD model modify by Sandy without the normalization
    """
    phi_x = 1 / (np.sqrt(2 * np.pi) * sig_M) * np.exp(
            - (log10_Mh - Mc)**2 / (2 * sig_M**2))
    PHI_gamma_x = 0.5 * (1 + math.erf(gamma * (log10_Mh - Mc)
                                   / (sig_M*np.sqrt(2))))
    A = (pmax - 1/Q)
    return Ac * (2 * A * phi_x * PHI_gamma_x + 0.5 / Q * (1 + math.erf(
            (log10_Mh - Mc) / 0.01)))

@njit(fastmath=True)
def mHMQ(log10_Mh, Ac, Mc, sig_M, gamma, Q, pmax, *args):
    """
    --- HMQ HOD model modify by Sandy without the normalization
    """
    phi_x = 1 / (np.sqrt(2 * np.pi) * sig_M) * np.exp(
            - (log10_Mh - Mc)**2 / (2 * sig_M**2))
    PHI_gamma_x = 0.5 * (1 + math.erf(gamma * (log10_Mh - Mc)
                                   / (sig_M*np.sqrt(2))))
    
    return Ac * pmax * 2 * phi_x * PHI_gamma_x
    
@njit(fastmath=True)
def Gaussian_fun(x, mean, sigma):
    """
    Gaussian function with centered at `mean' with standard deviation `sigma'.
    """
    return 0.3989422804014327/sigma*np.exp(-(x - mean)**2/2/sigma**2)


@njit(fastmath=True)
def GHOD(log10_Mh, Ac, Mc, sig_M, *args):
    """
    --- Gaussian HOD model from arXiv:1708.07628 (V. Gonzalez-Perez 2018)
    """
    return Ac / (np.sqrt(2 * np.pi) * sig_M) * np.exp(
        -(log10_Mh - Mc)**2 / (2 * sig_M**2))


@njit(fastmath=True)
def LNHOD(log10_Mh, Ac, Mc, sig_M, *args):
    """
    --- HOD model with lognormal distribution 
    """
    x = log10_Mh-Mc+1
    if x <= 0:
        return 0
    val = Ac * np.exp(-(np.log(x))**2 / (
          2 * sig_M**2)) / (x * sig_M * np.sqrt(2 * np.pi))
    #val[np.isnan(val)] = 0
    return val


@njit(fastmath=True)
def SFHOD(log10_Mh, Ac, Mc, sig_M, gamma, *args):
    """
    --- Star forming HOD model from arXiv:1708.07628 (V. Gonzalez-Perez 2018)
    """
    norm = Ac / (np.sqrt(2 * np.pi) * sig_M)
    if Mc >= log10_Mh:
        return norm * np.exp(- (log10_Mh-Mc)**2
                             / (2*sig_M**2))
    else:
        return norm * (10**log10_Mh/10**Mc)**gamma


@njit(fastmath=True)
def SHOD(log10_Mh, Ac, Mc, sig_M, *args):
    """
    --- Standard HOD model from arXiv:astro-ph/0408564 Zheng et al. (2007)
    """
    return Ac * 0.5 * (1 + math.erf((log10_Mh-Mc) / (sig_M)))


@njit(fastmath=True)
def compute_Nsat(log10_Mh, As, M_0, M_1, alpha):
    """
    ---  Standard Zheng et al. (2005) satellite HOD parametrization arXiv:astro-ph/0408564
    """
    N_sat = As * ((10**log10_Mh - 10**M_0) / 10**M_1)**alpha
    return N_sat



@njit(fastmath=True)
def f(x):
    '''
    --- Aiding function for NFW computation
    '''
    return np.log(1.+x)-x/(1.+x)


@njit(fastmath=True)
def reScale(a, b):
    '''
    --- Aiding function for NFW computation
    '''
    return a.transpose


@njit(parallel=True, fastmath=True)
def getPointsOnSphere(nPoints, Nthread, seed=None):
    '''
    --- Aiding function for NFW computation, generate random points in a sphere
    '''
    numba.set_num_threads(Nthread)
    ind = min(Nthread, nPoints)
    # starting index of each thread
    hstart = np.rint(np.linspace(0, nPoints, ind+1))
    ur = np.zeros((nPoints, 3), dtype=np.float64)    
    cmin = -1
    cmax = +1

    for tid in numba.prange(Nthread):
        if seed is not None:
            np.random.seed(seed[tid])
        for i in range(hstart[tid], hstart[tid + 1]):
            u1, u2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
            ra = 0 + u1*(2*np.pi-0)
            dec = np.pi - (np.arccos(cmin+u2*(cmax-cmin)))

            ur[i, 0] = np.sin(dec) * np.cos(ra)
            ur[i, 1] = np.sin(dec) * np.sin(ra)
            ur[i, 2] = np.cos(dec)
    return ur



def load_CompaSO(path_to_sim, usecols, mass_cut = None, verbose=True):
    """
    --- Function to load AbacusSummit halo catalogs
    """
    if verbose :
        start = time.time()
        print(f"Load Compaso cat from {path_to_sim} ...")
    if 'npoutA' in usecols:
        load_part = True
    else:
        load_part = False
    hcat_i = CompaSOHaloCatalog(f"{path_to_sim}", fields=usecols,  subsamples=dict(A=load_part),
                                cleaned=True)
    if mass_cut is not None:
        N = mass_cut/hcat_i.header['ParticleMassHMsun']
        if verbose :
            print(f"Done took", time.strftime("%H:%M:%S",time.gmtime(time.time() - start)), flush=True)
        return hcat_i.halos[hcat_i.halos['N'] > N], hcat_i.header
    else:
        if verbose :
            print(f"Done took", time.strftime("%H:%M:%S",time.gmtime(time.time() - start)), flush=True)
        return hcat_i.halos[hcat_i.halos['N'] > 0], hcat_i.header



@njit(parallel=True, fastmath=True)
def compute_col_from_Abacus(N, pos, vel, ParticleMassHMsun, 
                            x, y, z, vx, vy, vz, 
                            Mvir, log10_Mh, Nthread,
                            Rs=None, Rvir=None, c=None,
                            rvcirc_max=None, r50=None, r98=None, 
                            mean_density=None, SODensityL1=None):
    """
    --- Function to prepare colunms for the halo catalog 
    """
    
    numba.set_num_threads(Nthread)
    # starting index of each thread
    hstart = np.rint(np.linspace(0, len(N), Nthread + 1))
    # figuring out the number of halos kept for each thread

    for tid in numba.prange(Nthread):
        for i in range(int(hstart[tid]), int(hstart[tid + 1])):
                Mvir[i] = (N[i]*ParticleMassHMsun)
                log10_Mh[i] = np.log10(Mvir[i])
                x[i], y[i], z[i] = pos[i]
                vx[i], vy[i], vz[i] = vel[i]
                if rvcirc_max is not None:
                    Rvir[i] = 620.35049 * (Mvir[i] / (mean_density * SODensityL1))**(1/3) # 1000 * (3 / (4*np.pi))**(1/3)
                    Rs[i] = rvcirc_max[i]*462.96296   # 1000/2.16
                    c[i] = Rvir[i]/Rs[i]
                elif r50 is not None: 
                    Rs[i] = r50[i]*1000
                    Rvir[i] = r98[i]*1000
                    c[i] = r98[i]/r50[i]                  


def load_hcat_from_Abacus (path_to_sim, usecols, args, Lsuff, Nthread=64, mass_cut=None, verbose = True):
    """
    --- Function which returns Abacus halo catalog for HOD studies
    """
    hcat, header = load_CompaSO(path_to_sim, mass_cut=mass_cut, usecols=usecols)
    if verbose :
        start = time.time()
        print("Compute columns...")
        
    if args['use_particles']:
        dic = dict((col, np.empty(hcat['N'].size, dtype='float32')) 
                   for col in ['x', 'y','z','vx','vy','vz', 'Mvir', 'log10_Mh'])
        dic['npstartA'] = np.array(hcat['npstartA'])
        dic['npoutA'] = np.array(hcat['npoutA'])
        c_proxy = 'particles'
        compute_col_from_Abacus(hcat['N'], hcat[f'x_{Lsuff}com'], hcat[f'v_{Lsuff}com'], 
                                np.float32(header["ParticleMassHMsun"]),
                                dic['x'], dic['y'], dic['z'],
                                dic['vx'], dic['vy'], dic['vz'], 
                                dic['Mvir'], dic['log10_Mh'], Nthread)
    else :
        dic = dict((col, np.empty(hcat['N'].size, dtype='float32')) 
                   for col in ['x', 'y','z','vx','vy','vz', 'Rs','Rvir', 'c', 'Mvir', 'log10_Mh'])
        dic['Vrms'] = np.array(hcat[f'sigmav3d_{Lsuff}com'])
        if args['c_proxy'] == 0:
            mean_density = np.float32(
                        header["NP"] * header["ParticleMassHMsun"] / (header["BoxSizeHMpc"]**3))
            compute_col_from_Abacus(hcat['N'], hcat[f'x_{Lsuff}com'], hcat[f'v_{Lsuff}com'], 
                                    np.float32(header["ParticleMassHMsun"]),
                                    dic['x'], dic['y'], dic['z'],
                                    dic['vx'], dic['vy'], dic['vz'], 
                                    dic['Mvir'], dic['log10_Mh'], Nthread,
                                    dic['Rs'], dic['Rvir'], dic['c'],
                                    rvcirc_max=hcat[f'rvcirc_max_{Lsuff}com'], mean_density=mean_density, 
                                    SODensityL1=np.float32(header["SODensityL1"]))
        if args['c_proxy'] == 1:
            compute_col_from_Abacus(hcat['N'], hcat[f'x_{Lsuff}com'], hcat[f'v_{Lsuff}com'], 
                                    np.float32(header["ParticleMassHMsun"]),
                                    dic['x'], dic['y'], dic['z'],
                                    dic['vx'], dic['vy'], dic['vz'],
                                    dic['Mvir'], dic['log10_Mh'], Nthread,
                                    dic['Rs'], dic['Rvir'], dic['c'],
                                    r50=hcat[f'r50_{Lsuff}com'], r98=hcat[f'r98_{Lsuff}com'])
        
    
    dic['row_id'] = np.array(hcat['id'])
    
    if verbose : 
        print(f"Done took ", time.strftime("%H:%M:%S",time.gmtime(time.time() - start)))
    return dic, header["Omega_DE"], header["Omega_M"], header["H0"]/100, header["BoxSizeHMpc"]


def metropolis(nPoints, dir):
    """
    --- Function to generate random points in a NFW profile using Metropolis-Hastings algorithm 
    """
    
    if nPoints <= 100000:
        raise ValueError ('Error : NPoints must be above 10000')
    epsilon = 0.3
    previousX = 0.3

    def NFWprofile(x):
        # multiply by x2 to get P(r) and not rho(r)
        return 1./(x*(1+x)**2)*x**2
    previousP = NFWprofile(previousX)
    data = np.zeros(nPoints)
    i = 0
    for step in np.arange(nPoints)+1:
        evalX = previousX+2.*(np.random.uniform()-0.5)*epsilon
        evalP = NFWprofile(evalX)
        if evalX < 0.:
            evalP = 0.
        elif evalX < 0.01:
            evalP = NFWprofile(0.01)
        else:
            pass
        R = evalP/previousP
        if R >= 1:
            previousX = evalX*1.
            previousP = evalP*1.
        else:
            if np.random.uniform() < R:
                previousX = evalX*1.
                previousP = evalP*1.
        data[i] = previousX*1.
        i += 1
        # if step%100000==0 and step>1000 :
        #     print (step)
        #     print (np.max(data))
        #     x=np.linspace(0.,40.,1000)
        #     h,e=np.histogram(data[data>0],bins=1000,range=(0.,40.))
        #     plt.plot(x,h)
        #     x=np.linspace(0.,40.,1000)
        #     plt.plot(x,21000.*4.*NFWprofile(x))       #Normalisation a revoir mais profile OK
        #     plt.xscale('log')
        #     plt.yscale('log')
        #     plt.show()
    dataPruned = data[10000:]
    np.random.shuffle(dataPruned)
    np.save(dir+'nfw.npy', dataPruned)
    return dataPruned


def read_hcat(hcat_path, usecols=['row_id', "Mvir", "Rvir", "Rs", "x", "y",
                                  "z", "vx", "vy", "vz", "Vrms"]):
    """
    --- Function to load halo catalog from csv file
    """
    try:
        print('#Reading halo catalog...', flush=True)
        time1 = time.perf_counter()
        h_cat1 = dd.read_csv(hcat_path, usecols=usecols)
        h_cat = h_cat1.compute()
        h_cat = h_cat.set_index('row_id')
        print("#Halo catalog read ! time =",
              time.perf_counter()-time1, flush=True)
    except pd.errors.ParserError:
        print('Error : Wrong halo catalog file.', flush=True)
        sys.exit()
    return h_cat


def apply_rsd(cat, args, los='z', cosmo=None):
    if cosmo is None :
        from cosmoprimo.fiducial import DESI
        cosmo = DESI(engine='class')
    rsd_factor = 1 / (1 / (1 + args['z_simu']) * args['H_0'] * cosmo.efunc(args['z_simu']))
    pos_rsd = [cat[p] % args['boxsize'] if p !=los else (cat[p] + cat['v'+p]*rsd_factor) % args['boxsize'] for p in 'xyz']
    return pos_rsd
    
    
def compute_2PCF(cat, args, ells=(0, 2), edges=None, los='z', cosmo=None, R1R2=None):
    """
    --- Compute 2D correlation function and return multipoles for a galaxy/halo catalog in a cubic box.
    """
    if args['rsd']:
        X, Y, Z = apply_rsd (cat, args, los, cosmo)       
    else:
        X, Y, Z = cat['x']%args['boxsize'], cat['y']%args['boxsize'], cat['z']%args['boxsize']
        
    if edges is None:
        if args['bin_logscale']:
            r_bins = np.geomspace(args['rmin'], args['rmax'], args['n_r_bins']+1)
        else:
            r_bins = np.linspace(args['rmin'], args['rmax'], args['n_r_bins']+1)
        edges = (r_bins, np.linspace(-args['mu_max'], args['mu_max'], args['n_mu_bins']))
        
    result = TwoPointCorrelationFunction('smu', edges, 
                                         data_positions1=[X, Y, Z], engine='corrfunc', 
                                         boxsize=args['boxsize'], los=los, nthreads=args['nthreads'], R1R2=R1R2)
    
    return project_to_multipoles(result, ells=ells)

    
def compute_wp(cat, args, edges=None, los='z', cosmo=None, R1R2=None, pimax=40):
    """
    --- Compute projected correlation function for a galaxy/halo catalog in a cubic box.
    """
    if args['rsd']:
        X, Y, Z = apply_rsd (cat, args, los, cosmo)       
    else:
        X, Y, Z = cat['x']%args['boxsize'], cat['y']%args['boxsize'], cat['z']%args['boxsize']
    
    if edges is None:
        if args['bin_logscale']:
            r_bins = np.geomspace(args['rp_min'], args['rp_max'], args['n_rp_bins']+1, endpoint=(True))
        else:
            r_bins = np.linspace(args['rp_min'], args['rp_max'], args['n_rp_bins']+1)
        edges = (r_bins, np.linspace(-pimax, pimax, 2*pimax+1))
    
    result = TwoPointCorrelationFunction('rppi', edges, 
                                         data_positions1=[X, Y, Z], engine='corrfunc', 
                                         boxsize=args['boxsize'], los=los, nthreads=args['nthreads'], R1R2=R1R2)
    
    return project_to_wp(result)


def compute_chi2(model, data, inv_Cov2):
    """
    --- Compute chi2
    """
    arr_diff = data - model
    chi2 = np.matmul(arr_diff, np.matmul(inv_Cov2, arr_diff.T))
    return chi2


def subsample_halos(logM):
    downfactors = np.zeros(len(logM))
    mask = logM < 12.07
    downfactors[mask] = 0.2/(1.0 + 10*np.exp(-(logM[mask] - 11.2)*25))
    downfactors[~mask] = 1.0/(1.0 + 0.1*np.exp(-(logM[~mask] - 12.6)*7))
    return downfactors


class HOD:
    """
    --- HOD code 
    """

    def __init__(self, args, hcat=None, usecols=None, read_Abacus=False, use_L2=True, mass_cut=None):
        """
        ---
        """
        self.args = args.copy()
        if hcat is None:
            if read_Abacus:
                if use_L2:
                    Lsuff = 'L2'
                else : 
                    Lsuff = ''
                usecols=['id', f'x_{Lsuff}com', f'v_{Lsuff}com', 'N']
                str_z = format(self.args["z_simu"], ".3f")
                if 'small' in self.args['sim_name']:
                    path_to_sim = os.path.join("/feynman/scratch/dphp/ar264273/small", 
                                               f"{self.args['sim_name']}", "halos", f"z{str_z}")
                else:
                    path_to_sim = os.path.join("/feynman/scratch/dphp/ar264273/Abacus", 
                                               f"{self.args['sim_name']}", "halos", f"z{str_z}")
                tt = time.time()
                if self.args['use_particles']:
                    print('read particles')
                    self.args['particle_filename'] = f"/feynman/scratch/dphp/ar264273/Abacus/preprocess_particles/part_{self.args['sim_name']}z{self.args['z_simu']}.fits"
                    try : 
                        usecols += ['npstartA', 'npoutA']      
                        self.part_fits = fitsio.FITS(self.args['particle_filename'])
                    except OSError: 
                        print(f"Particles are not available for {self.args['sim_name']} at z={str_z}")
                        sys.exit()
                else :  
                    if self.args['c_proxy'] == 0:
                        usecols += [f'rvcirc_max_{Lsuff}com', f'sigmav3d_{Lsuff}com']
                    if self.args['c_proxy'] == 1:
                        usecols += [f'r50_{Lsuff}com', f'r98_{Lsuff}com', f'sigmav3d_{Lsuff}com']
                        
                self.hcat, self.args["Om_L"], self.args["Om_M"], self.args["h0"], self.args['boxsize'] = load_hcat_from_Abacus(path_to_sim, 
                                                                                                                               usecols, 
                                                                                                                               self.args,
                                                                                                                               Lsuff,
                                                                                                            Nthread=self.args['nthreads'], 
                                                                                                                        mass_cut=mass_cut)
                self.hcat['weight'] = np.ones_like(self.hcat['x'])
                print(f'Abacus Halo catalog loaded, took {time.strftime("%H:%M:%S",time.gmtime(time.time() - tt))}', flush=True)
                self.cosmo = cosmoprimo.fiducial.AbacusSummit(self.args['sim_name'].split('_c')[-1][:3]).get_background(engine='class')
                
            else:
                self.hcat = {}
                halo_cat = read_hcat(os.path.join(self.args['hcat_dir'], self.args['sim_name']))
                for col in halo_cat.columns:
                    self.hcat[col] = halo_cat[col].values
                self.hcat['log10_Mh'] = np.log10(self.hcat['Mvir'])
                self.hcat['row_id'] = halo_cat.index.values
                if 'c' not in halo_cat.columns:
                    self.hcat['c'] = self.hcat['Rvir']/self.hcat['Rs'] 
                
                print(f'WARNING COSMOLOGY SET WITH Om={self.args["Om_M"]} and sigma8 ={self.args["sigma_8"]} FOR RESHIFT DISTANCE CONVERSION')
                cosmo_custom = cosmoprimo.Cosmology(omega_m=self.args["Om_M"], sigma8=self.args["sigma_8"])
                self.cosmo = cosmo_custom.get_background(engine='class')
                #print('One can provide the following parameters (including conflicts):', cosmo.get_default_parameters(include_conflicts=True))
        else :
            if type(hcat) is not dict:
                raise ValueError ('halo catalog is not a dictionary')
            self.hcat = hcat
        
            if self.args['use_particles']:
                if ('npstartA'  not in hcat.keys()) & ('npoutA' in hcat.keys()):
                    raise ColumnError ('no column npstartA and npoutA in the halo catalog')
                print('read particles')
                self.args['particle_filename'] = f"/feynman/scratch/dphp/ar264273/Abacus/preprocess_particles/part_{self.args['sim_name']}z{self.args['z_simu']}.fits"
                try : 
                    self.part_fits = fitsio.FITS(self.args['particle_filename'])
                except OSError: 
                    print(f"Particles are not available for {self.args['sim_name']} at z={str_z}")
            
            print(f'WARNING COSMOLOGY SET WITH Om={self.args["Om_M"]} and sigma8 ={self.args["sigma_8"]} FOR RESHIFT DISTANCE CONVERSION')
            cosmo_custom = cosmoprimo.Cosmology(omega_m=self.args["Om_M"], sigma8=self.args["sigma_8"])
            self.cosmo = cosmo_custom.get_background(engine='class')
            
        if self.args['subsample_halos']: 
            tet = subsample_halos(self.hcat['log10_Mh'])
            submask = np.array(np.random.binomial(n = 1, p = tet), dtype=bool)
            for var in self.hcat.keys():
                self.hcat[var] = self.hcat[var][submask]

        try:
            file = open(self.args['path_to_NFW_draw'], 'rb')
        except FileNotFoundError:
            print("Generate random point in a NFW for satellite drawing")
            metropolis(15000000, self.args['path_to_NFW_draw'])
        self.NFW_draw = np.load(self.args['path_to_NFW_draw'])

        if ('c' not in self.hcat.keys()) & (not self.args['use_particles']):
            self.hcat['c'] = self.hcat['Rvir']/self.hcat['Rs']
        try :
            self.fun_HOD = globals()[self.args['HOD_model']]
        except :
            raise ValueError(
                'Wrong : HOD_model, only LNHOD, GHOD, SHOD, SFHOD, HMQ, HMQ_Sandy, mHMQ are allowed')
        self.rng = np.random.RandomState(seed=self.args['seed'])

    @property
    def ngal(self, verbose=True):
        '''
        --- Return the number of galaxy and the satelitte fraction form HOD parameters
        '''
        try :
            self.fun_HOD = globals()[self.args['HOD_model']]
        except :
            raise ValueError(
                'Wrong : HOD_model, only LNHOD, GHOD, SHOD, SFHOD, HMQ, HMQ_Sandy, mHMQ are allowed')
        start = time.time()
        ngal, fsat = self._compute_ngal(self.hcat['log10_Mh'], self.args['Ac'],  
                                       self.args['log_Mcent'], self.args['sigma_M'],
                                       self.args['gamma'], self.args['Q'], self.args['pmax'], 
                                       self.args['As'], 
                                       self.args['M_0'],
                                       self.args['M_1'], self.args['alpha'], 
                                       self.fun_HOD, self.args['nthreads'],self.hcat['weight'], self.args['satellites'])
        if verbose:
            print(time.time()-start)
        return ngal, fsat

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _compute_ngal(log10_Mh, Ac, Mc, sig_M, gamma, Q, pmax, As, M_0, M_1, alpha, fun_HOD, Nthread, weight, satellites):
        """
        --- Compute the number of galaxy and the satelitte fraction form HOD parameters 
        """
        nbinsM, logbinM = np.histogram(log10_Mh, bins=100)[:2]
        LogM = np.zeros(len(logbinM)-1)
        dM = np.diff(logbinM)[0]
        ngal_c = 0
        ngal_sat = 0

        # starting index of each thread
        hstart = np.rint(np.linspace(0, len(logbinM)-1, Nthread + 1))
        
        if satellites :
            for tid in numba.prange(Nthread):
                for i in range(int(hstart[tid]), int(hstart[tid + 1])):
                    LogM = ((logbinM[i]+logbinM[i+1])*0.5)
                    Ncent = fun_HOD(LogM, Ac, Mc, sig_M, gamma, Q, pmax)
                    if Ncent < 0:
                        Ncent = 0
                    if LogM < M_0:
                        N_sat = 0
                    else:
                        N_sat = compute_Nsat(LogM, As, M_0, M_1, alpha)
                    ngal_c += (nbinsM[i]/dM * Ncent*dM)
                    ngal_sat += (nbinsM[i]/dM * N_sat*dM)
            ngal_tot = ngal_c+ngal_sat
            return ngal_tot, ngal_sat/ngal_tot
        else : 
            for tid in numba.prange(Nthread):
                for i in range(int(hstart[tid]), int(hstart[tid + 1])):
                    LogM = ((logbinM[i]+logbinM[i+1])*0.5)
                    Ncent = fun_HOD(LogM, Ac, Mc, sig_M, gamma, Q, pmax)
                    if Ncent < 0:
                        Ncent = 0
                    ngal_c += (nbinsM[i]/dM * Ncent*dM)
           
            return ngal_c, 0
        
    @staticmethod
    def _compute_ngal2(self):
        """
        --- Compute the number of galaxy and the satelitte fraction form HOD parameters 
        MArche PAS
        """
        nbinsM, logbinM = np.histogram(self.hcat['log10_Mh'], bins=100)[:2]
        # starting index of each thread
        LogM = (logbinM[1:]+logbinM[:-1])*0.5
        Ncent, N_sat = self._compute_N(LogM,
                                       self.args['Ac'],
                                       self.args['log_Mcent'],
                                       self.args['sigma_M'],
                                       self.args['gamma'],
                                       self.args['Q'],
                                       self.args['pmax'],
                                       self.args['As'],
                                       self.args['M_0'],
                                       self.args['M_1'],
                                       self.args['alpha'],
                                       self.fun_HOD,
                                       self.args['nthreads'],
                                       self.hcat['weight'],
                                       self.args['satellites'],
                                       self.args['conformity_bias'],seed=None)[:2]
        ngal_c = (nbinsM * Ncent).sum()
        ngal_sat = (nbinsM * N_sat).sum()
        ngal_tot = ngal_c+ngal_sat
        return ngal_tot, ngal_sat/ngal_tot
     

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _compute_N(log10_Mh, Ac, Mc, sig_M, gamma, Q, pmax, As, M_0, M_1, 
                   alpha, fun_HOD, Nthread, weight, satellites, conformity, seed=None):
        """
        --- Compute the probability N for central galaxies given a HOD model
        """
        M_01 = 0.
        numba.set_num_threads(Nthread)
        # starting index of each thread
        hstart = np.rint(np.linspace(0, len(log10_Mh), Nthread + 1))
        Ncent = np.empty_like(log10_Mh)
        cond_cent = np.empty_like(log10_Mh)
        N_sat = np.empty_like(log10_Mh)
        proba_sat = np.empty_like(log10_Mh, dtype=np.int64)
        
        # figuring out the number of halos kept for each thread
        for tid in numba.prange(Nthread):
            if seed is not None:
                np.random.seed(seed[tid])
            for i in range(int(hstart[tid]), int(hstart[tid + 1])):
                Ncent[i] = fun_HOD(log10_Mh[i], Ac, Mc, sig_M, gamma, Q, pmax) * weight[i]
                cond_cent[i] = Ncent[i] - np.random.uniform(0, 1) > 0
                
                if satellites :
                    if (log10_Mh[i] - M_0) < 0.001:
                        M_01 = M_0 + 0.001
                    if log10_Mh[i] <= np.maximum(M_0,M_01):
                        N_sat[i] = 0
                    else:
                        N_sat[i] = compute_Nsat(log10_Mh[i], As, np.maximum(M_0,M_01), M_1, alpha)*weight[i]
                #if Ncent[i] == 0 : 
                #    proba_sat[i] = np.random.poisson(N_sat[i], size=np.int(np.round(weight[i]))).max()
                #else:
                    if conformity:
                        proba_sat[i] = np.random.poisson(N_sat[i]*cond_cent[i])
                    else :
                        proba_sat[i] = np.random.poisson(N_sat[i])
                    
        return Ncent, N_sat, cond_cent, proba_sat

    
    def _NFW(self, sat_cat, Nb_sat, seed):
        """
        --- Compute NFW postion and velocity shifts for satelittes galaxies non multithread method (used for fitting)
        """
        np.random.seed(seed)
        c = sat_cat['c']
        M = sat_cat["Mvir"]
        Rvir = sat_cat["Rvir"]

        NFW = self.NFW_draw[self.NFW_draw < sat_cat['c'].max()]
        if len(NFW) > Nb_sat:
            np.random.shuffle(NFW)
            eta = NFW[:Nb_sat]
        else:
            eta = NFW[np.random.randint(0, len(NFW), Nb_sat)]

        a = 0
        while len(eta[eta > c]) > 1:
            temp = len(eta[eta > c])
            if a == temp:
                break
            a = temp
            eta[eta > c] = NFW[np.random.randint(0, high=len(NFW), size=a)]
        tet = np.zeros(len(eta[eta > c]))
        for i in range(len(eta[eta > c])):
            a = eta[c[eta > c][i] > eta]
            tet[i] = a[np.random.randint(len(a))]
        eta[eta > c] = tet
        del tet

        etaVir = eta/c  # =r/rvir

        def f(x):
            return np.log(1.+x)-x/(1.+x)
        G = 4.302e-6  # in kpc/Msol (km.s)^2
        vVir = np.sqrt(G*M/Rvir)
        v = vVir * np.sqrt(f(c * etaVir) / (etaVir * f(c)))

        def getPointsOnSphere(nPoints, rng):
            u1, u2 = rng.uniform(size=(2, nPoints))
            cmin = -1
            cmax = +1
            ra = 0 + u1*(2*np.pi-0)
            dec = np.pi - (np.arccos(cmin+u2*(cmax-cmin)))
            ur = np.zeros((nPoints, 3))
            ur[:, 0] = np.sin(dec) * np.cos(ra)
            ur[:, 1] = np.sin(dec) * np.sin(ra)
            ur[:, 2] = np.cos(dec)
            return ur

        def reScale(a, b):
            return np.transpose(np.multiply(np.transpose(a), np.transpose(b)))
        ur = getPointsOnSphere(len(M), self.rng)
        uv = getPointsOnSphere(len(M), self.rng)
        del M, c

        return reScale(ur, (etaVir*Rvir)), reScale(uv, v)
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _compute_fast_NFW(NFW_draw, h_id, x_h, y_h, z_h, vx_h, vy_h, vz_h, vrms_h, c, M, Rvir, rd_pos,
                          rd_vel, num_sat, f_sigv, vel_sat, Nthread, seed=None):
        """
        --- Compute NFW positions and velocities for satelitte galaxies
        """
        numba.set_num_threads(Nthread)
        G = 4.302e-6  # in kpc/Msol (km.s)^2
        # figuring out the number of halos kept for each thread
        h_id = np.repeat(h_id, num_sat)
        M = np.repeat(M, num_sat)
        c = np.repeat(c, num_sat)
        Rvir = np.repeat(Rvir, num_sat)
        x_h = np.repeat(x_h, num_sat)
        y_h = np.repeat(y_h, num_sat)
        z_h = np.repeat(z_h, num_sat)
        vx_h = np.repeat(vx_h, num_sat)
        vy_h = np.repeat(vy_h, num_sat)
        vz_h = np.repeat(vz_h, num_sat)
        vrms_h = np.repeat(vrms_h, num_sat)
        x_sat = np.empty_like(x_h)
        y_sat = np.empty_like(y_h)
        z_sat = np.empty_like(z_h)
        vx_sat = np.empty_like(vx_h)
        vy_sat = np.empty_like(vy_h)
        vz_sat = np.empty_like(vz_h)

        # starting index of each thread
        hstart = np.rint(np.linspace(0, num_sat.sum(), Nthread + 1))
        for tid in numba.prange(Nthread):
            if seed is not None:
                np.random.seed(seed[tid])
            for i in range(int(hstart[tid]), int(hstart[tid + 1])):
                ind = i
                while (NFW_draw[ind] > c[i]):
                    ind = np.random.randint(0, len(NFW_draw))

                etaVir = NFW_draw[ind]/c[i]  # =r/rvir
                p = etaVir * Rvir[i] / 1000
                x_sat[i] = x_h[i] + rd_pos[i, 0] * p
                y_sat[i] = y_h[i] + rd_pos[i, 1] * p
                z_sat[i] = z_h[i] + rd_pos[i, 2] * p
                if vel_sat == 'NFW':
                    v = np.sqrt(G*M[i]/Rvir[i]) * \
                                np.sqrt(f(c[i] * etaVir) / (etaVir * f(c[i])))
                    vx_sat[i] = vx_h[i] + rd_vel[i, 0] * v
                    vy_sat[i] = vy_h[i] + rd_vel[i, 1] * v
                    vz_sat[i] = vz_h[i] + rd_vel[i, 2] * v
                elif vel_sat == 'rd_normal':
                    sig = vrms_h[i]*0.577*f_sigv
                    vx_sat[i] = np.random.normal(loc=vx_h[i], scale=sig)
                    vy_sat[i] = np.random.normal(loc=vy_h[i], scale=sig)
                    vz_sat[i] = np.random.normal(loc=vz_h[i], scale=sig)
                else:
                    raise ValueError(
                        'Wrong vel_sat argument only "rd_normal" or "NFW"')
        return h_id, x_sat, y_sat, z_sat, vx_sat, vy_sat, vz_sat, M
    
    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _find_part_index(Nsat, startA, outA, Nthread=64, seed=None):
        '''
        Draw the particle indexes that will be used for the satellite catalog
        '''    
        ind_part = np.zeros(Nsat.sum())
        end  = startA + outA
        hstart = np.rint(np.linspace(0, len(Nsat), Nthread + 1))
        for tid in numba.prange(Nthread):
            u = np.sum(Nsat[:int(hstart[tid])])
            if seed is not None:
                    np.random.seed(seed[tid])
            for i in range(int(hstart[tid]), int(hstart[tid + 1])):
                if outA[i] == 1:
                    ind_part[u] = startA[i]
                else:
                    liste = np.rint(np.linspace(startA[i], end[i]-1, outA[i]))
                    rr = np.random.choice(liste, size=Nsat[i], replace=False)
                    for k in range(len(rr)):
                        ind_part[u+k] = rr[k]
                u += Nsat[i]
        return ind_part 
    
    def _add_fsat(self, cen_cat, seed=None, verbose=True):
        
        """Method to add satellites to the central distribution 
        """   
        if seed is not None:
            np.random.seed(seed[0])
        ngal = len(cen_cat['x'])
        cen_cat = self.downsample_mock_cat(cen_cat, mask=np.random.uniform(size=len(cen_cat['x']))>self.args['fsat'])
        try : 
            indx=np.random.choice(cen_cat['row_id'], size=ngal-len(cen_cat['x']), replace=False)   
        except : 
            return 0
        mask_sat = np.in1d(self.hcat['row_id'], indx)

        sat_cat={}
        for var in self.hcat.keys():
            sat_cat[var] = self.hcat[var][mask_sat]
        Nb_sat = mask_sat.sum()
        
        if self.args['use_particles']:
            # Doesn't work I don't know why pb with Nsat 
            Nsat = np.ones_like(sat_cat['x'], dtype=int)
            mask = Nsat > sat_cat['npoutA']

            Nsat[mask] = sat_cat['npoutA'][mask]
            print(sat_cat['npoutA'][mask], Nsat[mask])
            print((Nsat > sat_cat['npoutA']).sum())
            print('nb sat', Nb_sat)
            print('nb part sat', Nsat.sum())
            print('nb halo for sat', len(sat_cat['x'])) 
            sat_cat = self._compute_satellites_with_particles(Nsat, sat_cat, seed, verbose=verbose)
        
        else : 
            nfw = self._NFW(sat_cat, Nb_sat, seed=None)

            for ll,col in enumerate(['x','y','z']):
                sat_cat[col] += (nfw[0][:,ll]/1000)

            if self.args['vel_sat'] == 'rd_normal':
                sig = sat_cat["Vrms"]*0.577*self.args['f_sigv']
                sat_cat["vx"] = self.rng.normal(
                    loc=sat_cat["vx"], scale=sig, size=Nb_sat).astype(sat_cat["vx"].dtype)
                sat_cat["vy"] = self.rng.normal(
                    loc=sat_cat["vy"], scale=sig, size=Nb_sat).astype(sat_cat["vx"].dtype)
                sat_cat["vz"] = self.rng.normal(
                    loc=sat_cat["vz"], scale=sig, size=Nb_sat).astype(sat_cat["x"].dtype)
            else:
                for ll,col in enumerate(['vx','vy','vz']):
                    sat_cat[col] += (nfw[1][:,ll]/1000)
            sat_cat['Central'] = np.zeros_like(sat_cat['x'])
        final_cat={}
        for col in cen_cat.keys():
            final_cat[col] = np.concatenate((cen_cat[col],sat_cat[col]))
        return final_cat
    
    

    def _compute_satellites_with_particles(self, Nsat, sat_halos, seed=None, verbose=True):
        '''
        Create satellite catalog from particle positions and velocities.  
        '''
        st = time.time()
        start = time.time()
        ind_part = self._find_part_index(Nsat, np.array(sat_halos['npstartA'], dtype='int64'), 
                                   np.array(sat_halos['npoutA'], dtype='int64'), 
                                   Nthread=self.args['nthreads'], seed=seed)
        print('time to get index', time.time()-start)
        print(ind_part.min(), ind_part.max())
        print(len(ind_part))

        start = time.time()
        print('read fits')
        part_cat = self.part_fits[1]['pos', 'vel'][ind_part]
        print('time to read part', time.time() - start)
        print('part cat',len(part_cat))
        start = time.time()
        sat_cat={}
        for i,col in enumerate(['x','y','z']):
            sat_cat[col] = part_cat['pos'].T[i]
            sat_cat[f'v{col}'] = part_cat['vel'].T[i]
        sat_cat['Mvir'] = np.repeat(sat_halos['Mvir'], Nsat)
        sat_cat['row_id'] = np.repeat(sat_halos['row_id'], Nsat)
        sat_cat['Central'] = np.zeros_like(sat_cat['x'])
        print('time create sat cat', time.time() - start)
        print('all', time.time()-st)
        return sat_cat

    def make_mock_cat(self, fix_seed=None, verbose=True, HOD_param=None):
        """
        Generate mock catalogs from HOD model.

        Parameters
        ----------
        self

        fix_seed : Fix the seed for reproductibility. Caveat : Only works for a same number of threads in args['nthreads']

        Output
        ------
        mock_cat : Pandas DataFrame
            DF of mock galaxies properties
        """
        
        if type(HOD_param) is dict: 
            self.args.update(HOD_param)
        
        try :
            self.fun_HOD = globals()[self.args['HOD_model']]
        except :
            raise ValueError(
                'Wrong : HOD_model, only LNHOD, GHOD, SHOD, SFHOD, HMQ, HMQ_Sandy, mHMQ are allowed')

        rng = np.random.RandomState(seed=fix_seed)
        timeall = time.time()

        start = time.time()
        Ac, Mc, sig_M, gamma, Q, pmax = self.args['Ac'], self.args[
            'log_Mcent'], self.args['sigma_M'], self.args['gamma'], self.args['Q'], self.args['pmax']

        if self.args['use_shift'] == 'Y':
            self.args['M_0'] = self.args['log_Mcent'] + self.args['shift_M_0']
            self.args['M_1'] = self.args['log_Mcent'] + self.args['shift_M_1']
        try: 
            if self.args['M0=Mc'] == 'Y':
                self.args['M_0'] = self.args['log_Mcent']
        except:
            pass

        As, M_0, M_1, alpha = self.args['As'], self.args['M_0'], self.args['M_1'], self.args['alpha']
        
        if self.args['density']:  # Set fixed density if asked
            if self.args['add_fsat'] | self.args['conformity_bias']:
                ds = self.args['density']*self.args['boxsize']**3 / self._compute_ngal(self.hcat['log10_Mh'], Ac, Mc, sig_M,
                                                                                       gamma, Q, pmax, As, M_0, M_1, 
                                                                                       alpha, self.fun_HOD, 
                                                                                       self.args['nthreads'],
                                                                                       self.hcat['weight'],
                                                                                       False)[0]
            else: 
                ds = self.args['density']*self.args['boxsize']**3 / self._compute_ngal(self.hcat['log10_Mh'], Ac, Mc, sig_M,
                                                                                   gamma, Q, pmax, As, M_0, M_1, 
                                                                                   alpha, self.fun_HOD, 
                                                                                   self.args['nthreads'],
                                                                                   self.hcat['weight'],
                                                                                   self.args['satellites'])[0]
            if (Ac > 1) or (As > 1):
                print(f'WARNING Ac={Ac}, As={As}, density doest fix at {self.args["density"]}')
            else : 
                Ac *= ds
                As *= ds
                
        if fix_seed is not None:
            seed = rng.randint(0, 4294967295, self.args['nthreads'])
        else:
            seed = None
        
        if self.args['add_fsat'] : 
            if not self.args['density']:
                raise('Density has to be set to use add_fsat method')
            
            if verbose:
                print('Use add_fsat method to assing satellites')
            cent, sat, cond_cent, proba_sat = self._compute_N(self.hcat['log10_Mh'], Ac, Mc, 
                                                          sig_M, gamma, Q, pmax,
                                                          As, M_0, M_1, alpha, self.fun_HOD, 
                                                          self.args['nthreads'], self.hcat['weight'],
                                                          False, self.args['conformity_bias'], seed)
        else : 
            cent, sat, cond_cent, proba_sat = self._compute_N(self.hcat['log10_Mh'], Ac, Mc, 
                                                          sig_M, gamma, Q, pmax,
                                                          As, M_0, M_1, alpha, self.fun_HOD, 
                                                          self.args['nthreads'], self.hcat['weight'],
                                                          self.args['satellites'], self.args['conformity_bias'], seed)
        
        mask_cent = cond_cent == 1

        cent_cat={}
        keys=  ["row_id", "x", "y", "z", "vx", "vy", "vz", "Mvir"]
        for col in keys:
            cent_cat[col] = self.hcat[col][mask_cent]
        cent_cat['Central'] = np.ones(cent_cat['x'].size,dtype='int')
        if (cent > 1).any():
            print(f'WARNING proba cent={cent.max()} >1')
        if verbose:
            print("gen proba", time.time() - start, flush=True)         

        if not self.args['satellites'] :
            return cent_cat

        if fix_seed is not None:
            seed = rng.randint(0, 4294967295, self.args['nthreads'])
        else:
            seed = None
        
        if self.args['add_fsat']:
            return self._add_fsat(cent_cat, seed=seed, verbose=verbose)
        
        start = time.time()
        Nb_sat = proba_sat.sum()
        if Nb_sat == 0: 
            return cent_cat
        mask_sat = proba_sat > 0
        sat_cat = {}
        for col in self.hcat.keys():
                sat_cat[col] = self.hcat[col][mask_sat]

        if self.args['use_particles']:
            mask = proba_sat[mask_sat] > sat_cat['npoutA']
            Nsat = proba_sat[mask_sat]
            Nsat[mask] = sat_cat['npoutA'][mask]
            print(sat_cat['npoutA'][mask], Nsat[mask])
            print((proba_sat[mask_sat] > sat_cat['npoutA']).sum())
            print('nb sat', Nb_sat)
            print('nb part sat', Nsat.sum())
            print('nb halo for sat', len(sat_cat['x']))
            sat = self._compute_satellites_with_particles(Nsat, sat_cat, seed, verbose=verbose)

        else:
            NFW = self.NFW_draw[self.NFW_draw < sat_cat['c'].max()]
            if len(NFW) > Nb_sat:
                rng.shuffle(NFW)
            else:
                NFW = NFW[rng.randint(0, len(NFW), Nb_sat)]

            if fix_seed is not None:
                seed1 = rng.randint(0, 4294967295, self.args['nthreads'])
            else:
                seed1 = None
            rd_pos = getPointsOnSphere(Nb_sat, np.minimum(Nb_sat, self.args['nthreads']), seed1)
            if self.args['vel_sat'] == 'NFW':
                seed2 = rng.randint(0, 4294967295, self.args['nthreads'])
                rd_vel = getPointsOnSphere(Nb_sat, np.minimum(Nb_sat, self.args['nthreads']), seed2)
            else:
                rd_vel = np.ones_like(rd_pos)
            res = self._compute_fast_NFW(NFW, sat_cat['row_id'], sat_cat['x'], 
                                         sat_cat['y'], sat_cat['z'], sat_cat['vx'],
                                         sat_cat['vy'], sat_cat['vz'], sat_cat['Vrms'], 
                                         sat_cat['c'], sat_cat['Mvir'], sat_cat['Rvir'], 
                                         rd_pos, rd_vel, proba_sat[mask_sat],
                                         self.args['f_sigv'], self.args['vel_sat'], 
                                         self.args['nthreads'], seed)
            keys= ["row_id", "x", "y", "z", "vx", "vy", "vz", "Mvir"]
            sat = dict(zip(keys, res))
            sat['Central'] = np.zeros(sat['x'].size, dtype='int')

        if verbose:
            print("gen satellites ", time.time() - start, flush=True)
            start = time.time()

        final_cat={}
        for col in cent_cat.keys():
            final_cat[col] = np.concatenate((cent_cat[col],sat[col]))
        if verbose:
            print("overall time ", time.time() - timeall, flush=True)
            print("number of central galaxies ", mask_cent.sum(), flush=True)
            print("number of satellite galaxies ", Nb_sat, flush=True)
            print("satellite fraction ", Nb_sat/final_cat['x'].size, flush=True)
        return final_cat
                                         
    def get_2PCF(self, mock_cat, verbose=True):
        """
        --- Return the 2PCF for a given mock catalog in a cubic box
        """
        if verbose:
            print('#Computing 2PCF...', flush=True)
            time1 = time.time()
        calc = compute_2PCF(mock_cat, self.args, self.args['multipole_index'], self.args['edges_smu'], self.args['los'], self.cosmo)
        if verbose:
            print('#2PCF computed !time =', time.time()-time1, flush=True)
        return calc

    def get_wp(self, mock_cat, pimax=40, verbose=True):
        """
        --- Return wp (projected correlation function) for a given mock catalog in a cubic box
        """
        if verbose:
            print('#Computing wp...', flush=True)
            time1 = time.time()
        rp, wp = compute_wp(mock_cat, self.args, self.args['edges_rppi'], self.args['los'], self.cosmo, pimax=pimax)
        if verbose:
            print('#wp computed !time =', time.time()-time1, flush=True)
        return rp, wp
    
    def dict_to_array(self, cat):
        """
        --- Convert dictionary to array 
        """
        dtype = [(v2,cat[v2].dtype) for v2 in cat.keys()]
        dtype[0] = ('row_id', 'int64')
        cat_arr = np.zeros(len(cat['x']), dtype)
        for v2 in cat.keys():
            cat_arr[v2] = cat[v2].copy()
        return cat_arr
    
    def save_cat(self, fn, cat, **kwargs):
        """
        --- Save to mock catalog to fits file
        """
        if type(cat) is dict:
            cat = dict_to_array(cat)
        fitsio.write(fn, cat , clobber=True, header=self.args, **kwargs)
        



class HOD_parallel(HOD):
    """
    --- HOD Fitting code using iterative gaussian process procedure, taking into account the realization noise in HOD models by computing n realizations for each parameter point. 
    
    The fitting procedure is describe as follow :
    - Generate training sample using by first computing clustering measurement for point in the parameter space distributed in a Latin Hypercube Sample (LHS) 
    - Then use a Gaussian Process to map the likelihood distrubtion, compute the aquisition function which will give the next point  in the parameter space to be computed and added in the training sample
    """

    def __init__(self, args, hcat=None, usecols=None, read_Abacus=False, use_L2=True, mass_cut=None):
        """
        ---
        """
        super().__init__(args, hcat, usecols, read_Abacus=read_Abacus, mass_cut=mass_cut, use_L2=use_L2)

    def compute_parallel_catalogues(self, ncat=20, verbose = False):
        """
        --- Compute in parallel, n mock catalogs for a given HOD parameters
        """
        def processInput(i):
            """
            ---
            """
            
            self.rng.seed(i)
            mask_cent = pcent - self.rng.uniform(0, 1, len(pcent)) > 0
                            
            cent_cat={}
            keys=  ['row_id', "x", "y", "z", "vx", "vy", "vz", "Mvir"]
            for col in keys:
                cent_cat[col] = self.hcat[col][mask_cent]
            cent_cat['Central'] = np.ones(cent_cat['x'].size,dtype='int')
            if not self.args['satellites']:
                return cent_cat
            elif self.args['add_fsat']:
                    return self._add_fsat(cent_cat, seed=[i], verbose=verbose)
            else:
                if self.args['conformity_bias']:
                    poisson_sat = self.rng.poisson(psat*mask_cent)
                else:
                    poisson_sat = self.rng.poisson(psat)
                Nb_sat = poisson_sat.sum()
                if Nb_sat == 0: 
                    return cent_cat
                mask_sat = poisson_sat > 0
                sat_cat = {}
                for col in self.hcat.keys():
                    sat_cat[col] = np.repeat(self.hcat[col][mask_sat], poisson_sat[mask_sat])
                
                nfw = self._NFW(sat_cat, Nb_sat, i)
                for ll,col in enumerate(['x','y','z']):
                    sat_cat[col] += (nfw[0][:,ll]/1000)
                
                if self.args['vel_sat'] == 'rd_normal':
                    sig = sat_cat["Vrms"]*0.577*self.args['f_sigv']
                    sat_cat["vx"] = self.rng.normal(
                        loc=sat_cat["vx"], scale=sig, size=Nb_sat).astype(sat_cat["vx"].dtype)
                    sat_cat["vy"] = self.rng.normal(
                        loc=sat_cat["vy"], scale=sig, size=Nb_sat).astype(sat_cat["vx"].dtype)
                    sat_cat["vz"] = self.rng.normal(
                        loc=sat_cat["vz"], scale=sig, size=Nb_sat).astype(sat_cat["x"].dtype)
                else:
                    for ll,col in enumerate(['vx','vy','vz']):
                        sat_cat[col] += (nfw[1][:,ll]/1000)

                cent_cat['Central'] = np.ones(cent_cat['x'].size,dtype='int')  
                sat_cat['Central'] = np.zeros(sat_cat['x'].size, dtype='int')
                       
                final_cat={}
                for col in cent_cat.keys():
                    final_cat[col] = np.concatenate((cent_cat[col],sat_cat[col]))
                return final_cat

        inputs = self.rng.randint(0, 4294967295, size=ncat)
            
        Ac, Mc, sig_M, gamma, Q, pmax = self.args['Ac'], self.args[
            'log_Mcent'], self.args['sigma_M'], self.args['gamma'], self.args['Q'], self.args['pmax']
        
        if self.args['use_shift'] == 'Y':
            self.args['M_0'] = self.args['log_Mcent'] + self.args['shift_M_0']
            self.args['M_1'] = self.args['log_Mcent'] + self.args['shift_M_1']
            
        try: 
            if self.args['M0=Mc'] == 'Y':
                self.args['M_0'] = self.args['log_Mcent']
        except:
            pass
            
        As, M_0, M_1, alpha = self.args['As'], self.args['M_0'], self.args['M_1'], self.args['alpha']
        
        if self.args['density']:  # Set fixed density if asked
            if self.args['add_fsat'] | self.args['conformity_bias']:
                ds = self.args['density']*self.args['boxsize']**3 / self._compute_ngal(self.hcat['log10_Mh'], Ac, Mc, sig_M,
                                                                                       gamma, Q, pmax, As, M_0, M_1, 
                                                                                       alpha, self.fun_HOD, 
                                                                                       self.args['nthreads'],
                                                                                       self.hcat['weight'],
                                                                                       False)[0]
            else: 
                ds = self.args['density']*self.args['boxsize']**3 / self._compute_ngal(self.hcat['log10_Mh'], Ac, Mc, sig_M,
                                                                                   gamma, Q, pmax, As, M_0, M_1, 
                                                                                   alpha, self.fun_HOD, 
                                                                                   self.args['nthreads'],
                                                                                   self.hcat['weight'],
                                                                                   self.args['satellites'])[0]
            if (Ac > 1) or (As > 1):
                print(f'WARNING Ac={Ac}, As={As}, density doest fix at {self.args["density"]}')
            else : 
                Ac *= ds
                As *= ds
        if verbose:
            start = time.time()
            print(f"Compute {ncat} catalogs...", flush=True)
        pcent, psat = self._compute_N(self.hcat['log10_Mh'], Ac, Mc, sig_M, gamma, Q, pmax, 
                                     As, M_0, M_1, alpha, 
                                     self.fun_HOD, self.args['nthreads'],self.hcat['weight'], 
                                     self.args['satellites'], self.args['conformity_bias'])[:2]
        print(psat)
        if (pcent > 1).any():
            print(f'WARNING proba cent={pcent.max()} >1')
        results = Parallel(n_jobs=self.args['nthreads'], prefer="threads")(
            delayed(processInput)(k) for k in inputs)  # prefer="threads"
        if verbose :
            print(f'Done took {time.strftime("%H:%M:%S",time.gmtime(time.time() - start))}', flush=True)
        return results

    def genereate_training_points(self, nPoints, prior, sampling='lhs', save=False):
        """
        --- Generate random points in a LHS or Hammersley sample
        """
        print(f"Creating {sampling} sample...", flush=True)
        if sampling == 'lhs':
            new_k1 = np.zeros((len(prior.keys()), nPoints))
            k = lhsmdu.sample(len(prior.keys()), nPoints)
            for i, var in enumerate(prior.keys()):
                new_k1[i] = prior[var][0] + k[i]*(prior[var][1] - prior[var][0])
        elif 'Hammersley' in sampling :
            from idaes.surrogate.pysmo.sampling import HammersleySampling
            new_k1 = HammersleySampling(np.array(list(prior.values())).T.tolist(), number_of_samples=nPoints).sample_points().T
        else:
            raise ValueError('Wrong sampling type only "lhs" or "Hammersley" allowed')

        if save:
            np.savetxt(os.path.join(self.args["path_to_training_point"], f'point_{sampling}.txt'),
                       new_k1.T, header=str(prior))
        return new_k1.T

    def generate_training_sample(self, Npoints, prior, sampling='lhs', 
                                 reprise=0, niter=None, training_points=None, save = True, verbose = False):
        """
        --- Compute clustering measurement for training points, following a LHS or Hammersley sampling
        Regarder si je peux split 3 calculs de 20 cat sur 60 threads
        """
        
        os.makedirs(self.args['path_to_training_point'], exist_ok=True)
        training_points = self.genereate_training_points(Npoints, prior, sampling=sampling, save=save)
        if verbose:
            print(f"{len(training_points)} points in the training sample", flush=True)
    
        if verbose :
            print('#Compute HOD and clustering from training point...', flush=True)
            time_function= time.time()
        
        stop = len(training_points)
        if niter is not None: 
            stop = niter
        for j in range(reprise, stop):
            for ll, var in enumerate(prior.keys()):
                self.args[var] = training_points[j][ll]
            if '_base_' in self.args['sim_name']:
                print('Compute serial for base box')
                time_function_compute_parralel_chi2 = time.time()
                list_cats = [self.make_mock_cat() for i in range(self.args['nb_real'])]
                print(f'Time to compute {self.args["nb_real"]} cats : {time.strftime("%H:%M:%S",time.gmtime(time.time() - time_function_compute_parralel_chi2))}', flush=True)
            else:
                list_cats = self.compute_parallel_catalogues(
                ncat=self.args['nb_real'], verbose=verbose)
            if verbose : 
                print("Compute clustering measurement...", flush=True)
                time_function_compute_parralel_chi2 = time.time()
            if self.args["fit_type"] == "both":
                model_wp = []
                model_xi = []
                for k in range(self.args['nb_real']):
                    model_wp += [self.get_wp(list_cats[k], verbose=False)[1]]
                    model_xi += [self.get_2PCF(list_cats[k], verbose=False)[1]]
                dict_to_save = self.args.copy()
                dict_to_save['prior'] = prior
                dict_to_save["wp"] = model_wp
                dict_to_save["xi"] = model_xi
            elif self.args["fit_type"] == "wp":
                model_wp = np.zeros(
                    (self.args['nb_real'], self.args['n_rp_bins']))
                for k in range(self.args['nb_real']):
                    model_wp[k] = self.get_wp(list_cats[k], verbose=False)[1]
                dict_to_save = self.args.copy()
                dict_to_save['prior'] = prior
                dict_to_save["wp"] = model_wp
            elif self.args["fit_type"] == "xi":
                model_xi = []
                for k in range(self.args['nb_real']):
                    model_xi += [self.get_2PCF(list_cats[k], verbose=False)[1]]
                dict_to_save = self.args.copy()
                dict_to_save['prior'] = prior
                dict_to_save["xi"] = model_xi
            if verbose:
                print(f'Iteration {j} done',  ' '.join(map(str, training_points[j])), flush=True)
                print(f'Time to compute {self.args["nb_real"]} measurement : {time.strftime("%H:%M:%S",time.gmtime(time.time() - time_function_compute_parralel_chi2))}', flush=True)
                
            np.save(os.path.join(self.args["path_to_training_point"], f'{sampling}_{j}.npy'), dict_to_save)
        if verbose:
            print(f'Time to compute training sample clustering measurment {time.strftime("%H:%M:%S",time.gmtime(time.time() - time_function))}', flush=True)


    def create_tranning_file(self, nPoints_training, data_arr, D_data, D_model, fit_name,
                                 M_corr_model, M_cov_data, path_to_save_lhs_training_set=None,
                                 inf_index=None, sup_index=None, sampling="lhs",save=True):
        """
        --- Used precomputed clustering measurement in a LHS to compute chi^2 using data and generate the LHS training set for the gaussian process procedure
        """
        read_dictionary = {}
        u = 0
        for i in range(nPoints_training):
            try:
                read_dictionary["%d" % u] = np.load(os.path.join(self.args['path_to_training_point'],
                                                                 f"{sampling}_{i}.npy"), allow_pickle='TRUE').item()
                u += 1
            except FileNotFoundError:
                print(i, os.path.join(self.args['path_to_training_point'],
                                                                 f"{sampling}_{i}.npy"))
                pass
        
        ncat = read_dictionary["0"]["nb_real"]
        prior = read_dictionary['0']['prior']
        clust_type = self.args['fit_type']
        nvar = len(prior.keys())
        param_values = {col:[] for col in list(prior.keys())+['wp', 'xi', 'both']}        
        #nPoints = sum(len(files) for _, _, files in os.walk(self.args['path_to_training_point'])) - 1
        nPoints = sum([sampling in file for file in os.listdir(self.args['path_to_training_point'])]) -1 
        n_rbins = read_dictionary['0']['n_r_bins']
        print(nPoints)
        print(len(read_dictionary.keys()))
        print(clust_type)
        
        for i in range(nPoints):
            for var in prior.keys():
                param_values[var] += [read_dictionary[f"{i}"][var]]
            if clust_type == "both":
                param_values[clust_type] += [np.array([np.hstack((read_dictionary[f"{i}"]["wp"][ii], 
                                                            read_dictionary[f"{i}"]["xi"][ii].flatten())) for ii in range(ncat)])]
            elif clust_type == "xi":
                param_values[clust_type] += [np.array(read_dictionary[f"{i}"]["xi"]).reshape(ncat,2*read_dictionary[f"{i}"]['n_r_bins'])]
            
            elif clust_type == "wp":
                param_values[clust_type] += [read_dictionary[f"{i}"]["wp"][:, inf_index:sup_index]]

        chi_2_med = np.zeros(nPoints)
        rms_chi2 = np.zeros(nPoints)
        print(len(param_values[clust_type]))
        for i in range(nPoints):
            model = param_values[clust_type][i]
            chi2 = np.zeros(ncat)
            rms_model = model.std(axis=0)
            M_cov_model_w_corr = M_corr_model[inf_index:sup_index, inf_index:sup_index]*rms_model*rms_model[:, None]
            if self.args['add_sig2']:
                    try:
                        np.fill_diagonal(M_cov_model_w_corr, M_cov_model_w_corr.diagonal() + self.args['sig2_cosmic'])
                    except:
                        raise ('sig2_model not well define')
            
            for k in range(ncat):
                chi2[k] = compute_chi2(model[k], data_arr[inf_index:sup_index], np.linalg.inv(
                    M_cov_data[inf_index:sup_index, inf_index:sup_index]/(1-D_data)+M_cov_model_w_corr/(1-D_model)))
            chi_2_med[i] = np.mean(chi2)
            rms_chi2[i] = chi2.std()/np.sqrt(ncat)

        new_training = np.zeros((nPoints, nvar+2))
        for i, name in enumerate(prior.keys()):
            new_training[:, i] = param_values[name]
        new_training[:, nvar] = chi_2_med
        new_training[:, nvar+1] = rms_chi2
        
        if save & (path_to_save_lhs_training_set is not None):
            os.makedirs(path_to_save_lhs_training_set, exist_ok=True)
            np.savetxt(path_to_save_lhs_training_set, new_training, 
                       header=str(prior)+"\n"+fit_name)
        return new_training, prior
    
    
    @staticmethod
    def downsample_mock_cat (cat, ds_fac=0.1, mask=None):
        if mask is None:
            mask = np.random.uniform(size=len(cat['x'])) < ds_fac
        cat_ds = {}
        for var in cat.keys():
            cat_ds[var] = cat[var][mask]
        return cat_ds


    @staticmethod
    def _run_gp_mcmc(grid, prior, niter, args, dir_output_mcmc_chains, dir_output_file, 
                     fit_name, func_aq='EI', logchi2=False, EI_sample ='grille', kernel_gp='RBF',
                     nb_points=0, n_tot=10000, n_trim=10000, nwalkers=12, draw_3sig=None,
                     run_chains=True, length_scale=None, remove_edges=0.9,
                     length_scale_bounds=[0.001, 10], sampler='emcee', random_state=None, verbose = True):
        """
        --- Function which computes Gaussian process prediction from a given training sample, then compute a MCMC over the GP prediction, and returns the next point(s) using the input aquisition function for the iterative procedure
        """
        nvar = len(prior.keys())
        ranges = np.zeros((nvar, 4))
        for i, var in enumerate(prior.keys()):
            ranges[i, 0:2] = prior[var]
            ranges[i, 2] = np.mean(prior[var])
            ranges[i, 3] = np.diff(prior[var])

         # A revalider likelihood diverge prêt des bord
        if logchi2:
            X = grid[:, :nvar].copy()  # :198,0:6].copy()   #param
            y = np.log(grid[:, nvar])  # 198,6]          #chi_2
            dy = grid[:, nvar+1]/grid[:, nvar]  # :198,7]        #rms_chi2
            print("#ATTENTION ON EST EN LOG DE CHI2", flush=True)
        else:
            X = grid[:, :nvar].copy()  # :198,0:6].copy()   #param
            y = grid[:, nvar]  # 198,6]          #chi_2
            dy = grid[:, nvar+1]  # :198,7]        #rms_chi2

        length_scale = np.ones(nvar)
        if length_scale_bounds == "fix":
            length_scale = length_scale

        if kernel_gp == 'RBF':
            kernel = 1.0 * skg.kernels.RBF(length_scale=length_scale,
                                           length_scale_bounds=length_scale_bounds)
        elif kernel_gp == 'Matern_52':
            kernel = 1.0 * skg.kernels.Matern(length_scale=length_scale,
                                           length_scale_bounds=length_scale_bounds, nu=5/2)
            print(f"use kernel: {kernel_gp}", flush=True)
        if verbose:
            print(f"Running GPR iteration {niter}...", flush=True)
            start = time.time()
        gp = skg.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                          alpha=dy**2, random_state=random_state).fit(X, y)
        if verbose:
            print(f"GPR computed took {time.strftime('%H:%M:%S',time.gmtime(time.time() - start))}")
            print('#score=', gp.score(X, y), flush=True)
            print("#", gp.kernel_.get_params(), flush=True)

        if logchi2:
            def likelihood(x):
                if 'LH/2' in func_aq:
                    L = -np.exp(gp.predict(x.reshape(-1, nvar)))/4
                else:
                    L = -np.exp(gp.predict(x.reshape(-1, nvar)))/2
                # print(L,x)
                cond = np.abs(x-ranges[:, 2]) < (ranges[:, 3]/2)*remove_edges
                # print(cond)
                if cond.all():
                    return L
                else:
                    return -np.inf
        else:
            def likelihood(x):
                L = -gp.predict(x.reshape(-1, nvar))/2
                # print(L,x)
                cond = np.abs(x-ranges[:, 2]) < (ranges[:, 3]/2)*remove_edges
                # print(cond)
                if cond.all():
                    return L
                else:
                    return -np.inf
        ndim = nvar
        p0 = np.random.uniform(0, 1, (nwalkers, ndim))
        for k in range(ndim):
            p0[:, k] = (p0[:, k]-0.5)*ranges[k, 3] * \
                        0.8 + ranges[k, 2]  # 0.8 edges

        if verbose:
            print(f"Running MCMC iteration {niter}...", flush=True)
            start = time.time()
        if sampler == "zeus":
            sampler_mcmc = zeus.EnsembleSampler(
                nwalkers, ndim, likelihood)  # , args=[nvar])
            sampler_mcmc.run_mcmc(p0, n_tot)  # per walker
            chain = sampler_mcmc.get_chain(flat=True)
        elif sampler == "emcee":
            sampler_mcmc = emcee.EnsembleSampler(
                nwalkers, ndim, likelihood)  # , args=[nvar])
            sampler_mcmc.run_mcmc(p0, n_tot)  # per walker
            chain = sampler_mcmc.flatchain
        trimmed = chain[n_trim*4:]
        if verbose:
            print(f'MCMC computed took {time.strftime("%H:%M:%S",time.gmtime(time.time() - start))}', flush=True)
            print(trimmed[0], flush=True)
        new_points = trimmed[:, :nvar][np.random.randint(len(trimmed), size=nb_points)]
        
        ##### Retire le MCMC avec la bonne LH pour voir les contours corrects
        if 'LH/2' in func_aq:
            def likelihood(x):
                    L = -np.exp(gp.predict(x.reshape(-1, nvar)))/2
                    # print(L,x)
                    cond = np.abs(x-ranges[:, 2]) < (ranges[:, 3]/2)*remove_edges
                    # print(cond)
                    if cond.all():
                        return L
                    else:
                        return -np.inf
            ndim = nvar
            p0 = np.random.uniform(0, 1, (nwalkers, ndim))
            for k in range(ndim):
                p0[:, k] = (p0[:, k]-0.5)*ranges[k, 3] * \
                            0.8 + ranges[k, 2]  # 0.8 edges

            if verbose:
                print(f"Running MCMC iteration {niter}...", flush=True)
                start = time.time()
            if sampler == "zeus":
                sampler_mcmc = zeus.EnsembleSampler(
                    nwalkers, ndim, likelihood)  # , args=[nvar])
                sampler_mcmc.run_mcmc(p0, n_tot)  # per walker
                chain = sampler_mcmc.get_chain(flat=True)
            elif sampler == "emcee":
                sampler_mcmc = emcee.EnsembleSampler(
                    nwalkers, ndim, likelihood)  # , args=[nvar])
                sampler_mcmc.run_mcmc(p0, n_tot)  # per walker
                chain = sampler_mcmc.flatchain
            trimmed = chain[n_trim*4:]
            if verbose:
                print(f'MCMC computed took {time.strftime("%H:%M:%S",time.gmtime(time.time() - start))}', flush=True)
                print(trimmed[0], flush=True)
            
            
        
        if draw_3sig : 
            #Test non concluant pour tiré dans le 3sigma de la chaine
            new_points = np.concatenate(
                (new_points,  np.quantile(trimmed[:,:nvar], q=[0.01,.99], axis=0)), axis=None).reshape(nb_points+2, nvar)

        Gp_pred = gp.predict(trimmed, return_std=True)
        ind = np.where(Gp_pred[0] == Gp_pred[0].min())[0][0]
        pred = gp.predict(ranges[:, 2].reshape(-1, nvar),
                          return_std=True)  # Best fit pred
        if verbose:
            print("#Pred at fix point (mean values of each params):",
                  ranges[:, 2], pred, flush=True)
            print("#best GP prediction:",
                  trimmed[ind], Gp_pred[0][ind], Gp_pred[1][ind], flush=True)

        def multivariate_gelman_rubin(chains):
            """
            Arnaud de Mattia code's
            http://www.stat.columbia.edu/~gelman/research/published/brooksgelman2.pdf
            dim 0: nchains
            dim 1: nsteps
            dim 2: ndim
            """
            nchains = len(chains)
            mean = np.asarray([np.mean(chain, axis=0) for chain in chains])
            variance = np.asarray([np.cov(chain.T, ddof=1)
                                   for chain in chains])
            nsteps = np.asarray([len(chain) for chain in chains])
            Wn1 = np.mean(variance, axis=0)
            Wn = np.mean(((nsteps-1.)/nsteps)[:, None, None]*variance, axis=0)
            B = np.cov(mean.T, ddof=1)
            V = Wn + (nchains+1.)/nchains*B
            invWn1 = np.linalg.inv(Wn1)
            assert np.absolute(
                Wn1.dot(invWn1)-np.eye(Wn1.shape[0])).max() < 1e-5
            eigen = np.linalg.eigvalsh(invWn1.dot(V))
            return eigen.max()

        multi_GR = multivariate_gelman_rubin(
            sampler_mcmc.get_chain().transpose([1, 0, 2])[:, 2500:, :])
        if verbose:
            print("#multivariate_gelman_rubin: ", multi_GR, flush=True)

        if logchi2:
            res = np.zeros((len(trimmed), nvar+2))
            res[:, :nvar] = trimmed
            res[:, nvar] = np.exp(Gp_pred[0])
            res[:, -1] = np.exp(Gp_pred[0])*Gp_pred[1]
        else:
            res = np.zeros((len(trimmed), nvar+2))
            res[:, :nvar] = trimmed
            res[:, nvar] = Gp_pred[0]
            res[:, -1] = Gp_pred[1]
        np.savetxt(os.path.join(dir_output_mcmc_chains,f'chain_{nvar}p_{fit_name}_{niter}.txt'), res)

        ### Expected improvement
        if func_aq == 'EI':
            if EI_sample == 'grille':
                nb = 100000
                test = np.random.uniform(size=(nb, nvar))
                for i in range(nvar):
                    test[:, i] = ranges[i][0] + ranges[i][3] * \
                        0.05 + test[:, i]*(ranges[i][3]*remove_edges)
            elif EI_sample == 'MCMC':
                test = trimmed[:,:nvar]

            def expected_improvement(X, X_sample, Y_sample, gpr):
                mu, sigma = gpr.predict(X, return_std=True)
                mu_sample = gpr.predict(X_sample)
                EI = (mu_sample.min() - mu) * norm.cdf((mu_sample.min() - mu)
                                                       / sigma) + sigma*norm.pdf((mu_sample.min() - mu) / sigma)
                return EI, mu_sample.min()

            EI, xmin_samp = expected_improvement(test, X, y, gp)
            mu_EI, sig_EI = gp.predict(
                test[EI.argmax()].reshape(-1, nvar), return_std=True)
            #if xmin_samp < (mu_EI - sig_EI) : sys.exit("Procedure stop xmin_samp < EI pred")
            if verbose:
                print("EI.max = ", EI.max())
                print("New point EI :", test[EI.argmax()], mu_EI, sig_EI)
                print("xmin_samp", xmin_samp, flush=True)
            new_points = np.concatenate(
                (new_points, test[EI.argmax()]), axis=None).reshape(nb_points+1, nvar)

        if niter == 0:
            list_lenghtscale = []
            for i in range(nvar):
                list_lenghtscale.append("ls%d" % i)
            f = open(os.path.join(dir_output_file, f'output_GP_{nvar}p_{fit_name}.txt'), "w")
            if func_aq == 'EI': 
                f.write("N_iter GPscore GP_predfix Er_predfix BestPred Er_BestPred multivariate_gelman_rubin EI_max GP_EI Err_EI xmin_samp "
                        + ' '.join(map(str, prior.keys()))+" "
                        + ' '.join(map(str, list_lenghtscale))+"\n")
                f.write(str(niter)+" "+str(gp.score(X, y))+" "
                        + str(pred[0][0])+" "+str(pred[1][0])+" "
                        + str(Gp_pred[0][ind])+" "
                        + str(Gp_pred[1][ind])+" " + str(multi_GR)+" "
                        + str(EI.max())+" "+str(mu_EI[0])+" "
                        + str(sig_EI[0])+" "+str(xmin_samp)+" "
                        + ' '.join(map(str, trimmed[ind]))+" "
                        + ' '.join(map(str, gp.kernel_.get_params(False)["k2"].length_scale))+"\n")
                f.close()
            else: 
                f.write("N_iter GPscore GP_predfix Er_predfix BestPred Er_BestPred multivariate_gelman_rubin "
                        + ' '.join(map(str, prior.keys()))+" "
                        + ' '.join(map(str, list_lenghtscale))+"\n")
                f.write(str(niter)+" "+str(gp.score(X, y))+" "
                        + str(pred[0][0])+" "+str(pred[1][0])+" "
                        + str(Gp_pred[0][ind])+" "
                        + str(Gp_pred[1][ind])+" " + str(multi_GR)+" "
                        + ' '.join(map(str, trimmed[ind]))+" "
                        + ' '.join(map(str, gp.kernel_.get_params(False)["k2"].length_scale))+"\n")
                f.close()
        else:
            f = open(os.path.join(dir_output_file, f'output_GP_{nvar}p_{fit_name}.txt'), "a")
            if func_aq == 'EI': 
                f.write(str(niter)+" "+str(gp.score(X, y))+" "
                        + str(pred[0][0])+" "+str(pred[1][0])+" "
                        + str(Gp_pred[0][ind])+" "
                        + str(Gp_pred[1][ind])+" " + str(multi_GR)+" "
                        + str(EI.max())+" "+str(mu_EI[0])+" "
                        + str(sig_EI[0])+" "+str(xmin_samp)+" "
                        + ' '.join(map(str, trimmed[ind]))+" "
                        + ' '.join(map(str, gp.kernel_.get_params(False)["k2"].length_scale))+"\n")
                f.close()
            else: 
                f.write(str(niter)+" "+str(gp.score(X, y))+" "
                    + str(pred[0][0])+" "+str(pred[1][0])+" "
                    + str(Gp_pred[0][ind])+" "
                    + str(Gp_pred[1][ind])+" " + str(multi_GR)+" "
                    + ' '.join(map(str, trimmed[ind]))+" "
                    + ' '.join(map(str, gp.kernel_.get_params(False)["k2"].length_scale))+"\n")
                f.close()

        return new_points

    def run_fit(self, fit_name, prior, data_arr, M_corr, M_cov_data, D_data, D_model, training_point, 
                dir_output_mcmc_chains, dir_output_file, sampling='lhs', logchi2=True, 
                inf_index=None, sup_index=None, kernel_gp='RBF', draw_3sig=None,
                sampler="emcee", n_tot=10000, n_trim=10000, nwalkers=12, nb_points=0, n_calls=400, 
                func_aq='EI', EI_sample='grille',
                length_scale=None, length_scale_bounds=[0.001, 10], reprise=False, remove_edges=0.9, 
                norm_param=False, nmock=20, scale_As=1, verbose = True):
        """
        --- Run the iterative procedure. 
        """
        os.makedirs(dir_output_file, exist_ok=True)
        os.makedirs(dir_output_mcmc_chains, exist_ok=True)
        
        # do check before run fit
        if self.args['fit_type'] == 'wp':
            if len(data_arr) != self.args['n_rp_bins'] :
                raise ValueError ('data and bin number n_rp_bins are not egal')
            if M_corr.shape != M_cov_data.shape :
                raise ValueError ("M_cov_data and M_corr don't have the same number of bins")
            if  len(M_corr)!= self.args['n_rp_bins'] :
                raise ValueError ('data and bin number n_rp_bins are not egal')
                
        elif self.args['fit_type'] == 'xi':
            if len(data_arr) != self.args['n_r_bins'] :
                raise ValueError ('data and bin number n_r_bins are not egal')
            if M_corr.shape != M_cov_data.shape :
                raise ValueError("M_cov_data and M_corr don't have the same number of bins")
            if  len(M_corr)!= self.args['n_r_bins'] :
                raise ValueError ('M_cov_data and bin number n_r_bins are not egal')
                
        elif self.args['fit_type'] == 'both':
            if M_corr.shape != M_cov_data.shape :
                raise ValueError (f"M_cov_data and M_corr don't have the same shape: {M_cov_data.shape},{M_corr.shape}")
                
            if (self.args['edges_rppi'] is None) & (self.args['edges_rppi'] is None): 
                if (len(data_arr) != (self.args['n_rp_bins']+self.args['n_r_bins']*2)):
                    raise ValueError (f"data and bin number 2*n_r_bins + n_rp_bins: {len(data_arr)}, {self.args['n_rp_bins']},{self.args['n_r_bins']}")
                
                if len(M_corr) != self.args['n_rp_bins']+self.args['n_r_bins']*2:
                    raise ValueError ('M_cov_data and bin number 2*n_r_bins + n_rp_bins or edges smu/rppi are not equal')    
            
            else:
                if len(data_arr) != (self.args['edges_rppi'][0].size - 1) + (self.args['edges_smu'][0].size - 1)*len(self.args['multipole_index']):
                    raise ValueError (f"data and bin number 2*n_r_bins + n_rp_bins: {len(data_arr)}, {self.args['n_rp_bins']},{self.args['n_r_bins']}")
                    
                if len(M_corr) != (self.args['edges_rppi'][0].size - 1) + (self.args['edges_smu'][0].size - 1)*len(self.args['multipole_index']):
                    raise ValueError ('M_cov_data and bin number 2*n_r_bins + n_rp_bins or edges smu/rppi are not equal')    
        else : 
            raise ValueError (f'fit_type {self.args["fit_type"]} is not correct, only "wp", "xi"or "both" are supported')
        
        nvar = len(prior.keys())
        iter = 0
        if reprise:
            output_point = pd.read_csv(os.path.join(dir_output_file, f"{nvar}p_{fit_name}.txt"), sep=" ", comment="#")
            training_point = np.concatenate(
                (training_point, output_point.values[:, 1:1+nvar+2]))
            iter = output_point["N_iter"].loc[len(output_point)-1]+1
            p = np.loadtxt(os.path.join(dir_output_mcmc_chains, f'chain_{nvar}p_{fit_name}_{iter-1}.txt'))[:,:nvar]
            D_kl = 10
            if verbose :
                print("#reprise ", iter, "len param point ",
                      len(training_point), flush=True)
                
        print("Run gpmcmc...", flush=True)
        for j in range(iter, n_calls):
            if verbose :
                print(f'Iteration {j}...', flush=True)
                time_compute_mcmc = time.time()
            new_param = self._run_gp_mcmc(training_point, prior, dir_output_mcmc_chains=dir_output_mcmc_chains, 
                                          fit_name=fit_name,dir_output_file=dir_output_file, niter=j, args=self.args,
                                          func_aq=func_aq, EI_sample=EI_sample, logchi2=logchi2, nb_points=nb_points,
                                          length_scale=length_scale, random_state=None, kernel_gp=kernel_gp,
                                          length_scale_bounds=length_scale_bounds, sampler=sampler, draw_3sig=draw_3sig,
                                          n_tot=n_tot, n_trim=n_trim, nwalkers=nwalkers, remove_edges=remove_edges)
            if verbose :
                print("#time_compute_gpmcmc =", time.time()
                      - time_compute_mcmc, flush=True)


            ### Test de Kullback Leibler
            D_kl1 = 10
            if j > 0:
                if j == 1:
                    q = np.loadtxt(os.path.join(dir_output_mcmc_chains, f'chain_{nvar}p_{fit_name}_{0}.txt'))[:,:nvar]
                    p = np.loadtxt(os.path.join(dir_output_mcmc_chains, f'chain_{nvar}p_{fit_name}_{1}.txt'))[:,:nvar]
                    D_kl = np.array([])
                else:
                    q = p
                    p = np.loadtxt(os.path.join(dir_output_mcmc_chains, f'chain_{nvar}p_{fit_name}_{j}.txt'))[:,:nvar]
                n_dim = nvar
                cov_q = np.cov(q.T)
                cov_p = np.cov(p.T)
                inv_cov_q = np.linalg.inv(cov_q)
                mean_q = np.mean(q, axis=0)
                mean_p = np.mean(p, axis=0)
                D_kl1 = 0.5 * (np.log10(np.linalg.det(cov_q) / np.linalg.det(cov_p)) - n_dim + np.trace(np.matmul(
                    inv_cov_q, cov_p)) + np.matmul((mean_q - mean_p).T, np.matmul(inv_cov_q, (mean_q - mean_p))))
                D_kl = np.append(D_kl1, D_kl)
                # print (j, D_kl)
                # if len(D_kl) > 5:
                #     if (D_kl[-5:] < 0.1).all():
                #         sys.exit("Procedure converged at iteration %d!" % j)

            #Compute chi2
            res = np.zeros((len(new_param), nvar+2))
            time_boucle_over_param_mcmc = time.time()

            #### old non parallel
            if verbose :
                print("#run old parralel chi2 points")
            if j == 0:
                f = open(os.path.join(dir_output_file, f"{nvar}p_{fit_name}.txt"), "w")
                f.write("N_iter "+' '.join(map(str, prior.keys()))
                        + " chi_2_med rms_chi2 D_kl1\n")
                f.close()
            for i in range(len(new_param)):
                if norm_param:
                    for ll, var in enumerate(prior.keys()):
                        self.args[var] = new_param[i][ll]*np.mean(prior[var])
                else:
                    for ll, var in enumerate(prior.keys()):
                        if var == "As":
                            self.args[var] = new_param[i][ll]*scale_As
                        else:
                            self.args[var] = new_param[i][ll]
                if verbose :
                    print("#", i, new_param[i], [self.args[var] for var in prior.keys()], flush=True)
                    time_function_compute_parralel_chi2 = time.time()
                if '_base_' in self.args['sim_name']:
                    print('Compute serial for base box')
                    time_function_compute_parralel_chi2 = time.time()                
                    results = [self.make_mock_cat() for i in range(nmock)]
                    print(f'Time to compute {nmock} cats : {time.strftime("%H:%M:%S",time.gmtime(time.time() - time_function_compute_parralel_chi2))}', flush=True)
                else:
                    results = self.compute_parallel_catalogues(
                    ncat=nmock, verbose=verbose)

                chi2 = np.zeros(nmock)                
                time_function_compute_parralel_chi2 = time.time()
                

                if self.args["fit_type"] == "wp":
                    model = [self.get_wp(res)[1] for res in results]
                elif self.args["fit_type"] == "xi":
                    model = [self.get_2PCF(res)[1].flatten() for res in results]
                elif self.args["fit_type"] == "both":
                    model = [np.hstack((self.get_wp(res)[1], self.get_2PCF(res)[1].flatten())) for res in results]
                        
                
                if verbose :                                                            
                    print("#time_compute_2PCFs =",
                          time.strftime("%H:%M:%S",time.gmtime(time.time()-time_function_compute_parralel_chi2)),
                          flush=True)
                model = np.array(model)[:, inf_index:sup_index]
                rms_model = model.std(axis=0)
                M_cov_model_w_corr = M_corr[inf_index:sup_index, inf_index:sup_index]*rms_model*rms_model[:, None]   
                if self.args['add_sig2']:
                    try:
                        np.fill_diagonal(M_cov_model_w_corr, M_cov_model_w_corr.diagonal() + self.args['sig2_cosmic'])
                    except:
                        print('add_sig2 failed, continue without')
                for k in range(nmock):
                    chi2[k] = compute_chi2(model[k], data_arr[inf_index:sup_index], np.linalg.inv(
                        M_cov_data[inf_index:sup_index, inf_index:sup_index]/(1-D_data)+M_cov_model_w_corr/(1-D_model)))
                del results
                chi_2_med = np.mean(chi2)
                rms_chi2 = chi2.std()/np.sqrt(nmock)
                if verbose :
                    print(' '.join(map(str, new_param[i])),
                      chi_2_med, rms_chi2, D_kl1, flush=True)
                res[i] = np.concatenate(
                    (new_param[i], [chi_2_med, rms_chi2]), axis=0)
                f = open(os.path.join(dir_output_file, f"{nvar}p_{fit_name}.txt"), "a")
                f.write(str(str(j)+" "+' '.join(map(str, new_param[i])))+" "+str(
                    chi_2_med)+" "+str(rms_chi2)+" "+str(D_kl1)+"\n")
                f.close()
            training_point = np.concatenate((training_point, res))
            if verbose :
                print(f'Iteration {j} done, took {time.strftime("%H:%M:%S",time.gmtime(time.time()-time_compute_mcmc))}', flush=True)

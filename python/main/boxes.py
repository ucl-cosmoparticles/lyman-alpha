import math as mh
import random as rd
import numpy as np
import numpy.random as npr
import scipy.integrate as spi
import copy as cp
import astropy.units as u
import spectra as sa
import griddedspectra as gs
import randspectra as rs
import sys

from power_spectra import *
from utils import *

class Box(object):
    """Class to generate a box of fluctuations"""
    def __init__(self,redshift,H0,omega_m):
        self._redshift = redshift
        self._H0 = H0
        self._omega_m = omega_m
        self.convert_fourier_units_to_distance = False

    def k_i(self,i):
        if self.convert_fourier_units_to_distance == False:
            box_units = self.voxel_velocities[i]
        else:
            box_units = self.voxel_lens[i]
        return np.fft.fftfreq(self._n_samp[i], d=box_units)

    def k_z_mod_box(self): #Generalise to any k_i
        x = np.zeros_like(self.k_i('x'))[:, np.newaxis, np.newaxis]
        y = np.zeros_like(self.k_i('y'))[np.newaxis, :, np.newaxis]
        z = self.k_i('z')[np.newaxis, np.newaxis, :]
        return x + y + np.absolute(z)

    def k_perp_box(self): #Generalise to any pair of k_i
        x = self.k_i('x')[:, np.newaxis, np.newaxis]
        y = self.k_i('y')[np.newaxis, :, np.newaxis]
        z = np.zeros_like(self.k_i('z'))[np.newaxis, np.newaxis, :]
        return np.sqrt(x**2 + y**2) + z

    def k_box(self):
        x = self.k_i('x')[:,np.newaxis,np.newaxis]
        y = self.k_i('y')[np.newaxis,:,np.newaxis]
        z = self.k_i('z')[np.newaxis,np.newaxis,:]
        return np.sqrt(x**2 + y**2 + z**2)

    def mu_box(self):
        x = self.k_i('x')[:, np.newaxis, np.newaxis]
        y = self.k_i('y')[np.newaxis, :, np.newaxis]
        z = self.k_i('z')[np.newaxis, np.newaxis, :]
        return z / np.sqrt(x**2 + y**2 + z**2)

    def hubble_z(self):
        return self._H0 * np.sqrt(self._omega_m * (1 + self._redshift) ** 3 + 1. - self._omega_m)


class GaussianBox(Box):
    """Sub-class to generate a box of fluctuations from a Gaussian random field"""
    def __init__(self,x_max,n_samp,redshift,H0,omega_m):
        self._x_max = x_max #Tuples for 3 dimensions
        self._n_samp = n_samp
        super(GaussianBox, self).__init__(redshift,H0,omega_m)

        self.voxel_lens = {}
        self.voxel_velocities = {}
        for i in ['x','y','z']:
            self.voxel_lens[i] = self._x_max[i] / (self._n_samp[i] - 1)
            self.voxel_velocities[i] = self.voxel_lens[i] * self.hubble_z()

    def _gauss_realisation(self, power_evaluated, k_box): #Really want Hermitian Fourier modes
        gauss_k=np.sqrt(0.5*power_evaluated)*(npr.standard_normal(size=power_evaluated.shape)+npr.standard_normal(size=power_evaluated.shape)*1.j)
        gauss_k[k_box == 0.] = 0. #Zeroing the mean
        return np.fft.ifftn(gauss_k, s=(self._n_samp['x'], self._n_samp['y'], self._n_samp['z']), axes=(0, 1, 2))

    def isotropic_power_law_gauss_realisation(self,pow_index,pow_pivot,pow_amp):
        box_spectra = PowerLawPowerSpectrum(pow_index, pow_pivot, pow_amp)
        power_evaluated = box_spectra.evaluate3d_isotropic(self.k_box())
        return self._gauss_realisation(power_evaluated,self.k_box())

    def anisotropic_power_law_gauss_realisation(self, pow_index, pow_pivot, pow_amp, mu_coefficients):
        box_spectra = PowerLawPowerSpectrum(pow_index, pow_pivot, pow_amp)
        box_spectra.set_anisotropic_functional_form(mu_coefficients)
        power_evaluated = box_spectra.evaluate3d_anisotropic(self.k_box(),self.mu_box())
        return self._gauss_realisation(power_evaluated,self.k_box())

    def isotropic_pre_computed_gauss_realisation(self,fname):
        box_spectra = PreComputedPowerSpectrum(fname)
        power_evaluated = box_spectra.evaluate3d_isotropic(self.k_box())
        return self._gauss_realisation(power_evaluated,self.k_box())

    def anisotropic_pre_computed_gauss_realisation(self, fname, mu_coefficients):
        box_spectra = PreComputedPowerSpectrum(fname)
        box_spectra.set_anisotropic_functional_form(mu_coefficients)
        power_evaluated = box_spectra.evaluate3d_anisotropic(self.k_box(),self.mu_box())
        return self._gauss_realisation(power_evaluated,self.k_box())

    def isotropic_CAMB_gauss_realisation(self):
        return 0

    def anisotropic_CAMB_gauss_realisation(self):
        return 0


class SimulationBox(Box):
    """Sub-class to generate a box of Lyman-alpha spectra drawn from Simeon's simulations"""
    def __init__(self,snap_num,snap_dir,grid_samps,spectrum_resolution,reload_snapshot=True,spectra_savefile_root='gridded_spectra'):
        self._n_samp = {}
        self._n_samp['x'] = grid_samps
        self._n_samp['y'] = grid_samps

        self.voxel_lens = {}
        self.voxel_velocities = {}

        self._snap_num = snap_num
        self._snap_dir = snap_dir
        self._grid_samps = grid_samps
        self._spectrum_resolution = spectrum_resolution
        self._reload_snapshot = reload_snapshot
        self._spectra_savefile_root = spectra_savefile_root

        self.spectra_savefile = '%s_%i_%i.hdf5'%(self._spectra_savefile_root,self._grid_samps,self._spectrum_resolution.value)

        self.element = 'H'
        self.ion = 1
        self.line_wavelength = 1215 * u.angstrom

        self.spectra_instance = gs.GriddedSpectra(self._snap_num,self._snap_dir,nspec=self._grid_samps,res=self._spectrum_resolution.value,savefile=self.spectra_savefile,reload_file=self._reload_snapshot)
        self._n_samp['z'] = int(self.spectra_instance.vmax / self.spectra_instance.dvbin)
        super(SimulationBox, self).__init__(self.spectra_instance.red, (self.spectra_instance.hubble * 100. * u.km) / (u.s * u.Mpc), self.spectra_instance.OmegaM)
        self.voxel_velocities['x'] = (self.spectra_instance.vmax / self._n_samp['x']) * (u.km / u.s)
        self.voxel_velocities['y'] = (self.spectra_instance.vmax / self._n_samp['y']) * (u.km / u.s)
        self.voxel_velocities['z'] = self.spectra_instance.dvbin * (u.km / u.s)
        print("Size of voxels in velocity units =", self.voxel_velocities)
        for i in ['x','y','z']:
            #self.voxel_lens[i] = self.voxel_velocities[i] / self.hubble_z()
            self.voxel_lens[i] = (self.spectra_instance.box / (self._n_samp[i] * self.spectra_instance.hubble)) * u.kpc

        self._col_dens_threshold = 2.e+20 / (u.cm * u.cm) #Default values
        self._dodge_dist = 10.*u.kpc

    def _generate_general_spectra_instance(self,cofm):
        axis = np.ones(cofm.shape[0])
        return sa.Spectra(self._snap_num, self._snap_dir, cofm, axis, res=self._spectrum_resolution.value, reload_file=True)

    def _get_optical_depth(self):
        tau = self.spectra_instance.get_tau(self.element, self.ion, int(self.line_wavelength.value))  # SLOW if not reloading
        self.spectra_instance.save_file()  # Save spectra to file
        return tau

    def _get_column_density(self):
        col_dens = self.spectra_instance.get_col_density(self.element, self.ion) / (u.cm * u.cm) #SLOW if not reloading
        self.spectra_instance.save_file()
        return col_dens

    def skewers_realisation(self):
        tau = self._get_optical_depth()
        delta_flux = np.exp(-1.*tau) / np.mean(np.exp(-1.*tau)) - 1.
        return delta_flux.reshape((self._grid_samps,self._grid_samps,-1))

    def _get_skewers_with_DLAs_bool_arr_simple_threshold(self,col_dens):
        return np.max(col_dens, axis=-1) > self._col_dens_threshold

    def _get_skewers_with_DLAs_bool_arr_local_sum_threshold(self, col_dens):
        size_of_bin_in_velocity = 100. * (u.km / u.s) #self.voxel_velocities['z'] #MAKE INPUT ARGUMENT!!!
        size_of_bin_in_samples = round(size_of_bin_in_velocity.value / self.voxel_velocities['z'].value)
        print("\nSize of bin in samples = %i" %size_of_bin_in_samples)
        col_dens_local_sum = (calculate_local_average_of_array(col_dens.value,size_of_bin_in_samples)*size_of_bin_in_samples)/(u.cm * u.cm)
        return self._get_skewers_with_DLAs_bool_arr_simple_threshold(col_dens_local_sum)

    def _get_skewers_with_DLAs_bool_arr(self,col_dens):
        assert is_astropy_quantity(col_dens)
        #return self._get_skewers_with_DLAs_bool_arr_simple_threshold(col_dens)
        return self._get_skewers_with_DLAs_bool_arr_local_sum_threshold(col_dens)

    def _get_optical_depth_for_new_skewers(self, skewers_with_DLAs_bool_arr):
        new_skewers_cofm = self.spectra_instance.cofm[skewers_with_DLAs_bool_arr] #Slicing out new skewers
        new_tau = self._generate_general_spectra_instance(new_skewers_cofm).get_tau(self.element, self.ion, int(self.line_wavelength.value))
        self.spectra_instance.tau[(self.element, self.ion, int(self.line_wavelength.value))][skewers_with_DLAs_bool_arr] = new_tau

    def _get_column_density_for_new_skewers(self, skewers_with_DLAs_bool_arr):
        new_skewers_cofm = self.spectra_instance.cofm[skewers_with_DLAs_bool_arr]  # Slicing out skewers with DLA's
        new_col_dens = self._generate_general_spectra_instance(new_skewers_cofm).get_col_density(self.element,self.ion) / (u.cm * u.cm)
        self.spectra_instance.colden[(self.element, self.ion)][skewers_with_DLAs_bool_arr] = new_col_dens.value  # Upd. col dens

    def _form_skewers_realisation_dodging_DLAs_single_iteration(self,skewers_with_DLAs_bool_arr):
        print("Number of skewers with DLA's = %i" % np.sum(skewers_with_DLAs_bool_arr))
        self.spectra_instance.cofm[skewers_with_DLAs_bool_arr, 1] += self._dodge_dist.value  # Dodging in y-axis
        self._get_column_density_for_new_skewers(skewers_with_DLAs_bool_arr)
        skewers_with_DLAs_bool_arr = self._get_skewers_with_DLAs_bool_arr(self.spectra_instance.colden[(self.element,self.ion)] / (u.cm * u.cm))
        return skewers_with_DLAs_bool_arr

    def _get_column_density_for_new_skewers_loop(self, skewers_with_DLAs_bool_arr):
        while np.sum(skewers_with_DLAs_bool_arr) > 0: #Continue dodging while there remains DLA's
            skewers_with_DLAs_bool_arr = self._form_skewers_realisation_dodging_DLAs_single_iteration(skewers_with_DLAs_bool_arr)

    def _save_new_skewers_realisation_dodging_DLAs(self,savefile_root):
        savefile_tuple = (self._snap_dir,self._snap_num,savefile_root,self._grid_samps,self._spectrum_resolution.value)
        self.spectra_instance.savefile = '%s/snapdir_00%i/%s_%i_%i.hdf5' % savefile_tuple
        self.spectra_instance.save_file()

    def form_skewers_realisation_dodging_DLAs(self, col_dens_threshold = 2.e+20 / (u.cm * u.cm), dodge_dist=10.*u.kpc, savefile_root='gridded_spectra_DLAs_dodged'):
        self._col_dens_threshold = col_dens_threshold #Update if changed
        self._dodge_dist = dodge_dist
        skewers_with_DLAs_bool_arr = self._get_skewers_with_DLAs_bool_arr(self._get_column_density())
        self._get_column_density_for_new_skewers_loop(skewers_with_DLAs_bool_arr)
        self._get_optical_depth_for_new_skewers(skewers_with_DLAs_bool_arr)
        self._save_new_skewers_realisation_dodging_DLAs(savefile_root)
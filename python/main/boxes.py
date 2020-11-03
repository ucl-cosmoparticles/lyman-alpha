import math as mh
import random as rd
import numpy as np
import numpy.random as npr
import scipy.integrate as spi
import scipy.optimize as spo
import copy as cp
import astropy.units as u

from fake_spectra import spectra as sa
from fake_spectra import griddedspectra as gs
from fake_spectra import randspectra as rs

import sys

from power_spectra import *
from utils import *

class Box(object):
    """Class to generate a box of fluctuations"""
    def __init__(self,redshift,H0,omega_m,nskewers):
        self._redshift = redshift
        self._H0 = H0
        self._omega_m = omega_m

        self.nskewers = nskewers
        self.scale_factor = 1. / (1. + self._redshift)
        self.convert_fourier_units_to_distance = False

    def _set_box_units(self,i):
        if self.convert_fourier_units_to_distance == False:
            return self.voxel_velocities[i]
        else:
            return self.voxel_lens[i]

    #1D coordinate arrays
    def k_i(self,i):
        box_units = self._set_box_units(i)
        if i == 'z':
            return np.fft.fftfreq(self._n_samp[i], d=box_units) * 2. * mh.pi #2 * pi for "cosmologist's k"
        else:
            return np.fft.fftfreq(self._n_samp[i], d=box_units) * 2. * mh.pi

    def k_z_mod(self):
        box_units = self._set_box_units('z')
        return np.fft.rfftfreq(self._n_samp['z'], d=box_units) * 2. * mh.pi

    #Cylindrical coordinate system
    def k_z_mod_box(self):
        x = np.zeros_like(self.k_i('x'))[:, np.newaxis, np.newaxis]
        y = np.zeros_like(self.k_i('y'))[np.newaxis, :, np.newaxis]
        z = self.k_i('z')[np.newaxis, np.newaxis, :]
        return x + y + np.absolute(z)

    def k_perp_box(self):
        x = self.k_i('x')[:, np.newaxis, np.newaxis]
        y = self.k_i('y')[np.newaxis, :, np.newaxis]
        z = np.zeros_like(self.k_i('z'))[np.newaxis, np.newaxis, :]
        return np.sqrt(x**2 + y**2) + z

    #Spherical coordinate system
    def k_box(self):
        x = self.k_i('x')[:,np.newaxis,np.newaxis]
        y = self.k_i('y')[np.newaxis,:,np.newaxis]
        z = self.k_i('z')[np.newaxis,np.newaxis,:]
        return np.sqrt(x**2 + y**2 + z**2)

    def mu_box(self):
        x = self.k_i('x')[:, np.newaxis, np.newaxis]
        y = self.k_i('y')[np.newaxis, :, np.newaxis]
        z = self.k_i('z')[np.newaxis, np.newaxis, :]
        k = np.sqrt(x**2 + y**2 + z**2)
        k[k == 0.] = np.nan
        return z / k

    #Configuration space coordinates
    def r_i(self,i):
        box_units = self._set_box_units(i)
        return np.arange(self._n_samp[i]) * box_units

    def r_box(self):
        x = self.r_i('x')[:,np.newaxis,np.newaxis]
        y = self.r_i('y')[np.newaxis,:,np.newaxis]
        z = self.r_i('z')[np.newaxis,np.newaxis,:]
        return np.sqrt(x**2 + y**2 + z**2)

    def mu_r_box(self):
        x = self.r_i('x')[:, np.newaxis, np.newaxis]
        y = self.r_i('y')[np.newaxis, :, np.newaxis]
        z = self.r_i('z')[np.newaxis, np.newaxis, :]
        r = np.sqrt(x**2 + y**2 + z**2)
        r[r == 0.] = np.nan
        return z / r

    def hubble_z(self):
        return self._H0 * np.sqrt(self._omega_m * (1 + self._redshift) ** 3 + 1. - self._omega_m)


class GaussianBox(Box):
    """Sub-class to generate a box of fluctuations from a Gaussian random field"""
    def __init__(self,x_max,n_samp,redshift,H0,omega_m):
        self._x_max = x_max #Tuples for 3 dimensions - comoving in Mpc
        self._n_samp = n_samp
        nskewers = self._n_samp['x'] * self._n_samp['y']
        super(GaussianBox, self).__init__(redshift,H0,omega_m,nskewers)

        self.voxel_lens = {}
        self.voxel_velocities = {}
        for i in ['x','y','z']:
            self.voxel_lens[i] = self._x_max[i] / (self._n_samp[i] - 1)
            self.voxel_velocities[i] = self.voxel_lens[i] * self.hubble_z() * self.scale_factor

        self._num_voigt = 0
        self.num_clean_skewers = self.nskewers
        self._voigt_profile_skewers_index_arr = np.zeros(self.nskewers)
        self._voigt_profile_skewers_bool_arr = np.zeros(self.nskewers, dtype=bool)

    def _gauss_realisation(self, power_evaluated, k_box):
        gauss_k=np.sqrt(0.5*power_evaluated)*(npr.standard_normal(size=power_evaluated.shape)+npr.standard_normal(size=power_evaluated.shape)*1.j)
        gauss_k[k_box == 0.] = 0. #Zeroing the mean
        gauss_k_hermitian = make_box_hermitian(gauss_k)
        return np.fft.ifftn(gauss_k_hermitian, s=(self._n_samp['x'], self._n_samp['y'], self._n_samp['z']), axes=(0, 1, 2))

    def isotropic_power_law_gauss_realisation(self,pow_index,pow_pivot,pow_amp):
        box_spectra = PowerLawPowerSpectrum(pow_index, pow_pivot, pow_amp)
        power_evaluated = box_spectra.evaluate3d_isotropic(self.k_box())
        return self._gauss_realisation(power_evaluated,self.k_box())

    def anisotropic_power_law_gauss_realisation(self, pow_index, pow_pivot, pow_amp, mu_coefficients):
        box_spectra = PowerLawPowerSpectrum(pow_index, pow_pivot, pow_amp)
        box_spectra.set_anisotropic_functional_form(mu_coefficients)
        power_evaluated = box_spectra.evaluate3d_anisotropic(self.k_box(),self.mu_box())
        return self._gauss_realisation(power_evaluated,self.k_box())

    def isotropic_pre_computed_gauss_realisation(self,fname,n_interpolation_samples='default'):
        box_spectra = PreComputedPowerSpectrum(fname,n_interpolation_samples=n_interpolation_samples)
        orig_fourier_units_bool = self.convert_fourier_units_to_distance
        self.convert_fourier_units_to_distance = True
        power_evaluated = box_spectra.evaluate3d_isotropic(self.k_box())
        self.convert_fourier_units_to_distance = orig_fourier_units_bool
        return self._gauss_realisation(power_evaluated,self.k_box())

    def anisotropic_pre_computed_gauss_realisation(self, fname, mu_coefficients, n_interpolation_samples='default'):
        box_spectra = PreComputedPowerSpectrum(fname,n_interpolation_samples=n_interpolation_samples)
        box_spectra.set_anisotropic_functional_form(mu_coefficients)
        orig_fourier_units_bool = self.convert_fourier_units_to_distance
        self.convert_fourier_units_to_distance = True
        power_evaluated = box_spectra.evaluate3d_anisotropic(self.k_box(),self.mu_box())
        self.convert_fourier_units_to_distance = orig_fourier_units_bool
        return self._gauss_realisation(power_evaluated,self.k_box())

    def isotropic_CAMB_gauss_realisation(self):
        return 0

    def anisotropic_CAMB_gauss_realisation(self):
        return 0

    def _choose_location_voigt_profiles_in_sky(self):
        self._voigt_profile_skewers_index_arr = npr.choice(self.nskewers, self._num_voigt, replace=True) #More than one
        self._voigt_profile_skewers_bool_arr[self._voigt_profile_skewers_index_arr] = True
        self.num_clean_skewers = self.nskewers - np.sum(self._voigt_profile_skewers_bool_arr)

    def _evaluate_voigt_profiles(self,z_values,z0_values,sigma,gamma,amp,wrap_around):
        voigt_profile_box = np.zeros((self.nskewers, self._n_samp['z']))
        voigt_profiles_unwrapped = voigt_amplified(z_values[np.newaxis, :], sigma, gamma, amp, z0_values[:, np.newaxis])
        voigt_profiles_wrapped = np.sum(voigt_profiles_unwrapped.reshape(voigt_profiles_unwrapped.shape[0], 1 + (2 * wrap_around),-1), axis=-2)
        j = 0
        for i in self._voigt_profile_skewers_index_arr:
            if is_astropy_quantity(voigt_profiles_wrapped[j]):
                additional_profile = voigt_profiles_wrapped[j].value
            else:
                additional_profile = voigt_profiles_wrapped[j]
            voigt_profile_box[i] += additional_profile
            j+=1
        return voigt_profile_box, voigt_profiles_unwrapped

    def _form_voigt_profile_box(self, sigma, gamma, amp, wrap_around):
        z_values = np.arange(start = (-1*wrap_around)*self._n_samp['z'], stop = (1+wrap_around)*self._n_samp['z']) * self.voxel_velocities['z']
        print(z_values)
        z0_values = npr.choice(self._n_samp['z'], self._num_voigt, replace=True) * self.voxel_velocities['z']  # km / s
        return self._evaluate_voigt_profiles(z_values,z0_values,sigma,gamma,amp,wrap_around)

    def add_voigt_profiles(self,gauss_box,num_voigt,sigma,gamma,amp,wrap_around=0): #Use velocity units
        self._num_voigt = num_voigt
        self._choose_location_voigt_profiles_in_sky()
        voigt_profile_box,voigt_unwrapped = self._form_voigt_profile_box(sigma, gamma, amp, wrap_around)
        return gauss_box - (voigt_profile_box.reshape(gauss_box.shape) * (1. + 0.j)), voigt_unwrapped


class SimulationBox(Box):
    """Sub-class to generate a box of Lyman-alpha spectra drawn from HDF5 simulations"""
    def __init__(self, snap_num, snap_dir, grid_samps, spectrum_pixel_width, spectrograph_FWHM='default', reload_snapshot=True, spectra_savefile_root='gridded_spectra', spectra_savedir=None):
        self._n_samp = {}
        self._n_samp['x'] = grid_samps
        self._n_samp['y'] = grid_samps
        nskewers = self._n_samp['x'] * self._n_samp['y']

        self.voxel_lens = {}
        self.voxel_velocities = {}

        self._snap_num = snap_num
        self._snap_dir = snap_dir
        self._grid_samps = grid_samps
        self._spectrum_pixel_width = spectrum_pixel_width
        self._reload_snapshot = reload_snapshot
        self._spectra_savefile_root = spectra_savefile_root
        self.spectra_savedir = spectra_savedir
        self.spectra_savefile = '%s_%i_%i.hdf5'%(self._spectra_savefile_root,self._grid_samps,self._spectrum_pixel_width.value)

        self.element = 'H'
        self.ion = 1
        self.line_wavelength = 1215 * u.angstrom

        if spectrograph_FWHM == 'default':
            self.spectra_instance = gs.GriddedSpectra(self._snap_num, self._snap_dir, nspec=self._grid_samps, res=self._spectrum_pixel_width.value, savefile=self.spectra_savefile, savedir=self.spectra_savedir, reload_file=self._reload_snapshot)
        else:
            self.spectra_instance = gs.GriddedSpectra(self._snap_num, self._snap_dir, nspec=self._grid_samps,
                                                      res=self._spectrum_pixel_width.value,
                                                      savefile=self.spectra_savefile, savedir=self.spectra_savedir,
                                                      reload_file=self._reload_snapshot, spec_res=spectrograph_FWHM.to(u.km/u.s).value)

        self._n_samp['z'] = int(self.spectra_instance.vmax / self.spectra_instance.dvbin)
        H0 = (self.spectra_instance.hubble * 100. * u.km) / (u.s * u.Mpc)
        super(SimulationBox, self).__init__(self.spectra_instance.red, H0, self.spectra_instance.OmegaM, nskewers)
        self.voxel_velocities['x'] = (self.spectra_instance.vmax / self._n_samp['x']) * (u.km / u.s)
        self.voxel_velocities['y'] = (self.spectra_instance.vmax / self._n_samp['y']) * (u.km / u.s)
        self.voxel_velocities['z'] = self.spectra_instance.dvbin * (u.km / u.s)
        print("Size of voxels in velocity units =", self.voxel_velocities)
        for i in ['x','y','z']:
            #Comoving units
            self.voxel_lens[i] = (self.spectra_instance.box / (self.spectra_instance.hubble * self._n_samp[i] * 1000.)) * u.Mpc

        self._col_dens_threshold = 2.e+20 / (u.cm * u.cm) #Default values
        self._dodge_dist = 10. * u.kpc

    def _generate_general_spectra_instance(self, cofm):
        axis = np.ones(cofm.shape[0])
        return sa.Spectra(self._snap_num, self._snap_dir, cofm, axis, res=self._spectrum_pixel_width.value, reload_file=True)
    
    def save_file(self):
        self.get_optical_depth(save_file=True)

    def get_optical_depth(self,save_file=False):
        tau = self.spectra_instance.get_tau(self.element, self.ion, int(self.line_wavelength.value))
        if save_file:
            self.spectra_instance.save_file()  # Save spectra to file
        return tau

    def get_column_density(self, element = None, ion = None, save_file=False):
        if element is None:
            element = self.element
        if ion is None:
            ion = self.ion
        col_density = self.spectra_instance.get_col_density(element, ion) / (u.cm * u.cm)
        if save_file:
            self.spectra_instance.save_file()
        return col_density

    def get_optical_depth_real(self, element=None, ion=None): #Should really calculate pre-factor
        column_density = self.get_column_density(element=element, ion=ion)
        return column_density.value * np.mean(self.get_optical_depth()) / np.mean(column_density.value)

    def _get_scale(self, tau, mean_flux_desired): #Courtesy of Simeon Bird
        """Get the factor by which we need to multiply the optical depth to get a desired mean flux.
        ie, we want F_obs = bar{F} = < e^-tau >
        Solve this iteratively, using Newton-Raphson:
        S' = S + (<F> - F_obs) / <tau e^-tau>
        This is really Lyman-alpha forest specific."""
        #This is amazingly slow compared to C!
        minim = lambda scale: mean_flux_desired - np.mean(np.exp(-scale * tau))
        scale = spo.brentq(minim, 0, 20., rtol=1e-6)
        print("Scaled by:", scale)
        return scale

    def get_mean_flux(self, optical_depth=None, tau_scaling_factor=1.):
        if optical_depth is None:
            optical_depth = self.get_optical_depth()
        return np.mean(np.exp(-1. * optical_depth * tau_scaling_factor))

    def _get_delta_flux(self, tau, mean_flux_desired, mean_flux_specified, tau_scaling_specified):
        if mean_flux_desired is None:
            tau_scaling = 1.
        else:
            tau_scaling = self._get_scale(tau, mean_flux_desired)

        if tau_scaling_specified is not None:
            tau_scaling = tau_scaling_specified

        if mean_flux_specified is None:
            mean_flux = self.get_mean_flux(optical_depth=tau, tau_scaling_factor=tau_scaling)
        else:
            mean_flux = mean_flux_specified

        print('Mean flux = %f' %mean_flux)
        return np.exp(-1. * tau * tau_scaling) / mean_flux - 1.

    def _get_delta_density(self, density):
        mean_density = np.mean(density)
        return density / mean_density - 1.

    def skewers_realisation(self, mean_flux_desired = None, mean_flux_specified = None, tau_scaling_specified = None, redshift_space=True):
        if redshift_space:
            tau = self.get_optical_depth()
        else:
            tau = self.get_optical_depth_real()
        delta_flux = self._get_delta_flux(tau, mean_flux_desired, mean_flux_specified, tau_scaling_specified)
        return delta_flux.reshape((self._grid_samps, self._grid_samps, -1))

    def skewers_realisation_hydrogen_overdensity(self, ion = None):
        column_density = self.get_column_density(ion = ion)
        delta_density = self._get_delta_density(column_density)
        return delta_density.reshape((self._grid_samps, self._grid_samps, -1))

    def skewers_realisation_without_DLAs(self,mean_flux_desired=None,mean_flux_specified=None,tau_scaling_specified=None,skewers_with_DLAs_bool_arr=None):
        tau = self.get_optical_depth()
        if skewers_with_DLAs_bool_arr == None:
            skewers_with_DLAs_bool_arr = self._get_skewers_with_DLAs_bool_arr(self.get_column_density())
        tau_without_DLAs = tau[~ skewers_with_DLAs_bool_arr]
        return self._get_delta_flux(tau_without_DLAs, mean_flux_desired, mean_flux_specified, tau_scaling_specified)

    def skewers_realisation_with_DLAs_only(self,mean_flux_desired=None,mean_flux_specified=None,tau_scaling_specified=None,skewers_with_DLAs_bool_arr=None):
        tau = self.get_optical_depth()
        if skewers_with_DLAs_bool_arr == None:
            skewers_with_DLAs_bool_arr = self._get_skewers_with_DLAs_bool_arr(self.get_column_density())
        tau_with_DLAs_only = tau[skewers_with_DLAs_bool_arr]
        return self._get_delta_flux(tau_with_DLAs_only, mean_flux_desired, mean_flux_specified, tau_scaling_specified)

    def skewers_realisation_subset(self, boolean_mask, mean_flux_desired=None, mean_flux_specified=None, tau_scaling_specified=None):
        tau = self.get_optical_depth()[boolean_mask.flatten()]
        return self._get_delta_flux(tau, mean_flux_desired, mean_flux_specified, tau_scaling_specified)

    def max_local_sum_of_column_density_in_each_skewer(self):
        col_dens_local_sum = self._get_local_sum_of_column_density(self.get_column_density())
        return np.max(col_dens_local_sum, axis=-1).reshape((self._grid_samps, self._grid_samps))

    def _get_skewers_with_DLAs_bool_arr_simple_threshold(self, col_dens):
        return np.max(col_dens, axis=-1) > self._col_dens_threshold

    def _get_local_sum_of_column_density(self, col_dens):
        size_of_bin_in_velocity = 100. * (u.km / u.s)
        size_of_bin_in_samples = round(size_of_bin_in_velocity.value / self.voxel_velocities['z'].value)
        print("\nSize of bin in samples = %i" % size_of_bin_in_samples)
        return (calculate_local_average_of_array(col_dens.value,size_of_bin_in_samples) * size_of_bin_in_samples) / (u.cm * u.cm)

    def _get_skewers_with_DLAs_bool_arr_local_sum_threshold(self, col_dens):
        col_dens_local_sum = self._get_local_sum_of_column_density(col_dens)
        return self._get_skewers_with_DLAs_bool_arr_simple_threshold(col_dens_local_sum)

    def _get_skewers_with_DLAs_bool_arr(self,col_dens):
        assert is_astropy_quantity(col_dens)
        return self._get_skewers_with_DLAs_bool_arr_local_sum_threshold(col_dens)

    def _get_optical_depth_for_new_skewers(self, skewers_with_DLAs_bool_arr):
        new_skewers_cofm = self.spectra_instance.cofm[skewers_with_DLAs_bool_arr] #Slicing out new skewers
        new_tau = self._generate_general_spectra_instance(new_skewers_cofm).get_tau(self.element, self.ion, int(self.line_wavelength.value))
        self.spectra_instance.tau[(self.element, self.ion, int(self.line_wavelength.value))][skewers_with_DLAs_bool_arr] = new_tau

    def _get_column_density_for_new_skewers(self, skewers_with_DLAs_bool_arr):
        new_skewers_cofm = self.spectra_instance.cofm[skewers_with_DLAs_bool_arr] #Slicing out skewers with DLAs
        new_col_dens = self._generate_general_spectra_instance(new_skewers_cofm).get_col_density(self.element,self.ion) / (u.cm * u.cm)
        self.spectra_instance.colden[(self.element, self.ion)][skewers_with_DLAs_bool_arr] = new_col_dens.value

    def _form_skewers_realisation_dodging_DLAs_single_iteration(self, skewers_with_DLAs_bool_arr):
        print("Number of skewers with DLAs = %i" % np.sum(skewers_with_DLAs_bool_arr))
        self.spectra_instance.cofm[skewers_with_DLAs_bool_arr, 1] += self._dodge_dist.value
        self._get_column_density_for_new_skewers(skewers_with_DLAs_bool_arr)
        skewers_with_DLAs_bool_arr = self._get_skewers_with_DLAs_bool_arr(self.spectra_instance.colden[(self.element,self.ion)] / (u.cm * u.cm))
        return skewers_with_DLAs_bool_arr

    def _get_column_density_for_new_skewers_loop(self, skewers_with_DLAs_bool_arr):
        while np.sum(skewers_with_DLAs_bool_arr) > 0: #Continue dodging while there remain DLAs
            skewers_with_DLAs_bool_arr = self._form_skewers_realisation_dodging_DLAs_single_iteration(skewers_with_DLAs_bool_arr)

    def _save_new_skewers_realisation_dodging_DLAs(self, savefile_root):
        if self.spectra_savedir == None:
            savefile_tuple = (self._snap_dir + '/snapdir_' + str(self._snap_num).rjust(3,'0'),savefile_root,self._grid_samps,self._spectrum_pixel_width.value)
        else:
            savefile_tuple = (self.spectra_savedir,savefile_root,self._grid_samps,self._spectrum_pixel_width.value)
        self.spectra_instance.savefile = '%s/%s_%i_%i.hdf5' % savefile_tuple
        self.spectra_instance.save_file()

    def form_skewers_realisation_dodging_DLAs(self, col_dens_threshold = 2.e+20 / (u.cm * u.cm), dodge_dist=10.*u.kpc, savefile_root='gridded_spectra_DLAs_dodged'):
        self._col_dens_threshold = col_dens_threshold #Update if changed
        self._dodge_dist = dodge_dist
        skewers_with_DLAs_bool_arr = self._get_skewers_with_DLAs_bool_arr(self.get_column_density())
        self._get_column_density_for_new_skewers_loop(skewers_with_DLAs_bool_arr)
        self._get_optical_depth_for_new_skewers(skewers_with_DLAs_bool_arr)
        self._save_new_skewers_realisation_dodging_DLAs(savefile_root)

import numpy as np
from scipy.interpolate import interp1d



file_dish_z_2pt86=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_2_86.dat',dtype=float,unpack=True)
file_interferom_z_2pt86=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_2_86.dat',dtype=float,unpack=True)
file_dish_z_2pt76=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_2_76.dat',dtype=float,unpack=True)
file_interferom_z_2pt76=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_2_76.dat',dtype=float,unpack=True)
file_dish_z_2pt66=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_2_66.dat',dtype=float,unpack=True)
file_interferom_z_2pt66=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_2_66.dat',dtype=float,unpack=True)
file_dish_z_2pt56=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_2_56.dat',dtype=float,unpack=True)
file_interferom_z_2pt56=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_2_56.dat',dtype=float,unpack=True)
file_dish_z_2pt46=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_2_46.dat',dtype=float,unpack=True)
file_interferom_z_2pt46=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_2_46.dat',dtype=float,unpack=True)
file_dish_z_2pt36=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_2_36.dat',dtype=float,unpack=True)
file_interferom_z_2pt36=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_2_36.dat',dtype=float,unpack=True)
file_dish_z_2pt26=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_2_26.dat',dtype=float,unpack=True)
file_interferom_z_2pt26=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_2_26.dat',dtype=float,unpack=True)
file_dish_z_2pt16=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_2_16.dat',dtype=float,unpack=True)
file_interferom_z_2pt16=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_2_16.dat',dtype=float,unpack=True)
file_dish_z_2pt06=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_2_06.dat',dtype=float,unpack=True)
file_interferom_z_2pt06=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_2_06.dat',dtype=float,unpack=True)
file_dish_z_1pt96=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_1_96.dat',dtype=float,unpack=True)
file_interferom_z_1pt96=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_1_96.dat',dtype=float,unpack=True)
file_dish_z_1pt86=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_1_86.dat',dtype=float,unpack=True)
file_interferom_z_1pt86=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_1_86.dat',dtype=float,unpack=True)
file_dish_z_1pt76=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_1_76.dat',dtype=float,unpack=True)
file_interferom_z_1pt76=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_1_76.dat',dtype=float,unpack=True)
file_dish_z_1pt66=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_1_66.dat',dtype=float,unpack=True)
file_interferom_z_1pt66=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_1_66.dat',dtype=float,unpack=True)
file_dish_z_1pt56=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_1_56.dat',dtype=float,unpack=True)
file_interferom_z_1pt56=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_1_56.dat',dtype=float,unpack=True)
file_dish_z_1pt46=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_1_46.dat',dtype=float,unpack=True)
file_interferom_z_1pt46=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_1_46.dat',dtype=float,unpack=True)
file_dish_z_1pt36=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_1_36.dat',dtype=float,unpack=True)
file_interferom_z_1pt36=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_1_36.dat',dtype=float,unpack=True)
file_dish_z_1pt26=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_1_26.dat',dtype=float,unpack=True)
file_interferom_z_1pt26=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_1_26.dat',dtype=float,unpack=True)
file_dish_z_1pt16=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_1_16.dat', dtype=float,unpack=True)
file_interferom_z_1pt16=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_1_16.dat', dtype=float,unpack=True)
file_dish_z_1pt06=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_1_06.dat', dtype=float,unpack=True)
file_interferom_z_1pt06=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_1_06.dat', dtype=float,unpack=True)
file_dish_z_pt96=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_0_96.dat', dtype=float, unpack=True)
file_interferom_z_pt96=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_0_96.dat', dtype=float,unpack=True)
file_dish_z_pt86=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_0_86.dat', dtype=float, unpack=True)
file_interferom_z_pt86=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_0_86.dat', dtype=float,unpack=True)
file_dish_z_pt76=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_0_76.dat', dtype=float, unpack=True)
file_interferom_z_pt76=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_0_76.dat', dtype=float,unpack=True)
file_dish_z_pt66=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_0_66.dat', dtype=float, unpack=True)
file_interferom_z_pt66=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_0_66.dat', dtype=float,unpack=True)
file_dish_z_pt56=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_0_56.dat', dtype=float, unpack=True)
file_interferom_z_pt56=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_0_56.dat', dtype=float,unpack=True)
file_dish_z_pt46=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_0_46.dat', dtype=float, unpack=True)
file_interferom_z_pt46=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_0_46.dat', dtype=float,unpack=True)
file_dish_z_pt36=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_0_36.dat', dtype=float, unpack=True)
file_interferom_z_pt36=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_0_36.dat', dtype=float,unpack=True)

def SKA_dish_interferom(ell, file_dish, file_interferom):
    dish_file_loaded=file_dish
    ell_dish=dish_file_loaded[:,0]
    SKA_dish=dish_file_loaded[:,1]
    SKA_noise_dish_interp=interp1d(ell_dish, SKA_dish)

    interferom_file_loaded=file_interferom
    ell_interferom=interferom_file_loaded[:,0]
    SKA_interferom=interferom_file_loaded[:,1]
    SKA_noise_interferom_interp=interp1d(ell_interferom, SKA_interferom)

    SKA_noise_total_interp=interp1d(ell_interferom,1./(1./(1.e6*SKA_noise_dish_interp(ell_interferom))+1./(1.e6*SKA_noise_interferom_interp(ell_interferom))))

    return 1.e6*SKA_noise_dish_interp(ell), 1.e6*SKA_noise_interferom_interp(ell), SKA_noise_total_interp(ell)

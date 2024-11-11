from pykingas.MieKinGas import MieKinGas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from thermopack.saftvrmie import saftvrmie
from thermopack.saftvrqmie import saftvrqmie
from scipy.constants import Boltzmann
from scipy.constants import R
from scipy.constants import Avogadro

Temp_change = [1.05, 1.1, 1.15]
T_values = []

# Calculate critical temperature and select temperatures 5%, 10% and 15% above the critical temperature for the mixture
def Tcrit(Molecules, Molar_composition):
    global T_values, eos, Tc, pc, Vc, Temp_change
    T_values = []
    eos = saftvrmie(Molecules)
    Tc, Vc, pc = eos.critical(Molar_composition)
    for i in Temp_change: 
        T_values.append(round(Tc*i, 2))
    
# Plot phase envelope
# REMEMBER TO CHANGE EoS for different mixtures
def phase_envelope(Molecules, Molar_composition): #Molecules = "", Molar_composition = [x1, x2, ...] x_n sum to 1
    global T_values, eos, kin
    T_values = []
    eos = saftvrmie(Molecules)
    kin = MieKinGas(Molecules, use_eos=eos)
    
    T, p, V = eos.get_envelope_twophase(1e5, Molar_composition, calc_v=True)
    plt.plot(1/V, T, "b", label = f"{Molecules}: {Molar_composition}")
    try:
        Tcrit(Molecules, Molar_composition)
        plt.plot(1/Vc, Tc, "r*", label="Critical point")
    except:
        print("Critical calculation failed")
        
    plt.ylabel(r"$T$ [K]")
    plt.xlabel(r"$\rho$ [mol m$^{-3}$]")
    plt.title(f"Phase diagram for {Molecules}: {Molar_composition}")
    plt.ylim(T[0],Tc*1.25)
    plt.legend()
    plt.show()
    return

def dim_less_visc(T, visc_data, density, average_particle_mass):
    return ((np.array(density)*Avogadro)**(-2/3)*(Boltzmann*average_particle_mass*T)**-(1/2)*np.array(visc_data)).tolist()

# Pressure range with 20 total values
p_start = 1e5
p_end = 50e5
p_increment = (p_end-p_start)/19
pressure = np.arange(p_start, p_end+1, p_increment)
m_Ar = 39.948 * 1.66054e-27 # kg
m_CO2 = 12.011 * 1.66054e-27 # kg

# Mixtures to analyze, these can be changed
Mixtures = {"AR": [1], "CO2": [1]}
m_mix = [m_Ar, m_CO2]

# To more easily extract data from dictionary
Mixture = []
Calculated_data = {}
Experimental_data = {}

# Iterate each mixture and create list with thermodynamic data
for Molecule in Mixtures.keys():
    
    # Calculate phase diagram and temperature values to make sure mixture is in vapour phase
    phase_envelope(Molecule, Mixtures[Molecule])
    
    # In case same components but different mixing ratio is beeing analyzed
    Mix = Molecule + str(Mixtures[Molecule])
    Mixture.append(Mix)

    Calculated_data[Mix] = {
        "cond_data": {T: [] for T in T_values}, # [W/m*K] conductivity
        "visc_data": {T: [] for T in T_values}, # [Pa*s] viscosity
        "red_entropy": {T: [] for T in T_values}, # Reduced entropy
        "id_entropy": {T: [] for T in T_values}, # Ideal entropy
        "res_entropy": {T: [] for T in T_values}, # Residual entropy
        "density": {T: [] for T in T_values}, # [mol/m^3] Density
        "spes_volume": {T: [] for T in T_values}, # [m^3/mol] Specific volume
        "cp_dict": {T: [] for T in T_values} # [J/mol*K] Heat capacity
        }
    
    Experimental_data[Mix] = {
        "cond_data": {T: [] for T in T_values},
        "visc_data": {T: [] for T in T_values},
        "red_entropy": {T: [] for T in T_values},
        "res_entropy": {T: [] for T in T_values},
        "density": {T: [] for T in T_values},
        "cp_dict": {T: [] for T in T_values}
        }
    
    # Calculate thermodynamic data and put it into the dictionary
    # To exctract the residual entropies for different pressures for Mixture[0] with temperature T: Calculated_data[Mixture[0]]["res_entropy"][T]
    for T in T_values:
        #print("NEW TEMPERATURE")
        test, = eos.specific_volume(T, 1000, Mixtures[Molecule], eos.VAPPH)
        testcond = kin.thermal_conductivity(T, test, [0.5, 0.5], N=2)
        #print(test)
        #print(testcond)
        i = 0
        for p in pressure:
            flsh = eos.two_phase_tpflash(T, p, [1])
            s_volume = eos.specific_volume(T, p, Mixtures[Molecule], eos.VAPPH) # Specific volume
            Calculated_data[Mix]["spes_volume"][T].append(s_volume[0])
            Calculated_data[Mix]["density"][T].append(1/s_volume[0])
            
            entro = eos.entropy_tv(T, s_volume[0], Mixtures[Molecule], property_flag="R") # Flag R to calculate residual entropy
            Calculated_data[Mix]["res_entropy"][T].append(entro[0])
            Calculated_data[Mix]["red_entropy"][T].append(-entro[0]/R)
            
            id_entro = eos.entropy_tv(T, s_volume[0], Mixtures[Molecule], property_flag="I") # Flag I to calculate ideal entropy
            Calculated_data[Mix]["id_entropy"][T].append(id_entro[0])
            
            _, Cp_vap = eos.enthalpy(T, p, Mixtures[Molecule], eos.VAPPH, dhdt=True) # Vapour phase heat capacity at constant pressure
            Calculated_data[Mix]["cp_dict"][T].append(Cp_vap)
            
            if len(Mixtures[Molecule]) == 1: # If the mixture is pure set the composition equal [0.5, 0.5] to allow calculations with Miekingas
                visc = kin.viscosity_tp(T, p, [0.5, 0.5], N=2) # Viscosity
                Calculated_data[Mix]["visc_data"][T].append(visc)
                
                cond = kin.thermal_conductivity(T, Calculated_data[Mix]["spes_volume"][T][i], [0.5, 0.5], N=2) # Conductivity
                Calculated_data[Mix]["cond_data"][T].append(cond)
            
            else:
                visc = kin.viscosity_tp(T, p, Mixtures[Molecule], N=2)
                Calculated_data[Mix]["visc_data"][T].append(visc)
                
                cond = kin.thermal_conductivity(T, Calculated_data[Mix]["spes_volume"][T][i], Mixtures[Molecule], N=2)
                Calculated_data[Mix]["cond_data"][T].append(cond)
            i += 1
            
        # Insert experimental data from NIST, requires file with specific name format
        dataframe = pd.read_csv("data/" + Molecule + "_" + str(round(T))+".txt", delimiter="\t")
        Experimental_data[Mix]["cond_data"][T].extend(dataframe["Therm. Cond. (W/m*K)"].tolist())
        Experimental_data[Mix]["visc_data"][T].extend(dataframe["Viscosity (Pa*s)"].tolist())
        Experimental_data[Mix]["res_entropy"][T].extend((np.array(dataframe["Entropy (J/mol*K)"])-np.array(Calculated_data[Mix]["id_entropy"][T])).tolist())
        Experimental_data[Mix]["red_entropy"][T].extend((np.array(Experimental_data[Mix]["res_entropy"][T])/-R).tolist())
        Experimental_data[Mix]["density"][T].extend(dataframe["Density (mol/m3)"].tolist())
        Experimental_data[Mix]["cp_dict"][T].extend(dataframe["Cp (J/mol*K)"].tolist())

i = 0
for Molecule in Mixtures.keys():
    Tcrit(Molecule, Mixtures[Molecule])
    
    for T in T_values:
        plt.plot(Calculated_data[Mixture[i]]["red_entropy"][T], np.log(dim_less_visc(T, Calculated_data[Mixture[i]]["visc_data"][T], Calculated_data[Mixture[i]]["density"][T], m_mix[i])), linestyle= "", marker = "o")
    plt.xlabel(r"$S_{res}^*$")
    plt.ylabel(r"ln($\eta^*)$")
    plt.title(f"Calculated values for {Molecule}: {Mixtures[Molecule]}")
    plt.show()
    
    for T in T_values:
        plt.plot(Experimental_data[Mixture[i]]["red_entropy"][T], np.log(dim_less_visc(T, Experimental_data[Mixture[i]]["visc_data"][T], Experimental_data[Mixture[i]]["density"][T], m_mix[i])), linestyle= "", marker = "o")
    plt.xlabel(r"$S_{res}^*$")
    plt.ylabel(r"ln($\eta^*)$")
    plt.title(f"Experimental values {Molecule}: {Mixtures[Molecule]}")
    plt.show()
    i += 1

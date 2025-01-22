#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:47:43 2024

@author: barnabywinser
"""

scenario = "tech cost parameters"
folder = "/Users/barnabywinser/Documents/local data/"

# Application parameters
Cap_p_nom = 10  # Power in MW
t = 8  # Discharge duration hours
Cyc_pa = 365  # Cycles per annum
P_el = 50  # Electricity purchase price $/MWh
el_g = 0  # Electricity price escalator % p.a.

graph = 'b'
ya = 50
yb = 400
scenario = f"{t}hrs"

import pandas as pd
import math
import matplotlib.pyplot as plt

file_path = '/Users/barnabywinser/Documents/local data/Tech cost parameters.xlsx'

#define technology parameters as df
df = pd.read_excel(file_path, sheet_name=f"Fluid across MW ({t}hrs)", index_col=0, header=0)

def calculate_lcos(fluid_cost, technology_params):
    C_p_inv = df.loc['Power CAPEX', technology]
    C_e_inv = df.loc['Energy CAPEX', technology]
    C_e_vol = df.loc['Volume per kwh', technology]
    C_e_inv = C_e_inv + C_e_vol * fluid_cost
    C_p_om = df.loc['Power OPEX', technology]
    C_e_om = df.loc['Energy OPEX', technology]
    C_p_eol = df.loc['Power EoL cost', technology]
    C_e_eol = df.loc['Energy EoL cost', technology]
    C_p_rep = df.loc['Replacement Power', technology]
    C_e_rep = df.loc['Replacement Energy', technology]
    r = (df.loc['WACC', technology]) / 100
    rt = (df.loc['Round-trip efficiency', technology]) / 100
    DoD = (df.loc['DoD', technology]) / 100
    n_self = (df.loc['Self-discharge', technology]) / 100
    Life_cyc = df.loc['Lifetime 100% DoD', technology]
    Cyc_rep = df.loc['Replacement interval', technology]
    Deg_t = (df.loc['Temporal degradation', technology]) / 100
    EoL = (df.loc['EoL threshold', technology]) / 100
    T_con = int(df.loc['Pre-dev + construction', technology])
    N_op1 = df.loc['Economic Life', technology]

    # Intermediate calculations
    Cap_e_nom = Cap_p_nom * t  # Energy capacity MWh
    Deg_c = 1 - EoL ** (1 / Life_cyc)  # Cycle degradation
    N_op = min(math.log(EoL) / (math.log(1 - Deg_t) + Cyc_pa * math.log(1 - Deg_c)), N_op1)  # Operational lifetime
    T_rep = Cyc_rep / Cyc_pa  # Years per replacement
    R = N_op / T_rep if T_rep > 0 else 0  # Replacements in lifetime

    # Project lifetime
    N_project = N_op + T_con  # Project lifetime

    # Calculate Total CAPEX (unchanged)
    Total_CAPEX = sum(1000 * (C_p_inv * Cap_p_nom + C_e_inv * Cap_e_nom) / (T_con * (1 + r) ** (n - 1)) for n in range(1, T_con + 1))

    # Function to calculate annual charged electricity
    def Ein(n):
        return (Cap_e_nom * DoD * Cyc_pa / rt) * ((1 - Deg_c) ** ((n - T_con - 1) * Cyc_pa)) * ((1 - Deg_t) ** (n - T_con - 1))

    # **Updated Charging Cost Calculation**
    # Charging costs are incurred only after construction (from year T_con + 1)
    total_charge_cost = sum(P_el * Ein(n) / (1 + r) ** (n - 1) for n in range(T_con + 1, int(N_op) + T_con + 1))

    # Add fractional year for charging cost
    fraction_yr = N_op - int(N_op)
    if fraction_yr > 0:
        n = int(N_op) + T_con + 1
        fractional_energy_input = Ein(n) * fraction_yr
        total_charge_cost += fractional_energy_input * P_el / (1 + r) ** (n - 1)

    # Calculate present value of replacement costs (unchanged)
    Rep_disc = 0
    if C_p_rep + C_e_rep > 0:
        Rep_disc = sum((1000 * C_p_rep * Cap_p_nom + C_e_rep * Cap_e_nom) / (1 + r) ** (T_con + k * T_rep) for k in range(1, int(R) + 1))
        fractional_R = R - int(R)
        if fractional_R > 0:
            Rep_disc += fractional_R * (1000 * C_p_rep * Cap_p_nom + C_e_rep * Cap_e_nom) / (1 + r) ** (T_con + (int(R) + 1) * T_rep)

    # Function to calculate Eout(n)
    def Eout(n):
        return Ein(n) * rt * (1 - n_self)

    # **Updated Energy Discharged Calculation**
    # Energy is discharged only after construction (from year T_con + 1)
    total_eout = 0
    full_years = int(N_op)
    fractional_year = N_op - full_years

    # Handle full years
    total_eout += sum(Eout(n) / (1 + r) ** (n - 1) for n in range(T_con + 1, full_years + T_con + 1))

    # Handle fractional year
    if fractional_year > 0:
        n = full_years + T_con + 1
        discharged_energy_fractional = Eout(n) * fraction_yr
        total_eout += discharged_energy_fractional / (1 + r) ** (n - 1)

    # Calculate total O&M cost (unchanged)
    total_om_cost = sum((C_p_om * Cap_p_nom * 1000 + C_e_om * Ein(n)) / (1 + r) ** (n - 1) for n in range(T_con + 1, full_years + T_con + 1))

    # Handle fractional year for O&M cost
    if fractional_year > 0:
        n = full_years + T_con + 1
        E_in_fractional = Ein(n) * fractional_year
        total_om_cost += (C_p_om * Cap_p_nom * 1000 + C_e_om * E_in_fractional) / (1 + r) ** (n - 1)

    # Calculate end-of-life cost (unchanged)
    Total_EoL = (C_p_eol * 1000 * Cap_p_nom + C_e_eol * 1000 * Cap_e_nom * (1 - Deg_t) ** N_op * (1 - Deg_c) ** (Cyc_pa * N_op)) / (1 + r) ** (N_project + 1)

    # Calculate LCOS
    numerator = Total_CAPEX + Rep_disc + total_charge_cost + total_om_cost + Total_EoL
    LCOS = numerator / total_eout if total_eout > 0 else None

    return LCOS

technologies = df.columns


results = []
fluid_costs = [c for c in range(80, 551)]

for technology in technologies:
    technology_params = df[technology]
    
    for c in fluid_costs:
        function_value = calculate_lcos(c, technology_params)
        results.append({
            'Technology': technology,
            'Fluid cost': c,
            'LCOS': function_value
        })

results_df = pd.DataFrame(results)

excel_file = '/Users/barnabywinser/Documents/local data/LCOS fluid.xlsx' 
results_df.to_excel(excel_file)

import seaborn as sns
from matplotlib import rcParams

# Set the font to Barlow
rcParams['font.family'] = 'Barlow'
rcParams['axes.titleweight'] = 'semibold'
rcParams['font.size'] = 12

# Define the Seaborn color palette
palette = sns.color_palette("muted", len(technologies))

plt.figure(figsize=(10, 6))

#technologies = ["HD Hydro", "Li-ion", "Li-ion (2030 NREL forecast)", "Pumped Hydro", "Vanadium Flow", "Compressed Air", "Hydrogen"]
technologies = ["HD Hydro", "Li-ion (2030 NREL forecast)"]

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'palette' is already defined (e.g., sns.color_palette("deep", len(technologies)))

# Find the color index for Li-ion so it can be reused for the 2030 forecast
li_ion_color = None

if graph == 'a':
    
    technologies = ["HD Hydro", "Li-ion", "Pumped Hydro", "Vanadium Flow", "Compressed Air", "Hydrogen"]
    
    # Iterate over each technology and plot its function values using Seaborn palette
    for idx, technology in enumerate(technologies):
        subset = results_df[results_df['Technology'] == technology]
        
        if technology == "Li-ion":
            # Save the color used for Li-ion
            li_ion_color = palette[idx]
            plt.plot(subset['Fluid cost'], subset['LCOS'], label=technology, color=li_ion_color)
            
        elif technology == "Li-ion (2030 NREL forecast)":
            # Use the same color as Li-ion but with a dotted line
            plt.plot(subset['Fluid cost'], subset['LCOS'], label=technology, color=palette[idx])
            
            # Get the data for "HD Hydro L" and "HD Hydro U" to plot error bounds
            l = results_df[results_df['Technology'] == "Li-ion (2030 NREL forecast) L"]
            u = results_df[results_df['Technology'] == "Li-ion (2030 NREL forecast) U"]
            
            # Ensure both lower and upper bound subsets are non-empty
            if not l.empty and not u.empty:
                # Plot the error bounds using fill_between
                plt.fill_between(subset['Fluid cost'], l['LCOS'], u['LCOS'], 
                                 color=palette[idx], alpha=0.3)
            
        elif technology == "HD Hydro":
            # Plot the LCOS line for HD Hydro
            plt.plot(subset['Fluid cost'], subset['LCOS'], label=technology, color=palette[idx])
            
            # Get the data for "HD Hydro L" and "HD Hydro U" to plot error bounds
            hd_hydro_l = results_df[results_df['Technology'] == "HD Hydro L"]
            hd_hydro_u = results_df[results_df['Technology'] == "HD Hydro U"]
            
            # Ensure both lower and upper bound subsets are non-empty
            if not hd_hydro_l.empty and not hd_hydro_u.empty:
                # Plot the error bounds using fill_between
                plt.fill_between(subset['Fluid cost'], hd_hydro_l['LCOS'], hd_hydro_u['LCOS'], 
                                 color=palette[idx], alpha=0.2)
            
        else:
            # Default plot for other technologies
            plt.plot(subset['Fluid cost'], subset['LCOS'], label=technology, color=palette[idx])

else:
# Iterate over each technology and plot its function values using Seaborn palette
    for idx, technology in enumerate(technologies):
        subset = results_df[results_df['Technology'] == technology]
        
        if technology == "Li-ion":
            # Save the color used for Li-ion
            li_ion_color = palette[idx]
            plt.plot(subset['Fluid cost'], subset['LCOS'], label='Li-ion (current)', color=palette[idx])
            
        elif technology == "Li-ion (2030 NREL forecast)":
            # Use the same color as Li-ion but with a dotted line
            plt.plot(subset['Fluid cost'], subset['LCOS'], label=technology, color=palette[idx])
            
            # Get the data for "HD Hydro L" and "HD Hydro U" to plot error bounds
            l = results_df[results_df['Technology'] == "Li-ion (2030 NREL forecast) L"]
            u = results_df[results_df['Technology'] == "Li-ion (2030 NREL forecast) U"]
            
            # Ensure both lower and upper bound subsets are non-empty
            if not l.empty and not u.empty:
                # Plot the error bounds using fill_between
                plt.fill_between(subset['Fluid cost'], l['LCOS'], u['LCOS'], 
                                 color=palette[idx], alpha=0.3)
            
        elif technology == "HD Hydro":
            # Plot the LCOS line for HD Hydro
            plt.plot(subset['Fluid cost'], subset['LCOS'], label=technology, color=palette[idx], alpha = 1)
            
            # Get the data for "HD Hydro L" and "HD Hydro U" to plot error bounds
            hd_hydro_l = results_df[results_df['Technology'] == "HD Hydro L"]
            hd_hydro_u = results_df[results_df['Technology'] == "HD Hydro U"]
            
            # Ensure both lower and upper bound subsets are non-empty
            if not hd_hydro_l.empty and not hd_hydro_u.empty:
                # Plot the error bounds using fill_between
                plt.fill_between(subset['Fluid cost'], hd_hydro_l['LCOS'], hd_hydro_u['LCOS'], 
                                 color=palette[idx], alpha=0.3)
            
        else:
            # Default plot for other technologies
            plt.plot(subset['Fluid cost'], subset['LCOS'], label=technology, color=palette[idx])


# Add vertical dotted lines at the specified points
x_values = [275, 325, 380, 440, 490, 540]
for x in x_values:
    plt.axvline(x=x, color='grey', linestyle=':', linewidth=1, alpha = 0.7)
    plt.text(x, yb * 0.95, f'${x}', rotation=0, verticalalignment='top', horizontalalignment='right', fontsize=10, color='grey')        
        

# Add the rest of your plotting code (e.g., labels, legend, etc.)
plt.suptitle(f"Levelized Cost of Storage (LCOS) Sensitivity to Fluid Cost - {t}hr duration", fontweight='semibold')
plt.xlabel('Fluid Cost ($/m3)', fontweight='semibold')
plt.ylabel('LCOS ($/MWh)', fontweight='semibold')

# Add x and y axis lines
plt.axhline(0, color='black',linewidth=1.5)  # Horizontal line at y=0 (x-axis)

# Add a legend
legend = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)

# Customize the grid and axis spines
plt.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.tick_params(axis='both', which='both', length=0)
plt.xlim(80, 550)
plt.ylim(ya, yb)
plt.axvline(80, color='grey',linewidth=1.5, alpha = 0.5)  # Vertical line at x=0 (y-axis)
plt.axhline(ya, color='grey', linewidth = 1.5, alpha = 0.5)

# Use tight_layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 1.01])

# Show the plot
plt.show()

plt.savefig(folder + scenario + graph, dpi = 300)

HD_10 = results_df[(results_df['Technology'] == 'HD Hydro') & (results_df['Fluid cost'] == 400)]['LCOS'].values[0]
LI_c = results_df[(results_df['Technology'] == 'Li-ion (2030 NREL forecast)') & (results_df['Fluid cost'] == 400)]['LCOS'].values[0]
LI_l = results_df[(results_df['Technology'] == 'Li-ion (2030 NREL forecast) L') & (results_df['Fluid cost'] == 400)]['LCOS'].values[0]

print(HD_10)

print("HD"+ scenario + " vs. Lithium mid: ", HD_10/LI_c)
print("HD vs. Lithium low: ", HD_10/LI_l)
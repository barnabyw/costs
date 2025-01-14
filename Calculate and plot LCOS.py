#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:47:43 2024

@author: barnabywinser
"""

scenario = "tech cost parameters"
output_path = "/Users/barnabywinser/Library/CloudStorage/OneDrive-SharedLibraries-Rheenergise/Commercial - Documents/Opportunities/Sibanye Stillwater/Sibanye Stillwater SSW due diligence/LCOS analysis/"

file = "Tech cost parameters - Ninja defaults.xlsx"

# Application parameters
Cap_p_nom = 10  # Power in MW
# t = 4  # Discharge duration hours
Cyc_pa = 365  # Cycles per annum
P_el = 50  # Electricity purchase price $/MWh
el_g = 0  # Electricity price escalator % p.a.

graph = 'b'
ya = 50
yb = 400
scenario = 'lcos'

import seaborn as sns
import pandas as pd
import math
import matplotlib.pyplot as plt

input_folder = '/Users/barnabywinser/Library/CloudStorage/OneDrive-SharedLibraries-Rheenergise/Commercial - Documents/Cost Models/LCOS/Input parameters/'
input_path = input_folder + file

#define technology parameters as df
df = pd.read_excel(input_path, sheet_name=scenario, index_col=0, header=0)


bounds = ["Li-ion (2030 NREL forecast)", "HD Hydro", "HD Hydro (cost down)"]# "Li-ion older NREL forecast"]
lines = ["Li-ion (2030 NREL forecast)","HD Hydro", "HD Hydro (cost down)"]# "Li-ion older NREL forecast"]#,"Li-ion with replacement", "Li-ion older NREL forecast", "Li-ion (2030 NREL forecast)", "Pumped Hydro"]

# Define the Seaborn color palette
palette = sns.color_palette("muted", len(lines)+5)

# Define a dictionary to map technologies to their colors
color_dict = {
    "Li-ion (2030 NREL forecast)": palette[1],
    "Gas CCGT": "#FF5733",               # Orange-red color for Gas CCGT
    "Gas OCGT": '#FF5733',
    "HD Hydro": palette[0],                # Blue color for HD Hydro
    "HD Hydro (cost down)": palette[2],            # Green color for Pumped Hydro
    "Li-ion older NREL forecast": "#FF5733",
    "Li-ion older NREL forecast with replacement": "#FF5733",
    # Purple color for Li-ion (2030 NREL)
    "Li-ion with replacement": "#9B59B6",      # Same color as Li-ion (2030 NREL forecast)
    "Gas CCGT L": "#FF5733",              # Same color for lower bound
    "Gas CCGT U": "#FF5733"               # Same color for upper bound
}


def calculate_lcos(t, technology_params):
    C_p_inv = df.loc['Power CAPEX', technology]
    C_e_inv = df.loc['Energy CAPEX', technology]
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
t_values = [t / 10 for t in range(1, 121)]

for technology in technologies:
    technology_params = df[technology]
    
    for t in t_values:
        function_value = calculate_lcos(t, technology_params)
        results.append({
            'Technology': technology,
            't': t,
            'LCOS': function_value
        })

results_df = pd.DataFrame(results)

excel_file = '/Users/barnabywinser/Documents/local data/LCOS all techs.xlsx' 
csv_file = '/Users/barnabywinser/Documents/local data/LCOS all techs.csv'
results_df.to_excel(excel_file)
results_df.to_csv(output_path+"Results.csv", index=False)

from matplotlib import rcParams

# Set the font to Barlow
rcParams['font.family'] = 'Roboto'
rcParams['axes.titleweight'] = 'semibold'
rcParams['font.size'] = 12


plt.figure(figsize=(10, 6))

#technologies = ["HD Hydro", "Li-ion", "Li-ion (2030 NREL forecast)", "Pumped Hydro", "Vanadium Flow", "Compressed Air", "Hydrogen"]
technologies = ["HD Hydro", "Li-ion (2030 NREL forecast)"]

import matplotlib.pyplot as plt


# Function to plot bounds with custom color
def plot_technology_bounds(technology, df, color_dict, palette=None, idx=None):
    subset = df[df['Technology'] == technology]
    
    # Check for lower (L) and upper (U) bounds
    tech_l = df[df['Technology'] == f"{technology} L"]
    tech_u = df[df['Technology'] == f"{technology} U"]
    
    # Use custom color if available, otherwise use palette
    color = color_dict.get(technology, palette[idx] if palette else 'black')
    
    if not tech_l.empty and not tech_u.empty and len(tech_l) == len(tech_u):
        #plt.plot(tech_l['t'], tech_l['LCOS/E'], color=color, linestyle=':', linewidth=2.5, label=f"{technology} L")
        #plt.plot(tech_u['t'], tech_u['LCOS/E'], color=color, linestyle=':', linewidth=2.5, label=f"{technology} U")
        plt.fill_between(subset['t'], tech_l['LCOS'], tech_u['LCOS'], color=color, alpha=0.2)

# Function to plot lines with custom color and linestyle
def plot_technology_lines(technology, df, color_dict, palette=None, idx=None):
    subset = df[df['Technology'] == technology]
    
    # Use custom color if available, otherwise use palette
    color = color_dict.get(technology, palette[idx] if palette else 'black')
    
    # Set linestyle for "Li-ion with replacement" to dotted, otherwise use solid
    #linestyle = '--' if technology == "Li-ion with replacement" else '-' -- input linestyle=linestyle below to use
    
    # Plot the main line for the technology
    plt.plot(subset['t'], subset['LCOS'], label=technology, color=color, linewidth=2.5)

# Plotting starts here
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot technologies with bounds, using color dictionary
for idx, technology in enumerate(bounds):
    plot_technology_bounds(technology, results_df, color_dict, palette=palette, idx=idx)

# Plot other lines with solid style, using color dictionary
for idx, technology in enumerate(lines):
    plot_technology_lines(technology, results_df, color_dict, palette=palette, idx=idx)
        

# Add the rest of your plotting code (e.g., labels, legend, etc.)
plt.suptitle("Levelized Cost of Storage (LCOS) by System Duration and Technology (" + scenario + "/m3 Fluid Cost)", fontweight='semibold')
plt.xlabel('System Duration (hrs)', fontweight='semibold')
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
plt.xlim(0, 12)
plt.ylim(ya, yb)
plt.axvline(0, color='grey',linewidth=1.5, alpha = 0.5)  # Vertical line at x=0 (y-axis)
plt.axhline(ya, color='grey', linewidth = 1.5, alpha = 0.5)

# Use tight_layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 1.01])

plt.savefig(output_path + scenario + graph, dpi = 300)
# Show the plot
plt.show()

HD_10 = results_df[(results_df['Technology'] == 'HD Hydro') & (results_df['t'] == 10)]['LCOS'].values[0]
LI_c = results_df[(results_df['Technology'] == 'Li-ion (2030 NREL forecast)') & (results_df['t'] == 10)]['LCOS'].values[0]
LI_l = results_df[(results_df['Technology'] == 'Li-ion (2030 NREL forecast) L') & (results_df['t'] == 10)]['LCOS'].values[0]

print("HD"+ scenario + " vs. Lithium mid: ", HD_10/LI_c)
print("HD vs. Lithium low: ", HD_10/LI_l)
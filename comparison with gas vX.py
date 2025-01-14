from matplotlib.lines import Line2D

import pandas as pd
import math
import matplotlib.pyplot as plt
scenario = "ocgts"
folder = "/Users/barnabywinser/Documents/local data/"
from matplotlib.ticker import FuncFormatter
import numpy as np

# Application parameters
Cap_p_nom = 10  # Power in MW
P_el = 10 * 1.2439  # Electricity purchase price $/MWh
el_g = 0  # Electricity price escalator % p.a.



def calculate_cyc_pa(t):
    """
    Calculate cycles per annum (Cyc_pa) based on the given formula.
    Parameters:
        t (float): Value of B3, which represents the duration in hours.
    Returns:
        float: Calculated Cycles per annum (Cyc_pa).
    """
    if t == 0:  # Avoid division by zero
        raise ValueError("t (B3) must be greater than 0")
    cyc_pa = ((1110.903 / 100) * (math.log((t / 26.25) + 0.2) + 5.1569) - 36.4018) * 8760 / 100 / t
    return cyc_pa

# Example usage:
t = 96  # Example value for B3
Cyc_pa = calculate_cyc_pa(t)
print(f"Cyc_pa for t={t}: {Cyc_pa}")

graph = 'b'
ya = 50
yb = 300
lcos_scenario = 'lcos w rep'
lcoe_scenario = 'lcoe'



file_path = '/Users/barnabywinser/Documents/local data/Tech cost parameters.xlsx'

# Define technology parameters as df
df = pd.read_excel(file_path, sheet_name=lcos_scenario, index_col=0, header=0)
df2 = pd.read_excel(file_path, sheet_name='lcoe (UK)', index_col=0)

# Define function for LCOS
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

    Cyc_pa = calculate_cyc_pa(t)

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

    # Charging costs incurred after construction (from year T_con + 1)
    total_charge_cost = sum(P_el * Ein(n) / (1 + r) ** (n - 1) for n in range(T_con + 1, int(N_op) + T_con + 1))

    # Add fractional year for charging cost
    fraction_yr = N_op - int(N_op)
    if fraction_yr > 0:
        n = int(N_op) + T_con + 1
        fractional_energy_input = Ein(n) * fraction_yr
        total_charge_cost += fractional_energy_input * P_el / (1 + r) ** (n - 1)

    # Calculate replacement costs (unchanged)
    Rep_disc = 0
    if C_p_rep + C_e_rep > 0:
        Rep_disc = sum((1000 * C_p_rep * Cap_p_nom + 1000 * C_e_rep * Cap_e_nom) / (1 + r) ** (T_con + k * T_rep) for k in range(1, int(R) + 1))
        fractional_R = R - int(R)
        if fractional_R > 0:
            Rep_disc += fractional_R * (1000 * C_p_rep * Cap_p_nom + 1000 * C_e_rep * Cap_e_nom) / (1 + r) ** (T_con + (int(R) + 1) * T_rep)

    # Function to calculate discharged energy (Eout)
    def Eout(n):
        return Ein(n) * rt * (1 - n_self)

    total_eout = sum(Eout(n) / (1 + r) ** (n - 1) for n in range(T_con + 1, int(N_op) + T_con + 1))

    # Total O&M costs
    total_om_cost = sum((C_p_om * Cap_p_nom * 1000 + C_e_om * Ein(n)) / (1 + r) ** (n - 1) for n in range(T_con + 1, int(N_op) + T_con + 1))

    # End-of-life costs
    Total_EoL = (C_p_eol * 1000 * Cap_p_nom + C_e_eol * 1000 * Cap_e_nom) / (1 + r) ** (N_project + 1)

    # Calculate LCOS
    numerator = Total_CAPEX + Rep_disc + total_charge_cost + total_om_cost + Total_EoL
    LCOS = float(numerator / total_eout) if total_eout > 0 else 10000
    LCOS = LCOS / 1.2439

    return LCOS

# Function for LCOE (unchanged)
def calculate_lcoe(t, technology_params):
    capex = df2.loc['CAPEX', technology]
    om_fix = df2.loc['Fixed O&M', technology]
    om_var = df2.loc['Variable O&M', technology]
    T_con = int(df2.loc['Construction Time', technology])
    N_op = int(df2.loc['Operational Life', technology])
    fuel_price = df2.loc['Fuel Price', technology]
    efficiency = df2.loc['Efficiency', technology]
    r = (df2.loc['WACC', technology]) / 100
    power = int(df2.loc['Power Capacity', technology])
    fuel_ci = df2.loc['Carbon Emission', technology]
    carbon_price = df2.loc['Carbon Price', technology]

    cf = ((1110.903 / 100) * (math.log((t / 26.25) + 0.2) + 5.1569) - 36.4018) / 100
    #Intermediate calculations
    energy = 8.76 * cf * power * 1000
    fuel = fuel_price/efficiency * energy
    carbon = fuel_ci/efficiency

    Total_CAPEX = sum(capex * power * 1000 / (T_con * (1 + r) ** (n - 1)) for n in range(1, T_con + 1))

    total_om_fix = 0
    for n in range(T_con, T_con + N_op + 1):
        total_om_fix += (power * 1000 * om_fix) / (1 + r) ** (n - 1)

    total_om_var = 0
    for n in range(T_con, T_con + N_op + 1):
        total_om_var += energy * om_var / (1 + r) ** (n - 1)

    co2_cost = 0
    for n in range(T_con, T_con + N_op + 1):
        co2_cost += energy * carbon * carbon_price / (1 + r) ** (n - 1)

    total_fuel = 0
    for n in range(T_con, T_con + N_op + 1):
        total_fuel += fuel / (1 + r) ** (n - 1)

    disc_energy = 0
    for n in range(T_con, T_con + N_op + 1):
        disc_energy += (energy / (1 + r) ** (n - 1))

    lcoe = (Total_CAPEX + total_om_fix + total_om_var + total_fuel + co2_cost) / disc_energy
    lcoe = lcoe / 1.2439
    
    return lcoe

# Collect technologies from df and df2
storage_techs = df.columns
other_techs = df2.columns

# Generate points for specific ranges
ultra_high_res_part = np.logspace(start=0.0, stop=0.1, num=500, base=10)  # Ultra-high resolution for 10^0 to 10^0.1
low_res_part = np.logspace(start=0.1, stop=1.0, num=500, base=10)         # High resolution for 10^0.1 to 10^1
high_res_part = np.logspace(start=1.0, stop=3.0, num=500, base=10)        # Regular resolution for 10^1 to 10^3

# Combine and deduplicate
time_values = np.unique(np.concatenate((ultra_high_res_part, low_res_part, high_res_part)))

# Results list
results = []

# Run calculations for storage techs using calculate_lcos
for technology in storage_techs:
    technology_params = df[technology]
    for t in time_values:
        # Calculate LCOS for storage technologies
        function_value = calculate_lcos(t, technology_params)
        
        # Append results as lines
        results.append({
            'Technology': technology,
            't': t,
            'LCOS/E': function_value,
        })

# Run calculations for other techs using calculate_lcoe
for technology in other_techs:
    technology_params = df2[technology]
    for t in time_values:
        # Calculate LCOS for storage technologies
        function_value = calculate_lcoe(t, technology_params)
        
        # Append results as lines
        results.append({
            'Technology': technology,
            't': t,
            'LCOS/E': function_value,
        })


results_df = pd.DataFrame(results)

results_df.to_csv(folder + 'error bounds gas3.csv', index = False)

# Concatenate all techs without duplicates
all_techs = df.columns.union(df2.columns)

bounds = ["Gas CCGT"]
lines = ["Li-ion with replacement", "Pumped Hydro", "HD Hydro", "Hydrogen"]

import seaborn as sns
from matplotlib import rcParams
from matplotlib import font_manager

# Path to the folder where your Roboto fonts are installed
font_dir = '/Users/barnabywinser/Downloads/Roboto/'

# Load all fonts from that directory
for font in font_manager.findSystemFonts(fontpaths=font_dir):
    font_manager.fontManager.addfont(font)

# Set the font family to Roboto
rcParams['font.family'] = 'Roboto'

rcParams['axes.titleweight'] = 'semibold'
rcParams['font.size'] = 14

palette = sns.color_palette("tab10", 8)

# Define a dictionary to map technologies to their colors
color_dict = {
    "Gas CCGT": "#FF5733",               # Orange-red color for Gas CCGT
    "Gas OCGT": '#FF5733',
    "HD Hydro": "#3498DB",                # Blue color for HD Hydro
    "Pumped Hydro": "#2ECC71",            # Green color for Pumped Hydro
    "Li-ion": "#9B59B6",  # Purple color for Li-ion (2030 NREL)
    "Li-ion with replacement": "#9B59B6",      # Same color as Li-ion (2030 NREL forecast)
    "Gas CCGT L": "#FF5733",              # Same color for lower bound
    "Gas CCGT U": "#FF5733",
    "Hydrogen": "#FDDA0D"               # Same color for upper bound
}

# Function to plot bounds with custom color
def plot_technology_bounds(technology, df, color_dict, palette=None, idx=None):
    subset = df[df['Technology'] == technology]
    
    # Check for lower (L) and upper (U) bounds
    tech_l = df[df['Technology'] == f"{technology} L"]
    tech_u = df[df['Technology'] == f"{technology} U"]
    
    # Use custom color if available, otherwise use palette
    color = color_dict.get(technology, palette[idx] if palette else 'black')
    
    if not tech_l.empty and not tech_u.empty and len(tech_l) == len(tech_u):
        plt.plot(tech_l['t'], tech_l['LCOS/E'], color=color, linestyle=':', linewidth=2.5, label=f"{technology} L")
        plt.plot(tech_u['t'], tech_u['LCOS/E'], color=color, linestyle=':', linewidth=2.5, label=f"{technology} U")
        plt.fill_between(subset['t'], tech_l['LCOS/E'], tech_u['LCOS/E'], color=color, alpha=0.2)

# Function to plot lines with custom color and linestyle
def plot_technology_lines(technology, df, color_dict, palette=None, idx=None):
    subset = df[df['Technology'] == technology]
    
    # Use custom color if available, otherwise use palette
    color = color_dict.get(technology, palette[idx] if palette else 'black')
    
    # Set linestyle for "Li-ion with replacement" to dotted, otherwise use solid
    #linestyle = '--' if technology == "Li-ion with replacement" else '-' -- input linestyle=linestyle below to use
    
    # Plot the main line for the technology
    plt.plot(subset['t'], subset['LCOS/E'], label=technology, color=color, linewidth=2.5)

# Plotting starts here
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot technologies with bounds, using color dictionary
for idx, technology in enumerate(bounds):
    plot_technology_bounds(technology, results_df, color_dict, palette=palette, idx=idx)

# Plot other lines with solid style, using color dictionary
for idx, technology in enumerate(lines):
    plot_technology_lines(technology, results_df, color_dict, palette=palette, idx=idx)

# Set labels, title, and axis limits
#plt.suptitle("Levelized Cost of Energy/Storage by System Duration and Technology", fontweight='semibold')
ax1.set_xlabel('Continuous discharge duration', fontweight='semibold', labelpad = 5)
ax1.set_ylabel('£/MWh', fontweight='semibold', rotation = 0, labelpad = 30)
ax1.set_xlim(0, 12)
ax1.set_ylim(ya, yb)  # Assuming ya and yb are defined

# Set the x-axis to logarithmic scale and define limits
ax1.set_xscale('log')
ax1.set_xlim(2, 100)
# Customize tick locations (logarithmic positions)
ticks = [2, 5, 10, 20, 50, 100]  # Linear values at specific log positions
ax1.set_xticks(ticks)

# Replace log-scale labels with linear labels
ax1.set_xticklabels([str(tick) for tick in ticks])

"""
# Add secondary axis for CF (t / 24)
ax2 = ax1.twiny()


# Set the limits for the top axis (CF) to match the bottom axis, scaled by 1/24
t_min, t_max = ax1.get_xlim()
cf_min, cf_max = 100 * t_min / 24, 100 * t_max / 24
ax2.set_xlim(cf_min, cf_max)

# Set label for top axis
ax2.set_xlabel('Capacity Factor (%) - Gas', fontweight='semibold', labelpad = 8)
"""


# Define custom legend elements
legend_elements = [
    Line2D([0], [0], color=color_dict['Gas CCGT'], lw=2.5, linestyle=':', label='Gas CCGT'),
    Line2D([0], [0], color=color_dict['Li-ion with replacement'], lw=2.5, linestyle='-', label='Li-ion with replacement'),
    Line2D([0], [0], color=color_dict['HD Hydro'], lw=2.5, linestyle='-', label='HD Hydro'),
    Line2D([0], [0], color=color_dict['Pumped Hydro'], lw=2.5, linestyle='-', label='Pumped Hydro'),
    Line2D([0], [0], color=color_dict['Hydrogen'], lw=2.5, linestyle='-', label='Hydrogen')
]

# Add custom legend to the plot
legend = ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# Customize the grid and axis spines
ax1.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.5)
#for spine in ax1.spines.values():
 #   spine.set_visible(False)

# Function to format the y-axis ticks with £ sign
def pounds(x, pos):
    return f'£{int(x)}'

# Apply the custom formatter to the y-axis
ax1.yaxis.set_major_formatter(FuncFormatter(pounds))

# Customize the tick marks
#ax1.tick_params(axis='both', which='both', length=0)


# Use tight_layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 1.01])

# Save the plot
plt.savefig(folder + scenario + graph + "g1", dpi=300, transparent = True)

# Show the plot
plt.show()



# Extract LCOS values for t = 10 for comparison
#HD_10 = results_df[(results_df['Technology'] == 'HD Hydro') & (results_df['t'] == 10)]['LCOS/E'].values[0]
#LI_c = results_df[(results_df['Technology'] == 'Li-ion (2030 NREL forecast)') & (results_df['t'] == 10)]['LCOS/E'].values[0]
#LI_l = results_df[(results_df['Technology'] == 'Li-ion (2030 NREL forecast) L') & (results_df['t'] == 10)]['LCOS/E'].values[0]

# Print comparison
#print(f"HD {scenario} vs. Lithium mid: {HD_10 / LI_c}")
#print(f"HD vs. Lithium low: {HD_10 / LI_l}")


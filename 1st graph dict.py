from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib import rcParams
from matplotlib import font_manager
import pandas as pd
import math
import matplotlib.pyplot as plt
from functions import calculate_lcos, calculate_lcoe


scenario = "older w rep"
folder = "/Users/barnabywinser/Documents/local data/"
from matplotlib.ticker import FuncFormatter

# Application parameters
Cap_p_nom = 10  # Power in MW
Cyc_pa = 365  # Cycles per annum
P_el = 50  # Electricity purchase price $/MWh
el_g = 0  # Electricity price escalator % p.a.

ya = 50
yb = 200

palette = sns.color_palette("tab10", 8)

bounds = ["Li-ion older NREL forecast with replacement"]# "Li-ion older NREL forecast"]
lines = ["HD Hydro","Li-ion older NREL forecast with replacement", "Pumped Hydro"]# "Li-ion older NREL forecast"]#,"Li-ion with replacement", "Li-ion older NREL forecast", "Li-ion (2030 NREL forecast)", "Pumped Hydro"]

# Define a dictionary to map technologies to their colors
color_dict = {
    "Li-ion (2030 NREL forecast)": palette[1],
    "Gas CCGT": "#FF5733",               # Orange-red color for Gas CCGT
    "Gas OCGT": '#FF5733',
    "HD Hydro": "#3498DB",                # Blue color for HD Hydro
    "Pumped Hydro": "#2ECC71",            # Green color for Pumped Hydro
    "Li-ion older NREL forecast": "#FF5733",
    "Li-ion older NREL forecast with replacement": "#FF5733",
    # Purple color for Li-ion (2030 NREL)
    "Li-ion with replacement": "#9B59B6",      # Same color as Li-ion (2030 NREL forecast)
    "Gas CCGT L": "#FF5733",              # Same color for lower bound
    "Gas CCGT U": "#FF5733"               # Same color for upper bound
}

bounds_lines = set(bounds) | set(lines)

# Automatically generate legend elements
legend_elements = []
for tech in bounds_lines:
    legend_elements.append(
        Line2D([0], [0], color=color_dict[tech], lw=2.5, linestyle='-', label=tech)
    )

graph = 'b'

lcos_scenario = 'lcos'
lcoe_scenario = 'lcoe'
file_path = '/Users/barnabywinser/Documents/local data/Tech cost parameters - li_HD_ph.xlsx'

# Define technology parameters as df
df = pd.read_excel(file_path, sheet_name=lcos_scenario, index_col=0, header=0)
df2 = pd.read_excel(file_path, sheet_name='lcoe (UK)', index_col=0)

# Collect technologies from df and df2
storage_techs = df.columns
other_techs = df2.columns

# Generate time_values for the LCOS/LCOE calculations
time_values = list(range(1,11))

# Results list
results = []

# Run calculations for storage techs using calculate_lcos
for technology in storage_techs:
    technology_params = df[technology]
    for t in time_values:

        # Calculate LCOS and other components for storage technologies
        result_values = calculate_lcos(t, technology_params)
        
        # Append results, including intermediate values
        results.append({
            'Tech': technology,
            't': t,
            'LCOS': result_values['LCOS'],
            'CAPEX': result_values['CAPEX'],
            'Replacement': result_values['Replacement'],
            'Charging': result_values['Charging'],
            'Energy discharged': result_values['Energy discharged'],
            'O&M': result_values['O&M'],
            'End of life': result_values['End of life'],
        })


results_df = pd.DataFrame(results)

results_df.to_csv(folder + 'lcos and rev.csv', index = False)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from functions import calculate_lcos
import os

# Get the current directory (where the script is located)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
input_folder = os.path.join(base_dir, 'Input parameters')
output_folder = os.path.join(base_dir, 'Results')
input_file = "Input parameters.xlsx"  # File name only HD Hydro - head heights
input_path = os.path.join(input_folder, input_file)

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "Full graph results.csv")
output_file_2 = os.path.join(output_folder, "Results broken down.csv")

# Manually specify which columns to use from the data?
manual = 'y'

# Which techs to plot as bounds and lines? only binds if above is 'y'
bounds = ["Li-ion (2030 NREL forecast)", "HD Hydro", "HD Hydro (cost down)"]
lines = ["HD Hydro (150m)", "HD Hydro (200m)", "HD Hydro"]

# Parameters
sheet = 'SSW'
Cap_p_nom = 10  # Power in MW
Cyc_pa = 365  # Cycles per annum
P_el = 30  # Electricity purchase price $/MWh
ya, yb = 50, 400  # Y-axis bounds
t_values = [t / 10 for t in range(1, 121)] #values of t for the plot
select_t = [x for x in t_values if (x).is_integer() and int(x) % 2 == 0] #values of t for the results breakdown (even whole numbers from above)

# Read input data
df = pd.read_excel(input_path, sheet_name=sheet, index_col=0)

# Plot settings
rcParams['font.family'] = 'Barlow'
rcParams['axes.titleweight'] = 'semibold'
rcParams['font.size'] = 12

# Color mapping
palette = sns.color_palette("muted", 12)
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

if manual == 'y':
    # Bounds and lines
    bounds = bounds
    lines = lines
else:
    lines = df.columns
    bounds = df.columns

# Calculate LCOS
results = []
for technology in df.columns:
    for t in t_values:
        lcos_value = calculate_lcos(t, technology, df, Cap_p_nom, Cyc_pa, P_el)['LCOS']
        results.append({'Technology': technology, 't': t, 'LCOS': lcos_value})
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv(output_file, index=False)

"""This is to save the tables in Results broken down"""


results_breakdown = []
for technology in df.columns:
    for t in select_t:
        lcos_data = calculate_lcos(t, technology, df, Cap_p_nom, Cyc_pa, P_el)
        lcos_data['Technology'] = technology
        lcos_data['Duration'] = t
        results_breakdown.append(lcos_data)
breakdown_df = pd.DataFrame(results_breakdown)

# Save detailed results to CSV
breakdown_df.to_csv(output_file_2, index=False)

def plot_technology(technology, df, color_dict, palette=None, idx=None, bounds=False):
    """
    Plot lines or bounds for a technology.

    Parameters:
        technology: str, name of the technology.
        df: DataFrame, contains data for plotting.
        color_dict: dict, maps technologies to specific colors.
        palette: list, optional color palette.
        idx: int, index for palette if no custom color.
        bounds: if True, plots bounds as well as lines.
    """
    subset = df[df['Technology'] == technology]
    color = color_dict.get(technology, palette[idx] if palette else 'black')

    if bounds:
        tech_l = df[df['Technology'] == f"{technology} L"]
        tech_u = df[df['Technology'] == f"{technology} U"]
        if not tech_l.empty and not tech_u.empty and len(tech_l) == len(tech_u):
            plt.fill_between(subset['t'], tech_l['LCOS'], tech_u['LCOS'], color=color, alpha=0.2)
    else:
        plt.plot(subset['t'], subset['LCOS'], label=technology, color=color, linewidth=2.5)

# Plotting starts here
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot technologies with bounds
for idx, technology in enumerate(bounds):
    plot_technology(technology, results_df, color_dict, palette=palette, idx=idx, bounds=True)

# Plot other technologies as lines
for idx, technology in enumerate(lines):
    plot_technology(technology, results_df, color_dict, palette=palette, idx=idx)

# Titles and labels
#plt.suptitle("Levelised Cost of Storage (LCOS) by System Duration and Technology", fontweight='semibold')
plt.xlabel('System Duration (hrs)', fontweight='semibold')
plt.ylabel('LCOS ($/MWh)', fontweight='semibold')

# Add axis lines
plt.axhline(0, color='black', linewidth=1.5)  # Horizontal line at y=0
plt.axvline(0, color='grey', linewidth=1.5, alpha=0.5)  # Vertical line at x=0
plt.axhline(ya, color='grey', linewidth=1.5, alpha=0.5)  # Horizontal line at ya

# Legend and grid
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
plt.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)

# Customize spines and ticks
for spine in ax1.spines.values():
    spine.set_visible(False)
plt.tick_params(axis='both', which='both', length=0)

# Axis limits
plt.xlim(0, 12)
plt.ylim(ya, yb)

plt.savefig(os.path.join(output_folder, f"{input_file}_{sheet}.png"), dpi=300)

# Show the plot
plt.show()

# Use tight layout and save
plt.tight_layout(rect=[0, 0, 1, 1.01])


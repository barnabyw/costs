import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from functions import calculate_lcos, calculate_lcoe
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
output_file = os.path.join(output_folder, "Graph results.csv")
output_file_2 = os.path.join(output_folder, "Results broken down.csv")

# Manually specify which columns to use from the data?
manual = 'y'

# Which techs to plot as bounds and lines? only binds if above is 'y'
bounds = ["Li-ion (2030 NREL forecast)", "HD Hydro", "Gas OCGT"]
lines = ["HD Hydro (150m)", "HD Hydro (200m)", "HD Hydro", "Gas OCGT"]

# Parameters
lcos_sheet = 'SSW'
lcoe_sheet = 'lcoe (UK)'
Cap_p_nom = 10  # Power in MW
Cyc_pa = 365  # Cycles per annum
P_el = 30  # Electricity purchase price $/MWh (for storage)
ya, yb = 50, 400  # Y-axis bounds
t_values = [t / 10 for t in range(1, 121)] #values of t for the plot
select_t = [x for x in t_values if (x).is_integer() and int(x) % 2 == 0] #values of t for the results breakdown (even whole numbers from above)

# Read input data
df = pd.read_excel(input_path, sheet_name=lcos_sheet, index_col=0)
df2 = pd.read_excel(input_path, sheet_name=lcoe_sheet, index_col=0)

# Collect technologies from df and df2
storage_techs = df.columns
other_techs = df2.columns

# Plot settings
rcParams['font.family'] = 'Barlow'
rcParams['axes.titleweight'] = 'semibold'
rcParams['font.size'] = 12

# Color mapping
palette = sns.color_palette("muted", 20)
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
    storage_techs = [tech for tech in storage_techs if tech in bounds or tech in lines]
    other_techs = [tech for tech in other_techs if tech in bounds or tech in lines]
    
# Calculate LCOS and LCOE
results = []

for technology in df.columns:  # Iterate through storage technologies
    for t in t_values:
        # Calculate LCOS
        lcos_value = calculate_lcos(t, technology, df, Cap_p_nom, Cyc_pa, P_el)['LCOS']
        results.append({'Technology': technology, 't': t, 'LCOE/S': lcos_value})

for technology in df2.columns:  # Iterate through other technologies
    for t in t_values:
        # Calculate LCOE
        lcoe_value = calculate_lcoe(t, technology, df2)  # Pass necessary arguments to calculate_lcoe
        results.append({'Technology': technology, 't': t, 'LCOE/S': lcoe_value})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv(output_file, index=False)

"""This is to save the tables in Results broken down"""


def plot_technology(technology, df, metric, color_dict, palette=None, idx=None, bounds=False):
    """
    Plot lines or bounds for a technology for a specific metric (LCOS or LCOE).

    Parameters:
        technology: str, name of the technology.
        df: DataFrame, contains data for plotting.
        metric: str, column name of the metric to plot ('LCOS' or 'LCOE').
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
            plt.fill_between(subset['t'], tech_l[metric], tech_u[metric], color=color, alpha=0.2)
    else:
        plt.plot(subset['t'], subset[metric], label=f"{technology}", color=color, linewidth=2.5)

# Plotting starts here
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot LCOS and LCOE for each technology
for idx, technology in enumerate(bounds):
    plot_technology(technology, results_df, 'LCOE/S', color_dict, palette=palette, idx=idx, bounds=True)

for idx, technology in enumerate(lines):
    plot_technology(technology, results_df, 'LCOE/S', color_dict, palette=palette, idx=idx)

# Titles and labels
#plt.title("Comparison of LCOS and LCOE by System Duration and Technology", fontweight='bold')
plt.xlabel('System Duration (hrs)', fontweight='semibold')
plt.ylabel('LCOE/S ($/MWh)', fontweight='semibold')

# Add axis lines
plt.axhline(0, color='black', linewidth=1.5)  # Horizontal line at y=0
plt.axvline(0, color='grey', linewidth=1.5, alpha=0.5)  # Vertical line at x=0
plt.axhline(ya, color='grey', linewidth=1.5, alpha=0.5)  # Horizontal line at ya

# Set y-axis limits
plt.ylim(ya, yb)
# Axis limits
plt.xlim(0, 12)

# Legend and grid
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
plt.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)

# Customize spines and ticks
for spine in ax1.spines.values():
    spine.set_visible(False)
plt.tick_params(axis='both', which='both', length=0)

# Save and show the plot
plt.savefig(os.path.join(output_folder, f"{input_file}_lcos_lcoe_comparison.png"), dpi=300)
plt.tight_layout(rect=[0, 0, 1, 1.01])
plt.show()
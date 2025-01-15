import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from functions import calculate_lcos, calculate_lcoe

input_folder = '/Users/barnabywinser/Library/CloudStorage/OneDrive-SharedLibraries-Rheenergise/Commercial - Documents/Cost Models/LCOS/Input parameters/'
output_path = "/Users/barnabywinser/Library/CloudStorage/OneDrive-SharedLibraries-Rheenergise/Commercial - Documents/Cost Models/LCOS/Results/"

file = "Tech cost parameters - Ninja defaults.xlsx"
sheet = 'Sheet1'

# Application parameters (consistent across technologies)
Cap_p_nom = 10  # Power in MW (generally impact cancels out)
# t = 4  # Discharge duration hours
Cyc_pa = 365  # Cycles per annum
P_el = 50  # Electricity purchase price $/MWh

# axis bounds on graph
ya = 50 
yb = 400


# define path to input file (excel workbook)
input_path = input_folder + file

# define technology parameters as df
df = pd.read_excel(input_path, sheet_name=sheet, index_col=0, header=0)

# define the lines and bounds wanted on the graph
bounds = ["Li-ion (2030 NREL forecast)", "HD Hydro", "HD Hydro (cost down)"]
lines = ["Li-ion (2030 NREL forecast)","HD Hydro", "HD Hydro (cost down)"]

# Define the Seaborn color palette
palette = sns.color_palette("muted", 10)

# Define a dictionary to map technologies to their colors
color_dict = {
    "Li-ion (2030 NREL forecast)": palette[1],
    "Gas CCGT": "#FF5733",               # Orange-red color for Gas CCGT
    "Gas OCGT": '#FF5733',
    "HD Hydro": palette[0],                # Blue color for HD Hydro
    "HD Hydro (cost down)": palette[2],            # Green color for Pumped Hydro
    "Li-ion older NREL forecast": "#FF5733",
    "Li-ion older NREL forecast with replacement": "#FF5733",
    "Li-ion with replacement": "#9B59B6",      # Same color as Li-ion (2030 NREL forecast)
    "Gas CCGT L": "#FF5733",              # Same color for lower bound
    "Gas CCGT U": "#FF5733"               # Same color for upper bound
}


technologies = df.columns

results = []
t_values = [t / 10 for t in range(1, 121)]

for technology in technologies:
    technology_params = df[technology]
    for t in t_values:
        function_output = calculate_lcos(t, technology, df, Cap_p_nom, Cyc_pa, P_el)
        lcos_value = function_output['LCOS']
        results.append({
            'Technology': technology,
            't': t,
            'LCOS': lcos_value
        })

results_df = pd.DataFrame(results)

excel_file = '/Users/barnabywinser/Documents/local data/LCOS all techs.xlsx' 
csv_file = '/Users/barnabywinser/Documents/local data/LCOS all techs.csv'
results_df.to_excel(excel_file)
results_df.to_csv(output_path+"Results.csv", index=False)

# Set the font to Barlow
rcParams['font.family'] = 'Barlow'
rcParams['axes.titleweight'] = 'semibold'
rcParams['font.size'] = 12


plt.figure(figsize=(10, 6))

#technologies = ["HD Hydro", "Li-ion", "Li-ion (2030 NREL forecast)", "Pumped Hydro", "Vanadium Flow", "Compressed Air", "Hydrogen"]
technologies = ["HD Hydro", "Li-ion (2030 NREL forecast)"]


# Function to plot bounds with custom color
def plot_technology_bounds(technology, df, color_dict, palette=None, idx=None):
    subset = df[df['Technology'] == technology]
    
    # Check for lower (L) and upper (U) bounds
    tech_l = df[df['Technology'] == f"{technology} L"]
    tech_u = df[df['Technology'] == f"{technology} U"]
    
    # Use custom color if available, otherwise use palette
    color = color_dict.get(technology, palette[idx] if palette else 'black')
    
    if not tech_l.empty and not tech_u.empty and len(tech_l) == len(tech_u):
        plt.fill_between(subset['t'], tech_l['LCOS'], tech_u['LCOS'], color=color, alpha=0.2)

# Function to plot lines with custom color and linestyle
def plot_technology_lines(technology, df, color_dict, palette=None, idx=None):
    subset = df[df['Technology'] == technology]
    
    # Use custom color if available, otherwise use palette
    color = color_dict.get(technology, palette[idx] if palette else 'black')

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
        
# titles
plt.suptitle("Levelised Cost of Storage (LCOS) by System Duration and Technology", fontweight='semibold')
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

plt.savefig(output_path + sheet, dpi = 300)
# Show the plot
plt.show()
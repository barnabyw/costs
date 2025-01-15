import math

# Define function for LCOS
def calculate_lcos(t, technology, df, Cap_p_nom, Cyc_pa, P_el):
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
    LCOS = numerator / total_eout if total_eout > 0 else None

    # Return a dictionary with all values
    return {
        'CAPEX': Total_CAPEX/total_eout,
        'Replacement': Rep_disc/total_eout,
        'Charging': total_charge_cost/total_eout,
        'Energy discharged': total_eout,
        'O&M': total_om_cost/total_eout,
        'End of life': Total_EoL/total_eout,
        'LCOS': LCOS
    }


# Function for LCOE (unchanged)
def calculate_lcoe(t, technology_params, technology, df2):
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

    cf = t / 24
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
    
    return lcoe
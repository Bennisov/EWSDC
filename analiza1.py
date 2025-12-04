import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid as trapz

def load_data(file_path):
    # Load data, replace commas with dots, and convert to float array
    data_np_str = np.loadtxt(file_path, delimiter='\t', dtype=str)
    data_cleaned_str = np.char.replace(data_np_str, ',', '.')
    final_2d_float_array = data_cleaned_str.astype(float)
    return final_2d_float_array

# --- Amplitude Characteristic Functions (in dB) ---
def A_CR_dB(u):
    # |H(jω)| = u / sqrt(1 + u^2) for single CR (differentiator)
    A = u / np.sqrt(1 + u**2)
    return 20 * np.log10(A)

def A_CR_RC_dB(u):
    # |H(jω)| = u / (1 + u^2) for CR-RC (1st order shaper)
    A = u / (1 + u**2)
    return 20 * np.log10(A)

def A_CR2_RC_dB(u):
    # |H(jω)| = u^2 / (1 + u^2)^1.5 for CR^2-RC
    A = u**2 / (1 + u**2)**1.5
    return 20 * np.log10(A)

def A_CR_RC2_dB(u):
    # |H(jω)| = u / (1 + u^2)^1.5 for CR-RC^2
    A = u / (1 + u**2)**1.5
    return 20 * np.log10(A)

def A_CR_RC3_dB(u):
    # |H(jω)| = u / (1 + u**2)**2 for CR-RC^3
    A = u / (1 + u**2)**2
    return 20 * np.log10(A)

# --- Theoretical Transfer Function Magnitude (Linear Scale) ---
# Used for noise calculation: |H(jω)|^2
def H_mag_sq(A_dB_func, u):
    # |H(jω)|^2 = (10^(A_dB/20))^2 = 10^(A_dB/10)
    return 10**(A_dB_func(u) / 10.0)

# --- Configuration Setup ---
fs = ["100k", "500k", "1M"]
directory = "WOJCIK_26_11_2025"
conf = ["CR", "CRRC", "CR^2RC", "CRRC^2", "CRRC^3"]
conf_functions = {
    "CR": A_CR_dB,
    "CRRC": A_CR_RC_dB,
    "CR^2RC": A_CR2_RC_dB,
    "CRRC^2": A_CR_RC2_dB,
    "CRRC^3": A_CR_RC3_dB
}

# --- 1. Amplitude Characteristics Plotting (Your initial code) ---

# Plot for Wzmacniacz (Amplifier)
wzmacniacz = load_data(f"{directory}/wzmacniacz.txt")
plt.figure()
plt.plot(wzmacniacz[:,0], wzmacniacz[:,1], label="Measured Amplifier", color='green')
plt.xscale('log')
plt.ylabel("Wzmocnienie [dB]")
plt.xlabel("Częstotliwość [Hz]")
plt.title("Charakterystyka Amplitudowa Wzmacniacza")
plt.grid(True, which="both", ls="--")
plt.savefig("plots/wzmacniacz_amplitude.png")
plt.close()
# 
# Plots for Filter Configurations
for f in fs:
    for c, c_f in conf_functions.items():
        file_path = f"{directory}/{c}{f}.txt"
        data = load_data(file_path)
        
        freq = f.replace("k", "e3").replace("M", "e6")
        freq = float(freq)
        # Calculate time constant tau = 1/(2*pi*f_g)
        tau = 1 / (2 * np.pi * freq)
        # Calculate normalized frequency u = 2*pi*f*tau = f/f_g
        u = 2 * np.pi * data[:, 0] * tau
        
        A_values = c_f(u)
        
        plt.figure()
        plt.plot(data[:,0], data[:,1], label="Zmierzone dane [dB]", color='blue', marker='.', linestyle='')
        plt.plot(data[:,0], A_values, label=f"Teoretyczna ({f.replace('k', ' kHz').replace('M', ' MHz')}) [dB]", color='red')
        
        plt.xscale('log')
        plt.title(f"Charakterystyka Amplitudowa: {c}")
        plt.xlabel("Częstotliwość [Hz]")
        plt.ylabel("Wzmocnienie [dB]")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(f"plots/{c}{f}_amplitude.png")
        plt.close()
# 
# --- 2. Noise Spectral Density Plotting and V_rms Calculation ---

# Load amplifier noise data (in uV/sqrt(Hz))
wzmacniacz_noise_data = load_data(f"{directory}/wzmacniacz_noise.txt")

# Calculate measured PSD in (uV^2/Hz) for amplifier
wzmacniacz_noise_psd_meas = wzmacniacz_noise_data[:, 1]**2

# Estimate the input white noise PSD (S_in) from the amplifier output noise
# Assuming the amplifier acts as a constant gain for f < 1MHz, we can estimate S_in
# Let's take the mean of the amplifier's measured PSD up to 1 MHz (index approx 100)
# (Your data's frequency range is needed, but assuming standard sweep to 10^7 Hz)
# For simplicity, we'll assume the input noise PSD is constant.
# The maximum value of the theoretical CR-RC^n filter's PSD is a good point to relate.
# From the old report (Fig 7), the noise PSD is roughly 2000 uV^2/Hz at low f.
# We will use the average measured PSD over the flat region of the *amplifier* noise.
flat_region_end_index = np.where(wzmacniacz_noise_data[:, 0] < 1e6)[0][-1] if len(wzmacniacz_noise_data[:, 0]) > 0 else -1
S_in_uV2_Hz = np.mean(wzmacniacz_noise_psd_meas[:flat_region_end_index]) # The "input" noise PSD is measured at the amp output *before* filtering.

# Plot for Wzmacniacz Noise (PSD)
plt.figure()
plt.plot(wzmacniacz_noise_data[:,0], wzmacniacz_noise_psd_meas, label="Wzmacniacz", color='green', marker='.', linestyle='')
plt.xscale('log')
plt.ylabel("Widmowa gęstość mocy szumów [µV²/Hz]")
plt.xlabel("Częstotliwość [Hz]")
plt.title("Widmowa Gęstość Mocy Szumów Wzmacniacza")
plt.grid(True, which="both", ls="--")
plt.savefig("plots/wzmacniacz_noise_psd.png")
plt.close()
# 

# Initialize table data for Vrms results
vrms_results = []

for f in fs:
    for c, c_f in conf_functions.items():
        # Load filter data (Noise in uV/sqrt(Hz))
        file_path = f"{directory}/{c}{f}_noise.txt" # Assuming noise files are named like 'CR100k_noise.txt'
        data_noise = load_data(file_path)
        
        # Calculate measured PSD in (uV^2/Hz)
        noise_psd_meas = data_noise[:, 1]**2
        
        freq = f.replace("k", "e3").replace("M", "e6")
        freq = float(freq)
        tau = 1 / (2 * np.pi * freq)
        
        # Calculate normalized frequency u
        u = 2 * np.pi * data_noise[:, 0] * tau
        
        # Calculate Theoretical PSD: S_out = |H(jω)|^2 * S_in
        H2_values = H_mag_sq(c_f, u)
        noise_psd_theor = H2_values * S_in_uV2_Hz
        
        # Plot Measured and Theoretical Noise PSD
        plt.figure()
        plt.plot(data_noise[:, 0], noise_psd_meas, label="Zmierzone dane [µV²/Hz]", color='blue', marker='.', linestyle='')
        plt.plot(data_noise[:, 0], noise_psd_theor, label=f"Teoretyczna ({f.replace('k', ' kHz').replace('M', ' MHz')}) [µV²/Hz]", color='red')
        
        plt.xscale('log')
        plt.title(f"Widmowa Gęstość Mocy Szumów: {c}")
        plt.xlabel("Częstotliwość [Hz]")
        plt.ylabel("Widmowa gęstość mocy szumów [µV²/Hz]")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(f"plots/{c}{f}_noise_psd.png")
        plt.close()
        # 

        # Calculate Vrms - Formula: V_rms = sqrt(Integral(S_out * df)) [cite: 114]
        # Measured Vrms (from uV^2/Hz) - need to convert uV^2/Hz to V^2/Hz before integration
        Vrms_meas_V = np.sqrt(trapz(y=noise_psd_meas * 1e-12, x=data_noise[:, 0]))
        Vrms_meas_uV = Vrms_meas_V * 1e6
        
        # Theoretical Vrms (from uV^2/Hz) - need to convert uV^2/Hz to V^2/Hz before integration
        Vrms_theor_V = np.sqrt(trapz(y=noise_psd_theor * 1e-12, x=data_noise[:, 0]))
        Vrms_theor_uV = Vrms_theor_V * 1e6
        
        # Save results
        vrms_results.append({
            "Konfiguracja": c,
            "Częstotliwość Graniczna (fg)": f.replace("k", " kHz").replace("M", " MHz"),
            "Vrms Zmierzone [µV]": Vrms_meas_uV,
            "Vrms Teoretyczne [µV]": Vrms_theor_uV
        })

# Create and display the table
vrms_df = pd.DataFrame(vrms_results)

# Pivot the table for a cleaner presentation, similar to the old report's table 1
vrms_pivot_meas = vrms_df.pivot(index="Konfiguracja", columns="Częstotliwość Graniczna (fg)", values="Vrms Zmierzone [µV]")
vrms_pivot_theor = vrms_df.pivot(index="Konfiguracja", columns="Częstotliwość Graniczna (fg)", values="Vrms Teoretyczne [µV]")

print("\n--- Zmierzone Wartości Vrms [µV] ---")
print(vrms_pivot_meas.round(2))
print("\n--- Teoretyczne Wartości Vrms [µV] ---")
print(vrms_pivot_theor.round(2))
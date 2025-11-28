import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    data_np_str = np.loadtxt(file_path, delimiter='\t', dtype=str)
    data_cleaned_str = np.char.replace(data_np_str, ',', '.')
    final_2d_float_array = data_cleaned_str.astype(float)
    return final_2d_float_array

wzmacniacz = load_data("WOJCIK_26_11_2025/wzmacniacz.txt")

fs = ["100k", "500k", "1M"]
directory = "WOJCIK_26_11_2025"
conf = ["CR", "CRRC", "CR^2RC", "CRRC^2", "CRRC^3"]
def A_CR_dB(u):
    A = u / np.sqrt(1 + u**2)
    return 20 * np.log10(A)
def A_CR_RC_dB(u):
    A = u / (1 + u**2)
    return 20 * np.log10(A)
def A_CR2_RC_dB(u):
    A = u**2 / (1 + u**2)**1.5
    return 20 * np.log10(A)
def A_CR_RC2_dB(u):
    A = u / (1 + u**2)**1.5
    return 20 * np.log10(A)
def A_CR_RC3_dB(u):
    A = u / (1 + u**2)**2
    return 20 * np.log10(A)
conf_functions = {
    "CR": A_CR_dB,
    "CRRC": A_CR_RC_dB,
    "CR^2RC": A_CR2_RC_dB,
    "CRRC^2": A_CR_RC2_dB,
    "CRRC^3": A_CR_RC3_dB
}

for f in fs:
    for c,c_f in conf_functions.items():
        file_path = f"{directory}/{c}{f}.txt"
        data = load_data(file_path)
        freq = f.replace("k", "e3").replace("M", "e6")
        freq = float(freq)
        tau = 1 / (2 * np.pi * freq)
        u = 2 * np.pi * data[:, 0] * tau
        A_values = c_f(u)
        plt.figure()
        plt.plot(data[:,0],data[:,1], label="Measured Data Rescaled", color='blue')
        plt.plot(data[:,0], A_values, label=f"Theoretical", color='red')
        
        plt.xscale('log')
        plt.title(f"Configuration: {c}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("H(f)")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(f"plots/{c}{f}.png")
        plt.close()
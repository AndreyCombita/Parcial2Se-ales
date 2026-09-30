import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, firwin, butter, filtfilt

# ==========================================
# 1. CARGA DE PARÁMETROS Y ARCHIVO
# ==========================================
file_path = "05_1_modulated_signal_20.0_200.0_40.0.csv"

f_carrier = 20000.0   # 20.0 kHz en Hz
fs = 200000.0         # 200.0 kHz en Hz
T_ms = 40.0           # Duración total 40 ms

df = pd.read_csv(file_path)
s_t = df.iloc[:, -1].values.astype(np.float64)
N = len(s_t)
t = np.arange(N) / fs  # Vector de tiempo en segundos

# ==========================================
# 2. ENVOLVENTE Y FRECUENCIA TRANSMITIDA (b)
# ==========================================
analytic_signal = hilbert(s_t)
envelope_raw = np.abs(analytic_signal)

# Eliminación de componente DC para hallar fm
dc_component = np.mean(envelope_raw)
envelope_ac = envelope_raw - dc_component

# FFT de la envolvente
fft_vals = np.fft.rfft(envelope_ac)
fft_freqs = np.fft.rfftfreq(N, d=1/fs)
idx_max = np.argmax(np.abs(fft_vals)[1:]) + 1
f_transmitted = fft_freqs[idx_max]

print("================ PUNTO 1 ================")
print(f"Frecuencia Portadora (fc): {f_carrier/1000:.2f} kHz")
print(f"Frecuencia Muestreo (fs):  {fs/1000:.2f} kHz")
print(f"Frecuencia Transmitida (fm): {f_transmitted:.2f} Hz")
print("=========================================")

# ==========================================
# 3. FILTRADO FIR (c) E IIR (d)
# ==========================================
cutoff_freq = f_transmitted * 2.5  # Frecuencia de corte ~8.75 kHz

# Filtro FIR (Orden 50)
b_fir = firwin(51, cutoff=cutoff_freq, fs=fs, window='hamming')
envelope_fir = filtfilt(b_fir, [1.0], envelope_ac)

# Filtro IIR Butterworth (Orden 4 < 10)
b_iir, a_iir = butter(4, Wn=cutoff_freq, btype='low', fs=fs)
envelope_iir = filtfilt(b_iir, a_iir, envelope_ac)

# ==========================================
# 4. GRAFICACIÓN
# ==========================================
display_ms = 3.0  # Muestra 3 ms para visualizar claramente las ondas
mask = (t * 1000) <= display_ms

fig, axs = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)
fig.suptitle("(Punto 1) Demodulación AM con Filtros FIR e IIR", fontsize=13, fontweight='bold')

axs[0].plot(t[mask]*1000, s_t[mask], color='navy', lw=0.8)
axs[0].set_title("1. Señal Modulada Original s(t)")
axs[0].set_ylabel("Amplitud")
axs[0].grid(True, linestyle='--')

axs[1].plot(t[mask]*1000, envelope_raw[mask], color='darkorange', lw=1)
axs[1].set_title("2. Magnitud Envolvente de Hilbert (Con Componente DC)")
axs[1].set_ylabel("Amplitud")
axs[1].grid(True, linestyle='--')

axs[2].plot(t[mask]*1000, envelope_fir[mask], color='green', lw=1.2)
axs[2].set_title(f"3. Señal Filtrada con FIR (Orden 50) - fm = {f_transmitted:.0f} Hz")
axs[2].set_ylabel("Amplitud")
axs[2].grid(True, linestyle='--')

axs[3].plot(t[mask]*1000, envelope_iir[mask], color='crimson', lw=1.2)
axs[3].set_title(f"4. Señal Filtrada con IIR Butterworth (Orden 4) - fm = {f_transmitted:.0f} Hz")
axs[3].set_xlabel("Tiempo (ms)")
axs[3].set_ylabel("Amplitud")
axs[3].grid(True, linestyle='--')

plt.tight_layout()
plt.savefig("grafica_punto1.png", dpi=300)
plt.show()
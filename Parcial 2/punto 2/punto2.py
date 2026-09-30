import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ==========================================
# 1. CARGA DE ARCHIVO Y PARÁMETROS
# ==========================================
file_path = "05_2_modulated_signal_30.0_300.0_1.0.csv"

f_carrier = 30000.0   # 30.0 kHz
fs = 300000.0         # 300.0 kHz
T_bit = 0.001         # 1.0 ms por bit

df = pd.read_csv(file_path)
s_t = df.iloc[:, -1].values.astype(np.float64)
N = len(s_t)
t = np.arange(N) / fs

smp_per_bit = int(round(fs * T_bit))
num_bits = N // smp_per_bit

# ==========================================
# 2. ENVOLVENTE Y EXTRACTION DE BITS (b)
# ==========================================
envelope = np.abs(hilbert(s_t))

square_wave = np.zeros_like(s_t)
bit_stream = []
threshold = (np.max(envelope) + np.min(envelope)) / 2.0

for i in range(num_bits):
    start, end = i * smp_per_bit, (i + 1) * smp_per_bit
    bit_val = 1 if np.mean(envelope[start:end]) > threshold else 0
    bit_stream.append(bit_val)
    square_wave[start:end] = bit_val

# ==========================================
# 3. DECODIFICACIÓN ASCII (c)
# ==========================================
bytes_list = [bit_stream[i:i+8] for i in range(0, len(bit_stream), 8)]
chars = []

print("================ PUNTO 2 ================")
print("Desglose de Bits a ASCII:")
for idx, b8 in enumerate(bytes_list):
    byte_str = "".join(map(str, b8))
    dec_val = int(byte_str, 2)
    char_val = chr(dec_val)
    chars.append(char_val)
    print(f"Byte {idx+1:02d}: {byte_str} -> Decimal: {dec_val:3d} -> Carácter: '{char_val}'")

decoded_message = "".join(chars)
print("\n-----------------------------------------")
print(f"MENSAJE DECODIFICADO COMPLETO: {decoded_message}")
print("=========================================")

# ==========================================
# 4. GRAFICACIÓN (b)
# ==========================================
display_ms = 40.0  # Muestra los primeros 40 ms (40 bits)
mask = (t * 1000) <= display_ms

fig, axs = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
fig.suptitle("(Punto 2) Demodulación de Señal Digital y Reconstrucción de Bits", fontsize=13, fontweight='bold')

axs[0].plot(t[mask]*1000, s_t[mask], color='navy', lw=0.8)
axs[0].set_title("1. Señal Modulada Original s(t)")
axs[0].set_ylabel("Amplitud")
axs[0].grid(True, linestyle='--')

axs[1].plot(t[mask]*1000, envelope[mask], color='darkorange', lw=1)
axs[1].set_title("2. Magnitud de la Transformada de Hilbert (Envolvente)")
axs[1].set_ylabel("Amplitud")
axs[1].grid(True, linestyle='--')

axs[2].step(t[mask]*1000, square_wave[mask], color='green', where='post', lw=1.5)
axs[2].set_title("3. Señal Cuadrada Digital Reconstruida (Bits ASCII)")
axs[2].set_xlabel("Tiempo (ms)")
axs[2].set_ylabel("Nivel Lógico (0 / 1)")
axs[2].set_ylim(-0.2, 1.2)
axs[2].grid(True, linestyle='--')

plt.tight_layout()
plt.savefig("grafica_punto2.png", dpi=300)
plt.show()
#!/usr/bin/env python3
"""
Lectura de ADS1292R desde Arduino vía Serial.
Reconstruye valores de 24 bits de 2 canales ECG.
Aplica filtros y guarda CSV.
"""

import tkinter as tk
import serial
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirnotch

# ==== CONFIGURACIÓN ====
PORT = "/dev/ttyUSB0"
BAUD = 115200        # ADS1292R típico
FS = 250             # Frecuencia de muestreo (ajustar según configuración ADS1292R)
SAVE_FILE = "ecg_ads1292r.csv"
NOTCH_FREQ = 50.0    # Cambiar a 60.0 si su red es de 60 Hz
Q = 30.0

# ==== ESCALADO ====
# ADS1292R: resolución 24 bits -> ±(2^23 - 1)
# Vref = 2.42 V (por defecto en algunos módulos)
# Ganancia típica = 6 (depende config de registro)
VREF = 2.42
GAIN = 6.0
LSB = (VREF / GAIN) / (2**23 - 1)  # Volts por bit

# ==== FILTROS ====
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=FS):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

def notch_filter(data, freq=NOTCH_FREQ, fs=FS, Q=Q):
    b, a = iirnotch(freq, Q, fs)
    return lfilter(b, a, data)

# ==== RECONSTRUCCIÓN 24 BITS SIGNADO ====
def twos_complement_24bit(val):
    if val & 0x800000:
        val = -((~val & 0xFFFFFF) + 1)
    return val

def parse_packet(packet_bytes):
    if len(packet_bytes) != 9:
        return None, None
    status = packet_bytes[0:3]  # no se usa por ahora
    ch1 = (packet_bytes[3] << 16) | (packet_bytes[4] << 8) | packet_bytes[5]
    ch2 = (packet_bytes[6] << 16) | (packet_bytes[7] << 8) | packet_bytes[8]
    ch1 = twos_complement_24bit(ch1)
    ch2 = twos_complement_24bit(ch2)
    # Convertir a microvoltios
    ch1_uv = ch1 * LSB * 1e6
    ch2_uv = ch2 * LSB * 1e6
    return ch1_uv, ch2_uv

# ==== INICIALIZAR SERIAL ====
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

# ==== CSV ====
csvfile = open(SAVE_FILE, "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["timestamp", "ch1_uv", "ch2_uv", "ch1_filtered", "ch2_filtered"])

# ==== GRAFICADO ====
plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], 'b-', label="CH1 filtrado")
line2, = ax.plot([], [], 'r-', label="CH2 filtrado")
ax.set_xlim(0, 5)
ax.set_ylim(-2000, 2000)
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("µV")
ax.legend()
time_buffer, ch1_buffer, ch2_buffer = [], [], []

print(">> Iniciando adquisición ADS1292R... (Ctrl+C para salir)")

start_time = time.time()

try:
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            print ("no line")
            continue
        try:
            parts = line.split()
            if len(parts) != 9:
                continue
            data_bytes = [int(p, 16) for p in parts]
        except Exception:
            continue

        ch1_uv, ch2_uv = parse_packet(data_bytes)
        if ch1_uv is None:
            continue

        t = time.time() - start_time
        time_buffer.append(t)
        ch1_buffer.append(ch1_uv)
        ch2_buffer.append(ch2_uv)

        # Filtros (últimos 5 s)
        if len(ch1_buffer) > FS:
            ch1_seg = np.array(ch1_buffer[-FS*5:])
            ch2_seg = np.array(ch2_buffer[-FS*5:])
            ch1_f = notch_filter(bandpass_filter(ch1_seg, 0.5, 40, FS), NOTCH_FREQ, FS, Q)[-1]
            ch2_f = notch_filter(bandpass_filter(ch2_seg, 0.5, 40, FS), NOTCH_FREQ, FS, Q)[-1]
        else:
            ch1_f, ch2_f = ch1_uv, ch2_uv

        # Guardar en CSV
        writer.writerow([t, ch1_uv, ch2_uv, ch1_f, ch2_f])

        # Graficar
        if len(time_buffer) > FS*5:
            ax.set_xlim(time_buffer[-FS*5], time_buffer[-1])
            line1.set_xdata(time_buffer[-FS*5:])
            line1.set_ydata(ch1_buffer[-FS*5:])
            line2.set_xdata(time_buffer[-FS*5:])
            line2.set_ydata(ch2_buffer[-FS*5:])
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

except KeyboardInterrupt:
    print("\n>> Adquisición terminada por usuario.")
    csvfile.close()
    ser.close()
    plt.ioff()
    plt.show()

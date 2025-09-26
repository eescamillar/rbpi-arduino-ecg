import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ==============================
# FUNCIONES AUXILIARES
# ==============================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=100):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

# ==============================
# ANALISIS DEL ECG
# ==============================
def analizar_ecg(filename, fs=100):
    print(f"\n?? Analizando archivo: {filename}")

    # Cargar CSV
    df = pd.read_csv(filename)
    df = df.dropna()  # limpiar filas incompletas
    signal = df["ecg_raw"].values
    t = np.arange(len(signal)) / fs

    # Filtrado pasa-banda
    filtered = bandpass_filter(signal, fs=fs)

    # Deteccion de picos R
    peaks, _ = find_peaks(filtered, distance=fs*0.6, height=np.std(filtered))
    rr_intervals = np.diff(peaks) / fs  # en segundos
    bpm_series = 60 / rr_intervals if len(rr_intervals) > 0 else []

    # Metricas
    bpm_avg = np.mean(bpm_series) if len(bpm_series) > 0 else 0
    sdnn = np.std(rr_intervals) * 1000 if len(rr_intervals) > 0 else 0  # en ms

    print(f"?? Frecuencia cardiaca promedio: {bpm_avg:.1f} BPM")
    print(f"?? Intervalo RR promedio: {np.mean(rr_intervals)*1000:.1f} ms")
    print(f"?? Variabilidad (SDNN): {sdnn:.1f} ms")

    # ==============================
    # GRAFICAS
    # ==============================
    plt.figure(figsize=(12, 8))

    # ECG filtrado con picos
    plt.subplot(3, 1, 1)
    plt.plot(t, filtered, label="ECG filtrado")
    plt.plot(peaks/fs, filtered[peaks], "ro", label="Picos R")
    plt.title("ECG filtrado con picos R detectados")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.legend()

    # Intervalos RR
    plt.subplot(3, 1, 2)
    plt.plot(rr_intervals * 1000, marker="o")
    plt.title("Intervalos RR")
    plt.xlabel("Latido")
    plt.ylabel("RR (ms)")

    # Histograma de intervalos RR
    plt.subplot(3, 1, 3)
    plt.hist(rr_intervals * 1000, bins=20, color="g", alpha=0.7)
    plt.title("Distribucion de intervalos RR")
    plt.xlabel("RR (ms)")
    plt.ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    archivo = "ecg_data_XXXXXXXX.csv"  # <-- Cambiar por el nombre real
    analizar_ecg(archivo, fs=100)

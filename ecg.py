import tkinter as tk
from tkinter import ttk, messagebox
import serial
import serial.tools.list_ports
import threading
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import os
from scipy.signal import find_peaks
import tflite_runtime.interpreter as tflite

# ==============================
# CONFIGURACIÓN SERIAL
# ==============================
SERIAL_PORT = "/dev/ttyUSB0"   # Cambiar a /dev/ttyUSB0 en Linux # Cambiar a COM3 en Windows
BAUD_RATE = 9600 # velocidad a la que esta transmitiendo arduino en el puerto, 9600 para AD8232, 115200 para ADS1292R
FS = 200  # Frecuencia de muestreo estimada 200(Hz)
NOTCH_FREQ = 50.0 # Cambiar a 60.0 si en la region la red es de 60 Hz
Q = 30.0  # Calidad notch

# ==============================
# VARIABLES GLOBALES
# ==============================
ser = None
running = False
data_buffer = deque(maxlen=1000)  # últimos 10s si FS=100Hz
ecg_data = []
time_buffer = []
r_peaks = deque(maxlen=10)  # timestamps de últimos picos R


# ==============================
# FUNCIÓN DE LECTURA SERIE
# ==============================
def read_serial():
    global running, ser, ecg_data
    start_time = time.time()
    while running and ser:
        try:
            line = ser.readline().decode(encoding="utf-8", errors="ignore").strip()
            print(f"Dato recibido: {line}")
            if line.isdigit():
                raw_val = int(line)
                timestamp = time.time() - start_time
                data_buffer.append(raw_val)
                time_buffer.append(timestamp)
                ecg_data.append([timestamp, raw_val]) # servira para crear el archivo csv
                # Filtro pasa banda + notch
                #if len(data_buffer) > FS:  # esperar al menos 1 s de datos
                #    segment = np.array(data_buffer[-FS*5:])  # ultimos 5 s
                #    filtered = bandpass_filter(segment, 0.5, 40, FS)
                #    filtered = notch_filter(filtered, NOTCH_FREQ, FS, Q)
                #    y = filtered[-1]
                #else:
                #y = raw_val

        except ser.SerialEXception as e:
            print(f"Error con el puerto serial durante la comunicacion: {e}")
        except KeyboardInterrupt:
            print("Cerrado por el usuario")
        except:
            pass

        # Retraso para no saturar la CPU
        time.sleep(0.1)

# ==============================
# FUNCIONES DE CONTROL
# ==============================
def start_stop():
    global running, ser, thread
    if not running:
        SERIAL_PORT = combo_ports.get()
        if not SERIAL_PORT:
            messagebox.showwarning("Aviso", "Seleccione un puerto primero")
            return
        try:
            # Abrir puerto serie
            print(f"Puerto serial abierto {SERIAL_PORT}")
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            running = True

            # Hilo de lectura
            thread = threading.Thread(target=read_serial, daemon=True)
            thread.start()
            btn_start_stop.config(text="Detener")
            update_plot()

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir {SERIAL_PORT}: {e}")
    else:
        running = False
        if ser and ser.is_open:
            print("puerto serial cerrado")
            ser.close()
        btn_start_stop.config(text="Iniciar")

def exportar_csv():
    if not ecg_data:
        messagebox.showwarning("Aviso","No hay datos para exportar")
        return
    filename = f"ecg_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "ECG_raw"])
        writer.writerows(ecg_data)
    messagebox.showinfo("Exportacion", f"Datos guardados en {filename}")

def salir():
    global running, ser
    running = False
    if ecg_data:
        exportar_csv() # Generar el archivo CSV
    if ser and ser.is_open:
        ser.close()
    root.destroy()

# ==============================
# DETECCIÓN DE PICOS R Y BPM
# ==============================
def calcular_bpm(signal):

    if len(signal) < FS * 3:  # al menos 3s de datos
        return None

    # Normalizacion y deteccion de picos
    signal = np.array(signal)
    signal = signal - np.mean(signal)

    peaks, _ = find_peaks(signal, distance=FS*0.6, height=np.std(signal))
    
    # Calcular BPM si hay al menos 1 pico
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / FS # en segundos
        bpm = 60 / np.mean(rr_intervals)
        return round(bpm, 1)

    return None

# ==============================
# GRAFICADO EN TIEMPO REAL
# ==============================
def update_plot():
    if running:
        ax.clear()
        signal = list(data_buffer)        

        #Graficar los ultimos 5 segundos (maximo 1000 muestras)
        if signal:
            t = np.linspace(-len(signal)/FS, 0, len(signal))  # tiempo relativo en segundos
            ax.plot(t, signal, label="ECG")

            ax.set_title("ECG AD8232 (ultimos 5 seg)")
            ax.set_xlim(-5, 0)
            ax.set_ylim(0, 1023)  # ajustar según amplitud

            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.legend(loc="upper right")

            # Ajuste dinamico del eje Y
            ymin = min(signal) - 50
            ymax = max(signal) + 50
            ax.set_ylim(ymin, ymax)
        
            # Calcular BPM
            bpm = calcular_bpm(signal)
            actualizar_bpm(bpm)
            
        canvas.draw()
        root.after(200, update_plot) # vuelve a llamar update_plot
           

# ==============================
# FILTROS DIGITALES
# ==============================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# pasa banda 0.5-40 Hz
def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=FS):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

# notch 50/60 Hz
def notch_filter(data, freq=NOTCH_FREQ, fs=FS, Q=Q):
    b, a = iirnotch(freq, Q, fs)
    return lfilter(b, a, data)

# ==============================
# DETECTAR PUERTOS USB DISPONIBLES
# ==============================
def listar_puertos():
    ports = [port.device for port in serial.tools.list_ports.comports()]
    combo_ports["values"] = ports
    if ports:
        combo_ports.current(0)

# ==============================
# TENSORFLOW
# ==============================
def crear_tensor():
    tensor = torch.tensor(ecg_data)

def cargar_dataset():
    ds = torch.utils.data.Dataset()
    dl = torch.utils.data.Dataloader()

# ==============================
# ACTUALIZAR COLOR DEL BPM
# ==============================
def actualizar_bpm(bpm):
    if bpm is None:
        bpm_value.set("BPM: ---")
        label_bpm.config(fg="black")
    else:
        bpm_value.set(f"BPM: {bpm}")
        if bpm < 60:
            label_bpm.config(fg="#2196F3") # azul
        elif 60 <= bpm <= 100:
            label_bpm.config(fg="#4CAF50") # verde
        elif 101 <= bpm <= 120:
            label_bpm.config(fg="#FF9800") # naranja
        else:
            label_bpm.config(fg="#f44336") # rojo

# ==============================
# INTERFAZ TKINTER
# ==============================
root = tk.Tk()
root.title("Monitor ECG AD8232")
root.configure(bg="#f0f4f7") # gris claro
root.geometry("800x600")

# Fuente global
default_font = ("Arial",12)

# Frame principal
frame_buttons = tk.Frame(root, bg="#f0f4f7")
frame_buttons.pack(fill="both", expand=True, padx=20, pady=20)

# ==============================
# WIDGETS EN GRID 3x3
# ==============================

# --- Fila 0 ---
tk.Label(frame_buttons, text="Puerto:", font=default_font, bg="#f0f4f7").grid(row=0, column=0, padx=5, pady=5, sticky="e")

combo_ports = ttk.Combobox(frame_buttons, width=15, font=default_font)
combo_ports.grid(row=0, column=1, padx=5, pady=5)

btn_refresh = tk.Button(frame_buttons, text="Actualizar", width=20, bg="#cccccc", fg="black", font=default_font, command=listar_puertos)
btn_refresh.grid(row=0, column=2, padx=5, pady=5)

# --- Fila 1 ---
btn_start_stop = tk.Button(frame_buttons, text="Iniciar", width=20, bg="#4CAF50", fg="white", font=default_font, command=start_stop) # verde
btn_start_stop.grid(row=1, column=0, padx=5, pady=5)

btn_export = tk.Button(frame_buttons, text="Exportar CSV", width=20, bg="#2196F3", fg="white", font=default_font, command=exportar_csv) # azul
btn_export.grid(row=1, column=1, padx=5, pady=5)

btn_salir = tk.Button(frame_buttons, text="Salir", width=20, bg="#f44336", fg="white", font=default_font, command=salir) # rojo
btn_salir.grid(row=1, column=2, padx=5, pady=5)

# --- fila2 ---
# Label para BPM
bpm_value = tk.StringVar(root, value="BPM: ---")
label_bpm = tk.Label(frame_buttons, textvariable=bpm_value, font=("Arial", 18, "bold"), bg="#f0f4f7", fg="black")
label_bpm.grid(row=2, column=0, columnspan=3, pady=15)

# --- fila 3 (canvas)---
fig, ax = plt.subplots(figsize=(6,3))
fig.patch.set_facecolor("#ffffff") # fondo blanco

# Integrar matplotlib en tkinter
canvas = FigureCanvasTkAgg(fig, master=frame_buttons)
canvas.get_tk_widget().grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

#root.protocol("WM_DELETE_WINDOW", salir)

# Expansion de columnas y filas
for col in range(3):
    frame_buttons.grid_columnconfigure(col, weight=1)
frame_buttons.grid_rowconfigure(3, weight=1)

# Cargar lista de puertos
listar_puertos()

root.mainloop()

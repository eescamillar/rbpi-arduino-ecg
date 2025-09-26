import serial
import time

# --- Configuracion ---
# reemplazar puerto, segun si es linux o windows "COM3"
port_arduino = "/dev/ttyUSB0"
# la velocidad debe coincidir con la configuracionen arduino
velocidad_baud = 9600

try:
    # abre la conexion serial
    ser = serial.Serial(port_arduino, velocidad_baud, timeout=1)
    print(f"Conexion abierta {port_arduino}")
    time.sleep(2) # Espera a que la conexion se establezca

    while True:
        # Leer datos linea por linea (hasta el salto de linea 'n')
        if ser.in_waiting > 0:
            linea_recibida = ser.readline().decode("utf-8").strip()
            if linea_recibida: #Asegura qie no sea una linea vacia
                print(f"Dato recibido: {linea_recibida}")
                
        # un pequeo retraso para no saturar la CPU
        time.sleep(0.1)
                

except serial.SerialException as ex:
    print(f"Error al abrir el puerto serial o durante la comunicacion: {ex}")

except KeyboardInterrupt:
    print("Comunicacion cerrada por el usuario")

finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Puerto serial cerrado")
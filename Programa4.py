import face_recognition
import cv2
# comenzamos la captura de video
cap = cv2.VideoCapture(0)
# Cargamos la imagen en un contenedor y lo convertimos
mi_img = face_recognition.load_image_file("yo.jpg")
mi_img_cod = face_recognition.face_encodings(mi_img)[0]
# Cargamos una 2da imagen y la convertimos en array
segunda_img = face_recognition.load_image_file("caramodel.jpg")
segunda_img_cod = face_recognition.face_encodings(segunda_img)[0]
# Inicializamos los contenedores para comparacion
localizacion_cara = []
contenedor_codificadas = []
nombre_de_cara = []
bandera = True
# Creamos un arreglo con las fotos conocidas
caras_codificadas_conocidas = [
    mi_img_cod,
    segunda_img_cod,
]
nombres_conocidos = [
    "Zedrick",
    "Novia de Zedrick"
]
while True:
    ret, frame = cap.read()
    # Se escala la imagen para un reconocimiento mas rapido
    frame_escalado = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
    # Conversion de BRG(cv2) a RGB(face_recognition)
    frame_escalado_rgb = frame_escalado[:, :, ::-1]

    # Solo procesa los demas cuadros para ahorrar tiempo
    if bandera:
        # Encuentra todas las caras en el frame en tiempo real
        localizacion_cara = face_recognition.face_locations(frame_escalado_rgb)
        contenedor_codificadas = face_recognition.face_encodings(
            frame_escalado_rgb, localizacion_cara)

        nombre_de_cara = []
        for cara_codificada in contenedor_codificadas:
            # Hace la comparacion para saber si hace match o no
            matches = face_recognition.compare_faces(
                caras_codificadas_conocidas, cara_codificada)
            nombre = "Unknown"

            # Si existe match entonces solo se le asigna el nombre 1 vez.
            if True in matches:
                first_match_index = matches.index(True)
                nombre = nombres_conocidos[first_match_index]

            nombre_de_cara.append(nombre)

    bandera = not bandera

    # Mostramos los resultados
    for (arriba, derecha, abajo, izquierda), nombre in zip(localizacion_cara,
                                                           nombre_de_cara):
        # Se escala la localizacion del rostro a 1/5 parte
        arriba *= 5
        derecha *= 5
        abajo *= 5
        izquierda *= 5
        # Dibujamos un rectangulo al rededor de la cara para identificar
        cv2.rectangle(frame, (izquierda, arriba), (derecha, abajo), (0, 255, 0), 2)
        # Se dibuja el nombre debajo del cuadro
        cv2.rectangle(frame, (izquierda, abajo - 35),
                      (derecha, abajo), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, nombre, (izquierda + 6, abajo - 6),
                    font, 1.0, (255, 255, 255), 1)
    cv2.imshow('face_recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
cap.release()
cv2.destroyWindow()


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
    cv2.imshow('face_recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
cap.release()
cv2.destroyWindow()

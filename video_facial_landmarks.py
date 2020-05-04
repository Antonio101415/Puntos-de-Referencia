# Heimdall-EYE USO:
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# Importamos los paquetes y librerias necesarios
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
 
# Contruimos el argumento para analizar los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
 
# Inicializamos el detector de la cara de dlib (Basado en HOG)
# Luego creamos el predictor facial
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Inicializamos el sernsor de la camara para que se active 
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# Recorre los fotogramas de la trasmision de video
while True:
	# Agarramos el fotograma de la secuencia de video y lo redimensionamos
	# A un ancho maximo de 400 pixeles y lo convertimos en una escala de grises
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detectamos rostros en una escala de grises
	rects = detector(gray, 0)

	# Bucle para la deteccion de rostros
	for rect in rects:
		# Determinamos los puntos de referencia faciales para la region de la cara
		# Luego convertimos el hito facial (x,y)  y coordinamos a Numpy
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# Bucle sobre las coordenadas (x,y) para los puntos de referencia faciales
		# Y lo dibujamos en el frame de la pantalla
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	  
	# Mostramos el frame en la pantalla 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# Si quieres salir de el frame pulsa q
	if key == ord("q"):
		break
 
# Realizamos la limpieza de los paquetes utilizados
cv2.destroyAllWindows()
vs.stop()

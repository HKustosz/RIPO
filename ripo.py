import cv2
import numpy as np

# Wczytanie obrazu
# image = cv2.imread('2.png')
cap = cv2.VideoCapture('111.mp4')
ret, frame = cap.read()

while True:
    # Konwersja do przestrzeni kolorów HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Określenie zakresu koloru niebieskiego
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Binaryzacja obrazu
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Wykrycie konturów
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iteracja przez wszystkie kontury i rysowanie prostokątów
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if h > 200:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, "Pole odkładcze", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Wyświetlenie wynikowego obrazu
    cv2.imshow("Wynik", frame)
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie obiektu VideoCapture i zamknięcie okna wideo
cap.release()
cv2.destroyAllWindows()
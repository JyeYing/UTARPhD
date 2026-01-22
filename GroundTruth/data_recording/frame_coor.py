import cv2

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(x, y)

img = cv2.imread("frame46_c.jpg")
cv2.imshow("frame", img)
cv2.setMouseCallback("frame", click)
cv2.waitKey(0)
cv2.destroyAllWindows()
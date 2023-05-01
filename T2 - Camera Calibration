import numpy as np
import cv2 as cv
import glob

chessBoardSize = (24,17)
frameSize = (1155,1079)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessBoardSize[0] * chessBoardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessBoardSize[0], 0:chessBoardSize[1]].T.reshape(-1,2)

objPoints = [] # Pontos 3D no espaço do mundo real
imgPoints = [] # Pontos 2D no plano da imagem

images = glob.glob('*.png')

for image in images:
    print(image)
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # encontrar os cantos do tabuleiro de xadrez
    ret, corners = cv.findChessboardCorners(gray, chessBoardSize, None)

    # Uma vez encontrado, adicionar pontos do objeto, pontos da imagem (após refiná-los)
    if ret == True:
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners)

        # desenhar e exibir os cantos
        cv.drawChessboardCorners(img, chessBoardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()


## Calibração 

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

print("Câmera Calibrada: ", ret)
print("\nMatriz da câmera:\n", cameraMatrix)
print("\nParâmetros de distorção:\n", dist)
print("\nVetores de Tradução:\n", tvecs)

## Sem Distorção

img = cv.imread("cali5.png")
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# Processo Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
# cortar a imagem
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)

# Remapeamento com Undistort
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# cortar a imagem
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('CaliResult2.png', dst)

# Erro de reprojeção
mean_error = 0

for i in range(len(objPoints)):
    imgPoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2)/len(imgPoints)
    mean_error =+ error

print("\nErro Total: {}".format(mean_error/len(objPoints)))
print("\n\n\n")

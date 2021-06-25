import cv2

def onMouse(event, x, y, flag, param):
    global img_ori
    img = img_ori.copy()
    B, G, R = cv2.split(img)
    B = B[y,x]
    G = G[y,x]
    R = R[y,x]
    
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    H = H[y,x]
    S = S[y,x]
    V = V[y,x]

    position_str = 'x, y = (' + str(x) + ', ' + str(y) +  ')'
    bgr_str = 'B=' + str(B) + ' G=' + str(G) + ' R=' + str(R)
    hsv_str = 'H=' + str(H) + ' S=' + str(S) + ' V=' + str(V)
    font = cv2.FONT_HERSHEY_SIMPLEX
    print(position_str)
    cv2.putText(img, position_str, (10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(img, bgr_str, (10, 60), font, 0.7, (0, 255, 0), 2)
    cv2.putText(img, hsv_str, (10, 90), font, 0.7, (0, 255, 0), 2)
    cv2.imshow("color", img)

img_ori = cv2.imread("/home/jwwangchn/Documents/Nutstore/100-Work/110-Projects/2020-BS/01-CVPR/01-绘图/01-不同时间拍摄的图像/ms_L18_106968_219488.jpg")
print(img_ori.max())
cv2.namedWindow("color", cv2.WINDOW_NORMAL)
cv2.resizeWindow("color", 800, 800)
cv2.imshow("color", img_ori)
cv2.setMouseCallback("color", onMouse, 0)

while(1):
    key = cv2.waitKey(1)
    if key == ord(' '):
        break
cv2.destroyAllWindows()
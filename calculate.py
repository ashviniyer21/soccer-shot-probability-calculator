import cv2 as cv
import sys
import torch
import ssl
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
img_name = "shot.png"
img = cv.imread(img_name)
results = model(img_name)
idx = 0
prompts = ['Click on Bottom Left Goal', 'Click on Top Left Goal', 'Click on Top Right Goal', 'Click on Bottom Right Goal', 'Click on the ball', 'Click on the penalty spot', 'Probability: ']
coordinates = []
yaw = 0
pitch = 0
flip = False
width = img.shape[1]
center_x = 0
center_y = 0

finished = False

def coordinate_transform(x, y):
    cosa = np.cos(yaw)
    sina = np.sin(yaw)
    cosb = np.cos(-pitch)
    sinb = np.sin(-pitch)
    
    if flip:
        x = width - x
    points = [x - center_x, y - center_y, 0]
    px = points[0]
    py = points[1]
    pz = points[2]
    points[0] = cosa * cosb * px - sina * py + cosa * sinb * pz
    points[1] = sina * cosb * px + cosa * py + sina * sinb * pz
    points[2] = -sinb * px + cosb * pz

    return points

def get_length(x0, y0, z0, x1, y1, z1):
    return np.sqrt((x1-x0) * (x1-x0) + (y1-y0) * (y1-y0) + (z1-z0) * (z1-z0))

def get_angle(p1, p2, p3):
    v1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
    v2 = [p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2]]
    length = get_length(v1[0], v1[1], v1[2], 0, 0, 0) * get_length(0, 0, 0, v2[0], v2[1], v2[2])
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    print
    return abs(np.arccos(dot / length))

def calculate_prob():
    global pitch
    global yaw
    global flip
    global center_x
    global center_y
    if (coordinates[4][0] < coordinates[0][0]):
        flip = True
    #Getting player locations
    table = results.pandas().xyxy[0]
    player_coordinates = []
    for i in range(len(table)):
        row = table.loc[i]
        try:
            if(row["name"] == "person"): 
                temp = []
                temp.append(row.xmin)
                temp.append(row.ymin)
                temp.append(row.xmax)
                temp.append(row.ymax)
                player_coordinates.append(temp)
        except:
            pass
    print(player_coordinates)

    #Coordinate Transformation
    m1 = (coordinates[1][1] - coordinates[0][1]) / (coordinates[1][0] - coordinates[0][0])
    m2 = (coordinates[2][1] - coordinates[1][1]) / (coordinates[2][0] - coordinates[1][0])
    yaw = np.arctan2(m2 - m1, m1 + m2 + 1)
    if yaw < 0:
        yaw *= -1
    if yaw > np.pi/2:
        yaw = np.pi - yaw
    center_x = coordinates[0][0]
    center_y = coordinates[0][1]
    tempp0 = coordinate_transform(coordinates[0][0], coordinates[0][1])
    tempp1 = coordinate_transform(coordinates[1][0], coordinates[1][1])
    tempp2 = coordinate_transform(coordinates[2][0], coordinates[2][1])
    l1 = get_length(tempp0[0], tempp0[1], tempp0[2], tempp1[0], tempp1[1], tempp1[2])
    l2 = get_length(tempp2[0], tempp2[1], tempp2[2], tempp1[0], tempp1[1], tempp1[2])
    pitch = l1 / (l1 + l2) * np.pi / 2
    print("Pitch: " + str(pitch))
    print("Yaw : " + str(yaw))

    for i in range(len(coordinates)):
        coordinates[i] = coordinate_transform(coordinates[i][0], coordinates[i][1])
    for i in range(len(player_coordinates)):
        player_coordinates[i] = coordinate_transform(player_coordinates[i][0], player_coordinates[i][1])
    print(coordinates)
    print(player_coordinates)

    #Calculating Probability
    base_hor_angle = 0.6435
    hor_angle = get_angle(coordinates[0], coordinates[5], coordinates[3])
    prob = hor_angle / base_hor_angle
    base_ver_angle = 0.2187
    ver_angle = get_angle(coordinates[0], coordinates[5], coordinates[1])
    prob *= ver_angle / base_ver_angle

    hor_angle = get_angle(coordinates[0], coordinates[4], coordinates[3])
    ver_angle = get_angle(coordinates[0], coordinates[4], coordinates[1])
    prob *= hor_angle / base_hor_angle
    prob *= ver_angle / base_ver_angle
    
    prob = round(prob, 2)
    prob = min(prob, 0.99)
    prob = max(prob, 0.01)
    return prob

def click_event(event,x,y,flags,param):
    global idx
    global finished
    if event == cv.EVENT_LBUTTONDOWN:
        if idx < len(prompts) - 1:
            coordinates.append((x, y))
            print(x, ' ', y)
            cv.circle(img,(x,y),10,(255,0,0), 3)
            idx += 1
        elif not finished:
            prompts[6] = "Probability: " + str(calculate_prob())
            finished = True
        img_text = img.copy()
        cv.putText(img_text, prompts[idx], (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow("image", img_text)
        


if __name__ == "__main__":
    if img is None:
        sys.exit("Could not read the image.")
    img_text = img.copy()
    cv.putText(img_text, prompts[idx], (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.imshow("image", img_text)
    cv.setMouseCallback('image', click_event)
    k = cv.waitKey(0)
    print(coordinates)
    cv.destroyAllWindows()

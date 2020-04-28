import cv2
import numpy as np

import glob

from imagedescriptor import ImageDescriptor
from imagematcher import ImageMatcher

useSIFT = False#args["sift"] > 0
useHamming = False#args["sift"] == 0
ratio = 1
minMatches = 10

if useSIFT:
    minMatches = 50

imd = ImageDescriptor(useSIFT = useSIFT)
imv = ImageMatcher(imd, glob.glob("images/*.png"), ratio = ratio, minMatches = minMatches, useHamming = useHamming)

# board = cv2.imread("./images/picture.jpeg")
# cv2.imshow("Picture", board)
# cv2.waitKey(0)

filter = False


# file_path = './images/picture.jpeg'
# file_path = './images/picture2.jpeg'
file_path = './images/picture4.jpeg'
img = cv2.imread(file_path)
img_copy = img.copy()

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150)



# kernel = np.ones((5,5),np.uint8)
# edges = cv2.dilate(edges,kernel,iterations = 1)
# kernel = np.ones((7,7),np.uint8)
# edges = cv2.erode(edges,kernel,iterations = 1)



def euclidean_distance(a, b, axis=None):
    """Calculate the euclidean distance between two vectors.
    :param a Vector 1
    :param b Vector 2
    :returns The euclidean distance between the two vectors.
    """
    return np.linalg.norm(np.array(a) - np.array(b), axis=axis)

def line_ABC(p1, p2):
    """Get a line in ABC from from two points"""
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C

def intersection_xyxy(line1, line2):
    """Get the intersection point of two lines that are in [(xy), (xy)] form"""
    L1 = line_ABC(*line1)
    L2 = line_ABC(*line2)
    return intersection(L1, L2)

def intersection(L1, L2):
    """Get the intersection point of two lines that are in ABC form"""
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False
def putX(img, center, radius, thickness=1):
    if img is None: return
    if thickness < 0: return

    cx, cy = tuple([int(round(c)) for c in center])
    color = (0, 255, 0)
    radius = int(round(radius))

    cv2.line(img, (cx - radius, cy - radius), (cx + radius, cy + radius), color, thickness=thickness)
    cv2.line(img, (cx - radius, cy + radius), (cx + radius, cy - radius), color, thickness=thickness)


blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# cv2.imwrite('mask.jpg',thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
c = 0
for i in contours:
    area = cv2.contourArea(i)
    if area > max_area:
        max_area = area
        best_cnt = i
    c+=1

mask = np.zeros((gray.shape),np.uint8)
cv2.drawContours(mask,[best_cnt],0,255,-1)
# cv2.drawContours(mask,[best_cnt],0,0,2)
cv2.imwrite('mask.jpg',mask)


out = np.zeros_like(gray)
out[mask == 255] = gray[mask == 255]

edges = cv2.Canny(out, 50,150)
cv2.imwrite('canny.jpg',edges)

lines = cv2.HoughLines(edges,1,np.pi/180,150)

new_lines = []
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    new_lines.append([(x1, y1), (x2, y2)])



# for line in new_lines:
#     for x in new_lines:
#         print((euclidean_distance(line[0], x[0]) + euclidean_distance(line[1], x[1])) / 2 )



filtered_lines = []
for line in new_lines:
    close_lines = [x for x in new_lines if (euclidean_distance(line[0], x[0]) + euclidean_distance(line[1], x[1])) / 2 < 25 and x != line]
    if close_lines and all(l not in filtered_lines for l in close_lines):
        # print("APEEND", line)
        filtered_lines.append(line)
    
    if not close_lines:
        filtered_lines.append(line)


intersection_list = []

for line in filtered_lines:
    # print(line)

    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[1][0]
    y2 = line[1][1]

    for _line in filtered_lines:
        inter = intersection_xyxy(line, _line)
        if not inter:
            continue

        if not intersection_list and len(intersection_list) == 0:
            if inter[0] <= img_copy.shape[1] and inter[0] >= 0 and inter[1] <= img_copy.shape[0] and inter[1] >= 0:
                intersection_list.append(inter)

        if all(euclidean_distance(inter, x) > 10 for x in intersection_list):
            if inter in intersection_list:
                continue
            if inter[0] <= img_copy.shape[1] and inter[0] >= 0 and inter[1] <= img_copy.shape[0] and inter[1] >= 0:
                intersection_list.append(inter)

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

for i in intersection_list:
    putX(img, i, 5, 2) 

cv2.imwrite('hough.jpg',img)


ideal_points = []

ideal_board = np.zeros((1200, 1200),np.uint8)
for x in range(0, 1200, 100):
    cv2.line(ideal_board, (0, x), (1200, x), (255, 255, 255))
    cv2.line(ideal_board, (x, 0), (x, 1200), (255, 255, 255))


sorted_list = sorted(intersection_list, key=lambda x: (x[0] + x[1]) / 2)

top_right = sorted(intersection_list, key=lambda x: x[1])[0]
if top_right[0] <= sorted_list[0][0]:
    top_right = sorted(intersection_list, key=lambda x: x[0])[-1]

bottom_left = sorted(intersection_list, key=lambda x: x[0])[0]
if bottom_left[0] >= sorted_list[0][0]:
    bottom_left = sorted(intersection_list, key=lambda x: x[1])[-1]    

intersection_four = [sorted_list[0], top_right, bottom_left, sorted_list[-1]]
ideal_four = [(0, 0), (1200, 0), (0, 1200), (1200, 1200)]

# print(intersection_four)
# print(ideal_four)

h, status = cv2.findHomography(np.array(intersection_four), np.array(ideal_four))
im_dst = cv2.warpPerspective(img_copy, h, (1200, 1200))

image = im_dst

gray = cv2.cvtColor(im_dst, cv2.COLOR_BGR2GRAY)

crop_img = None
results = []
for x in range(0, 1200, 100):
    for y in range(0, 1200, 100):
        crop_img = gray[y:y+100, x:x+100]
        (queryKps, queryDescs) = imd.describe(crop_img)
        results = imv.search(queryKps, queryDescs, .5)

cv2.imwrite('cropped.jpg', crop_img)

print("RESULTS", results)

test = cv2.imread(results[0][1])

h,w, _ = test.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,results[0][0][1])
img2 = cv2.polylines(gray,[np.int32(dst)],True,255,5, cv2.LINE_AA)

# for r in results:
#     for point in r[0][1]:
#         putX(gray, point, 5, 6)

cv2.imwrite('board_bw.jpg', gray)






brightness = np.mean(image)

_, image_b = cv2.threshold(gray, brightness * .9, 255, cv2.THRESH_BINARY)

# cv2.imwrite('board_bw.jpg', image_b)

square_points = []
for y in range(0, 1200, 100):
    for x in range(0, 1200, 100):
        crop_img = image_b[y:y+100, x:x+100].copy()
        color = 'black'
        if cv2.countNonZero(crop_img) > (100*100) / 2:
            color = 'white'

        square_points.append({"point": (x, y), "color": color})

for i, p in enumerate(square_points):
    image = cv2.putText(image, str(i+1), (p['point'][0] + 25, p['point'][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    if p['color'] == 'white':
        color = (0, 0, 0)
    else:
        color = (255, 255, 255)
    image = cv2.putText(image, p['color'], (p['point'][0] + 25, p['point'][1] + 75), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=2)

cv2.imwrite('board.jpg',image)











# lines = cv2.HoughLines(edges,1,np.pi/180,150)

# if not lines.any():
#     print('No lines were found')
#     exit()

# if filter:
#     rho_threshold = 15
#     theta_threshold = 0.1

#     # how many lines are similar to a given one
#     similar_lines = {i : [] for i in range(len(lines))}
#     for i in range(len(lines)):
#         for j in range(len(lines)):
#             if i == j:
#                 continue

#             rho_i,theta_i = lines[i][0]
#             rho_j,theta_j = lines[j][0]
#             if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
#                 similar_lines[i].append(j)

#     # ordering the INDECES of the lines by how many are similar to them
#     indices = [i for i in range(len(lines))]
#     indices.sort(key=lambda x : len(similar_lines[x]))

#     # line flags is the base for the filtering
#     line_flags = len(lines)*[True]
#     for i in range(len(lines) - 1):
#         if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
#             continue

#         for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
#             if not line_flags[indices[j]]: # and only if we have not disregarded them already
#                 continue

#             rho_i,theta_i = lines[indices[i]][0]
#             rho_j,theta_j = lines[indices[j]][0]
#             if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
#                 line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

# print('number of Hough lines:', len(lines))

# filtered_lines = []

# if filter:
#     for i in range(len(lines)): # filtering
#         if line_flags[i]:
#             filtered_lines.append(lines[i])

#     print('Number of filtered lines:', len(filtered_lines))
# else:
#     filtered_lines = lines

# for line in filtered_lines:
#     rho,theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imwrite('hough.jpg',img)
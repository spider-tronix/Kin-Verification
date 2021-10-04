import cv2 
import numpy as np


protoFile = "/mnt/sda2/Spider/Kin-Verification/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "/mnt/sda2/Spider/Kin-Verification/pose_iter_160000.caffemodel"
# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

frame = cv2.imread("/mnt/sda2/Spider/Kin-Verification/test6.jpg")

frameWidth=frame.shape[1]
frameHeight=frame.shape[0]
print(type(frame))
# Specify the input frame dimensions
inWidth = 368
inHeight = 368

PARTS = { "RightAnkle": 0, "RightKnee": 1, "RightHip": 2, "LeftHip": 3, "LeftKnee": 4,
             "LeftAnkle": 5, "Pelvis": 6, "Thorax": 7, "upper neck": 8, "Head": 9,
             "Rightwrist": 10, "Rightelbow": 11, "Rightshoulder": 12, "Leftshoulder": 13, "Leftelbow": 14,
             "Leftwrist": 15}
#0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
PAIRS = [ ]




def gait_feature(image,points):
    features=[]
    pelvis=6
    for i in range(16):
        if i!=pelvis:
            features.append(calculate_distance(points[pelvis],points[i]))
    return np.array(features)




def calculate_distance(point1,point2):
    point1=np.array(point1)
    point2=np.array(point2)
    return np.sqrt(np.sum(np.square(point1-point2)))
  
# Prepare the frame to be fed to the network
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
  
# Set the prepared object as the input blob of the network
net.setInput(inpBlob)
output = net.forward()

H = output.shape[2]
W = output.shape[3]
# Empty list to store the detected keypoints
points = []
for i in range(16):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H
  
    if prob > 0:
        cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255),
                   thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame, "{}".format(i), (int(x), int(
            y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
  
        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)
    print(points)

print(len(points))
#for pair in PAIRS:
#        partF = pair[0]
#        partT = pair[1]
#        assert(partF in PARTS)
#        assert(partT in PARTS)
#
#        idF = PARTS[partF]
#        idT = PARTS[partT]
#        print(idF,idT)
#        if points[idF] and points[idT]:
#            cv2.line(frame, points[idF], points[idT], (0, 255, 0), 3)
#            cv2.ellipse(frame, points[idF], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
#            cv2.ellipse(frame, points[idT], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

  
cv2.imshow("Output-Keypoints", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
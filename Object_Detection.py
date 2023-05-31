import cv2
import numpy as np
Coco = "./YoLo/coco.txt"
Yolocfg = "./YoLo/yolov-tiny.cfg"
Yoloweights = "./YoLo/yolov-tiny.weights"
whT = 320
ObjectNames = []
with open(Coco, "r") as f:
    ObjectNames = f.read().rstrip("\n").split("\n")


def Load(cfg, weight):
    net = cv2.dnn.readNetFromDarknet(cfg, weight)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


Net = Load(Yolocfg, Yoloweights)
cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    blob = cv2.dnn.blobFromImage(
        img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    Net.setInput(blob)
    layerNames = Net.getLayerNames()
    outputnames = [layerNames[i-1]for i in Net.getUnconnectedOutLayers()]
    outputs = Net.forward(outputnames)

    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            score = det[5:]
            classid = np.argmax(score)
            config = score[classid]
            if config > 0.5:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classid)
                confs.append(config)

    ind = cv2.dnn.NMSBoxes(bbox, confs, 0.5, 0.3)
    for i in ind:
        box = bbox[i]
        x, y, w, h = box
        # parameters=> image,point 1, point 2, color(rgb),thickness of line
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f"{ObjectNames[classIds[i]].upper()} {int(confs[i]*100)}%",
                    (x, y-13), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255))  # parameters => image, text, position (in x and y), font family, text thickness, text color

    cv2.imshow("Object Detection", img)
    cv2.waitKey(1)

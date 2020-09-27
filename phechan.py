# import necessary packages
import cv2
import numpy as np

def load_model(config_file, weights_file, label):
    """
    Takes model configuration file, weights file and label file(if available) and loads the model.
    :param config_file: model configuration file (.pbtxt file)
    :param weights_file: model weights file (.pb file)
    :param label: file containing the class names on which the model was trained, if available
    :returns classnames as list and model
    """
    config_file = config_file
    weights_file = weights_file
    label = label

    # get the class names
    # Note: ssd_mobilenet_v3 model that we have is trained on coco dataset and the class names are in the file so let's fetch it
    # classnames_file_path = 'model/pretrained_files/ssd_mobilenet_v3_coco_14-jan-2020/coco.names'
    classnames = []
    with open(label, 'rt') as f:
        classnames = f.read().rstrip('\n').split('\n')
    # print(classnames)

    # load the model
    net = cv2.dnn_DetectionModel(config_file, weights_file)
    net.setInputSize(320, 320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    return classnames, net

def detect(data_type, source_loc, model, label, thres=0.45, nms_thres=0.2):
    """
    Takes in Image, Video or Camera feed and detects objects

    :param data_type: takes values 'img' for image file, 'vid' for video file or 'cam' for camera feed depending on the type of data being fed.
    :param source_loc: path of image/video file or input camera number(use 0 for default)
    :param thres: confidence threshold to detect object (by default it's 0.45)
    :param nms_thres: Non-Maximum Threshold value (by default it's 0.2)
    """
    net = model
    classnames = label

    if data_type == 'img':
        # read the image
        img = cv2.imread(source_loc)

        # detect using the loaded model
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        # print(classIds, bbox)



        # draw a rectangle around each object and label the detected class
        for classid, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0,0,255), thickness=2)
            cv2.putText(img, classnames[classid - 1].upper(), (box[0] + 10, box[1] + 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(0, 0, 225), thickness=2)
            cv2.putText(img, str(round(confidence * 100, 2)) + '%', (box[0] + 150, box[1] + 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(0, 0, 225), thickness=2)

        # display the image in a new window
        cv2.imshow("Image", img)
        cv2.waitKey(0)

    elif data_type == 'cam':
        cap = cv2.VideoCapture(source_loc)
        cap.set(3, 640)
        cap.set(4, 480)

        while True:
            success, img = cap.read()

            # detect using the loaded model
            classIds, confs, bbox = net.detect(img, confThreshold=thres)
            # print(classIds, bbox)
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1, -1)[0])
            confs = list(map(float, confs))

            indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_thres)
            # print(indices)

            for i in indices:
                i = i[0]
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
                cv2.putText(img, classnames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # if (len(classIds) != 0):
            #     # draw a rectangle around each object and label the detected class
            #     for classid, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            #         cv2.rectangle(img, box, color=(0, 0, 255), thickness=2)
            #         cv2.putText(img, classnames[classid - 1].upper(), (box[0] + 10, box[1] + 30),
            #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
            #                     color=(0, 0, 225), thickness=2)
            #         cv2.putText(img, str(round(confidence*100,2)) + '%', (box[0] + 150, box[1] + 30),
            #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
            #                     color=(0, 0, 225), thickness=2)

            # display the image in a new window
            cv2.imshow("Camera", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    config_file = 'model/pretrained_files/ssd_mobilenet_v3_coco_14-jan-2020/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weights_file = 'model/pretrained_files/ssd_mobilenet_v3_coco_14-jan-2020/frozen_inference_graph.pb'
    label = 'model/pretrained_files/ssd_mobilenet_v3_coco_14-jan-2020/coco.names'

    classnames, net = load_model(config_file, weights_file, label)
    # detect('img', 'data/lena.png', net, classnames) # for image
    detect('cam', 0, net, classnames) # for cam
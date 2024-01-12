import time
import cv2

COSINE_THRESHOLD = 0.4

model_file = "D:\\TEDFaceRecognition\\data\\models\\face_detection_yunet_2023mar.onnx"
config_file = "D:\\TEDFaceRecognition\\data\\models\\face_recognizer_fast.onnx"


net = cv2.dnn.readNet(model_file, config_file)

# Set the backend and target to use CUDA
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def match(recognizer, feature1, dictionary):
    max_score = 0.0
    sim_user_id = ""
    for user_id, feature2 in zip(dictionary.keys(), dictionary.values()):
        score = recognizer.match(
            feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        if score >= max_score:
            max_score = score
            sim_user_id = user_id
    if max_score < COSINE_THRESHOLD:
        return False, ("", 0.0)
    return True, (sim_user_id, max_score)


def recognize_face(image, face_detector, face_recognizer, file_name=None):
    if image is None:
        #print("Görüntü boş (None) olduğu için işlenemiyor.")
        return None, None
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.shape[0] > 1000:
        image = cv2.resize(image, (0, 0),
                           fx=500 / image.shape[0], fy=500 / image.shape[0])

    height, width, _ = image.shape
    face_detector.setInputSize((width, height))
    try:
        dts = time.time()
        _, faces = face_detector.detect(image)
        if file_name is not None:
            if faces is None or len(faces) == 0:
                #print(f"No face detected in the file: {file_name}")
                return None, None

        faces = faces if faces is not None else []
        features = []
        # print(f'time detection  = {time.time() - dts}')
        for face in faces:
            rts = time.time()

            aligned_face = face_recognizer.alignCrop(image, face)
            feat = face_recognizer.feature(aligned_face)
            # print(f'time recognition  = {time.time() - rts}')

            features.append(feat)
        return features, faces
    except Exception as e:
        print(e)
        print(file_name)
        return None, None

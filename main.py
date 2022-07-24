from landmark_detection import MarkDetector, FaceDetector
import cv2
import numpy as np
import math

mark_detector = MarkDetector()
cap = cv2.VideoCapture(0)
_, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX 

#Trained ideal 3D model points.
model_points = np.array([
                        (0.0, 0.0, 0.0),             # Nose tip
                        (0.0, -330.0, -65.0),        # Chin
                        (-225.0, 170.0, -135.0),     # Left eye left corner
                        (225.0, 170.0, -135.0),      # Right eye right corne
                        (-150.0, -150.0, -125.0),    # Left Mouth corner
                        (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                        [[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double"
                        )

while True:
    ret, img = cap.read()
    if ret == True:
        faceboxes = mark_detector.extract_cnn_facebox(img)

        for facebox in faceboxes:
            #squared face image
            face_img = img[facebox[1]: facebox[3],facebox[0]: facebox[2]]

            #yin guobing's model
            face_img = cv2.resize(face_img, (128, 128))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            marks = mark_detector.detect_marks([face_img])
            
            #scale to width and height
            marks *= (facebox[2] - facebox[0])
            
            #adding x y offset
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]
            shape = marks.astype(np.uint)
            
            # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
            image_points = np.array([
                                    shape[30],    # Nose tip
                                    shape[8],     # Chin
                                    shape[36],    # Left eye left corner
                                    shape[45],    # Right eye right corne
                                    shape[48],    # Left Mouth corner
                                    shape[54]     # Right mouth corner
                                ], dtype="double")
            
            #assuming no distortion for simplicity
            dist_coeffs = np.zeros((4,1))
            

            #(https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html#:~:text=The%20solvePnP%20and%20related%20functions,the%20Y%2Daxis%20downward%20and)
            '''
            The solvePnP and related functions estimate the object pose given a set of object points, their corresponding image projections, 
            as well as the camera intrinsic matrix and the distortion coefficients, 
            see the figure below (more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward and the Z-axis forward).
            '''
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

            axis = np.float32([[500,0,0], [0,500,0], [0,0,500]]) # the to be projected one in x y z for pitch yaw roll representation
                                
            imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            #converts rotation vector to rotation matrix using Rodrigues transformation
            rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

            proj_matrix = np.hstack((rvec_matrix, translation_vector))

            '''
            Input
                projMatrix 3x4 input projection matrix P.
            Output
                cameraMatrix 3x3 camera matrix K.
                rotMatrix 3x3 external rotation matrix R.
                transVect 4x1 translation vector T.
                S:
                    rotMatrX 3x3 rotation matrix around x-axis.
                    rotMatrY 3x3 rotation matrix around y-axis.
                    rotMatrZ 3x3 rotation matrix around z-axis.
                eulerAngles 3-element vector containing three Euler angles of rotation in degrees.
            '''
            eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 


            pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


            pitch = math.degrees(math.asin(math.sin(pitch)))
            roll = -math.degrees(math.asin(math.sin(roll)))
            yaw = math.degrees(math.asin(math.sin(yaw)))

            nose = (shape[30][0],shape[30][1])
            imgpts = imgpts.astype(int)
            #print("type:",type(tuple(imgpts[1].ravel())[0]))

            cv2.line(img, nose, tuple(imgpts[1].ravel()), (220,220,220), 3)
            cv2.line(img, nose, tuple(imgpts[0].ravel()), (220,220,220), 3)
            cv2.line(img, nose, tuple(imgpts[2].ravel()), (220,220,220), 3)

            #roll:
            img = cv2.putText(img,"Roll:{}".format(int(roll)),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

            #pitch
            img = cv2.putText(img,"Pitch:{}".format(int(pitch)),(10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2, lineType=2)

            #yaw:
            img = cv2.putText(img,"Yaw:{}".format(int(yaw)),(10,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()
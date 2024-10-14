import cv2
import mediapipe as mp
import pickle
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Update to 478 landmarks when refined landmarks are enabled
EXPECTED_LANDMARKS = 478

def normalize_landmarks(landmarks):
    nose_landmark = landmarks[1]
    normalized_landmarks = [
        {
            'x': landmark.x - nose_landmark.x,
            'y': landmark.y - nose_landmark.y,
            'z': landmark.z - nose_landmark.z
        }
        for landmark in landmarks
    ]
    return normalized_landmarks

def generate_face_signature(landmarks):
    return [landmark['x'] for landmark in landmarks] + [landmark['y'] for landmark in landmarks] + [landmark['z'] for landmark in landmarks]

def save_face_signature(signature, filename='face_signature.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(signature, f)
    print(f"Signature saved with length: {len(signature)}")

cap = cv2.VideoCapture(0)

# Initialize the Face Mesh model
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Ensuring refined landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if len(face_landmarks.landmark) == EXPECTED_LANDMARKS:
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                    normalized_landmarks = normalize_landmarks(face_landmarks.landmark)
                    face_signature = generate_face_signature(normalized_landmarks)
                    
                    # Save the signature
                    save_face_signature(face_signature)
                    break  # Save the signature and exit

        cv2.imshow('Face Mesh', frame_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
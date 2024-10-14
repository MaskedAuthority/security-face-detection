import cv2
import mediapipe as mp
import pickle
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Update to 478 landmarks
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

def load_face_signature(filename='face_signature.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def calculate_euclidean_distance(signature1, signature2):
    return np.linalg.norm(np.array(signature1) - np.array(signature2))

def is_match(live_signature, stored_signature, threshold=0.5):
    if len(live_signature) != len(stored_signature):
        print(f"Length mismatch: live signature has {len(live_signature)} elements, stored signature has {len(stored_signature)} elements.")
        return False

    distance = calculate_euclidean_distance(live_signature, stored_signature)
    print(f"Euclidean distance: {distance}")
    return distance < threshold

# Load the stored face signature
try:
    stored_signature = load_face_signature()
    print(f"Stored face signature loaded with length: {len(stored_signature)}")
except FileNotFoundError:
    print("No stored signature found.")
    exit()

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
                    live_signature = generate_face_signature(normalized_landmarks)

                    print(f"Live signature length: {len(live_signature)}")

                    if is_match(live_signature, stored_signature):
                        cv2.putText(frame_bgr, "Face Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame_bgr, "No Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Face Mesh', frame_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
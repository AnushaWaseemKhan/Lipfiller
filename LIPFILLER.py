import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Upper and lower lip indices
UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183]
LOWER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# Scaling factors for upper and lower lips
UPPER_SCALING = 1.4
LOWER_SCALING = 1.3
ALPHA = 0.7  # Transparency factor for blending

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Face Mesh
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face mesh detection
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Function to process and sample natural lip color
                def process_and_sample_lip(indices, scaling_factor, alpha):
                    # Get lip points
                    lip_points = np.array([(
                        int(face_landmarks.landmark[idx].x * w),
                        int(face_landmarks.landmark[idx].y * h)
                    ) for idx in indices], np.int32)

                    # Create a convex hull around the lip points
                    lip_hull = cv2.convexHull(lip_points)

                    # Create a mask for the lip region
                    lip_mask = np.zeros_like(frame, dtype=np.uint8)
                    cv2.fillPoly(lip_mask, [lip_hull], (255, 255, 255))

                    # Extract the lip region from the frame
                    lip_region = cv2.bitwise_and(frame, lip_mask)

                    # Calculate the average color in the lip region
                    lip_color = cv2.mean(lip_region, mask=lip_mask[:,:,0])[0:3]

                    # Enlarge lip points outward (scaling)
                    center_x, center_y = np.mean(lip_points, axis=0).astype(int)
                    enlarged_points = []
                    for x, y in lip_points:
                        dx, dy = x - center_x, y - center_y
                        enlarged_x = int(center_x + scaling_factor * dx)
                        enlarged_y = int(center_y + scaling_factor * dy)
                        enlarged_points.append((enlarged_x, enlarged_y))

                    # Create a mask for the enlarged lip and fill it with the sampled lip color
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [np.array(enlarged_points)], lip_color)

                    # Blend the color with the original frame
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Process and fill the upper and lower lips with the natural sampled color
                process_and_sample_lip(UPPER_LIP_INDICES, UPPER_SCALING, ALPHA)
                process_and_sample_lip(LOWER_LIP_INDICES, LOWER_SCALING, ALPHA)

        # Display the frame with natural lips
        cv2.imshow('Natural Lips Effect', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

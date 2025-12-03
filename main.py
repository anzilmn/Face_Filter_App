import cv2
import numpy as np
import os
import time

# --- Configuration and File Paths ---

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_CASCADE_PATH = os.path.join(BASE_DIR, 'assets', 'haarcascade_frontalface_default.xml')

# IMPORTANT: This must match your filter image name: cat_filter_full (1).png
FILTER_IMAGE_PATH = os.path.join(BASE_DIR, 'assets', 'cat_filter_full (1).png')


# --- Helper Function for Overlaying PNGs with Transparency ---

def overlay_transparent(background, overlay, x, y, scale=1.0):
    """
    Overlays a transparent PNG image onto a background image at (x, y) with proper blending.
    """
    if scale != 1.0:
        width = int(overlay.shape[1] * scale)
        height = int(overlay.shape[0] * scale)
        overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)

    h, w, _ = background.shape
    h_img, w_img, _ = overlay.shape

    # Define the region of interest (ROI) and handle clipping at the frame edges
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + w_img), min(h, y + h_img)

    bg_slice = background[y1:y2, x1:x2]
    overlay_slice = overlay[y1-y:y2-y, x1-x:x2-x]

    # Ensure the overlay slice is valid and has an alpha channel (4 channels)
    if bg_slice.size == 0 or overlay_slice.size == 0 or overlay_slice.shape[2] < 4:
        return background

    # Extract the alpha channel and normalize it (0.0 to 1.0)
    alpha = overlay_slice[:, :, 3] / 255.0
    color = overlay_slice[:, :, :3]

    # Create masks for blending
    alpha_mask = alpha[:, :, np.newaxis]
    inverse_alpha = 1.0 - alpha_mask

    # Blend the background and the overlay: New = (Overlay * Alpha) + (Background * (1 - Alpha))
    blended_color = (color * alpha_mask) + (bg_slice * inverse_alpha)

    # Put the blended result back into the background frame
    background[y1:y2, x1:x2] = blended_color.astype(np.uint8)

    return background

# --- Main Application Logic ---

def run_snap_filter():
    print("--- Snap Filter App Initializing ---")
    
    # 1. Load Files
    try:
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        filter_img = cv2.imread(FILTER_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        if face_cascade.empty() or filter_img is None:
             raise IOError("One of the asset files failed to load.")
        print("✅ SUCCESS: Assets loaded.")
    except IOError as e:
        print(f"❌ ERROR: Asset loading failed. Details: {e}")
        return

    # 2. Initialize the Video Capture (Webcam)
    
    # !!! --- CRITICAL LINE: ADJUST THIS NUMBER (0, 1, 2, ...) IF THE APP CLOSES INSTANTLY --- !!!
    camera_index = 0 
    # !!! ---------------------------------------------------------------------------------- !!!
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open webcam at index {camera_index}. Try changing the index (0, 1, or 2) in the code.")
        return
    
    print(f"✅ SUCCESS: Webcam started at index {camera_index}. Press 'q' to quit.")

    # Main loop for video processing
    while True:
        # Read a frame. If this fails, the loop breaks and the app closes (which happened before).
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to read frame from camera. Closing.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 4. Detect Faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        # 5. Apply Filter and Draw Green Debug Box
        for (x, y, w, h) in faces:
            
            # --- DEBUGGING: Green Rectangle around the detected face ---
            # If you see this box, face detection is working!
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # --- APPLY FILTER LOGIC ---
            # 1. Scaling: 1.5 times the face width
            filter_scale = 1.5 * w / filter_img.shape[1]
            filter_width = int(filter_img.shape[1] * filter_scale)
            
            # 2. Positioning X: Center horizontally
            filter_x_pos = x - (filter_width - w) // 2
            
            # 3. Positioning Y: Move UP by half the face height
            filter_y_pos = y - int(0.5 * h) 

            # Apply the filter
            frame = overlay_transparent(frame, filter_img, filter_x_pos, filter_y_pos, scale=filter_scale)


        cv2.imshow('Snap Filter Python Cat App (Single Image)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("--- Application Closed ---")

if __name__ == '__main__':
    time.sleep(1)
    run_snap_filter()
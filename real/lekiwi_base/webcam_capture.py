#!/usr/bin/env python3
"""
Webcam image acquisition application for macOS using OpenCV.
This app provides functionality to capture images from the webcam.
"""

import cv2
import numpy as np
import time
from datetime import datetime
import os


class WebcamCapture:
    """Simple webcam capture class using OpenCV."""
    
    def __init__(self, camera_index=0, width=640, height=480):
        """
        Initialize the webcam capture.
        
        Args:
            camera_index: Index of the camera (0 is typically the default webcam)
            width: Width of the captured frame
            height: Height of the captured frame
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.is_running = False
        
    def start(self):
        """Start the webcam capture."""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.is_running = True
        print(f"Webcam started (camera {self.camera_index})")
        print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
    def read_frame(self):
        """
        Read a single frame from the webcam.
        
        Returns:
            tuple: (success, frame) where success is a boolean and frame is the image
        """
        if not self.is_running or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def save_frame(self, frame, directory="captures", prefix="capture"):
        """
        Save a frame to disk.
        
        Args:
            frame: The image frame to save
            directory: Directory to save the image
            prefix: Prefix for the filename
            
        Returns:
            str: Path to the saved image
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(directory, filename)
        
        # Save the image
        cv2.imwrite(filepath, frame)
        print(f"Image saved: {filepath}")
        return filepath
    
    def stop(self):
        """Stop the webcam capture and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            print("Webcam stopped")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop()


def run_interactive_app():
    """Run an interactive webcam application with live preview."""
    print("=" * 60)
    print("Webcam Capture Application")
    print("=" * 60)
    print("Controls:")
    print("  SPACE    - Capture image")
    print("  S        - Save current frame")
    print("  Q or ESC - Quit")
    print("=" * 60)
    
    # Initialize webcam
    webcam = WebcamCapture(camera_index=0, width=1280, height=720)
    
    try:
        webcam.start()
        
        # Create window
        window_name = "Webcam Capture (Press SPACE to capture, Q to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        frame_count = 0
        fps_time = time.time()
        fps = 0
        
        while True:
            # Read frame
            ret, frame = webcam.read_frame()
            
            if not ret:
                print("Failed to read frame")
                break
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()
            
            # Add FPS text to frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC
                print("Quitting...")
                break
            elif key == ord(' ') or key == ord('s'):  # SPACE or s
                filepath = webcam.save_frame(frame)
                print(f"✓ Captured: {filepath}")
                
                # Flash effect
                white_frame = np.ones_like(frame) * 255
                cv2.imshow(window_name, white_frame)
                cv2.waitKey(50)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        webcam.stop()
        cv2.destroyAllWindows()
        print("Application closed")


def capture_single_image(save_path="capture.jpg", camera_index=0):
    """
    Capture a single image without GUI (useful for automation).
    
    Args:
        save_path: Path where to save the captured image
        camera_index: Index of the camera to use
        
    Returns:
        tuple: (success, image) where success is a boolean
    """
    webcam = WebcamCapture(camera_index=camera_index)
    
    try:
        webcam.start()
        
        # Wait a moment for camera to warm up
        time.sleep(0.5)
        
        # Capture frame
        ret, frame = webcam.read_frame()
        
        if ret:
            cv2.imwrite(save_path, frame)
            print(f"Image captured and saved to: {save_path}")
            return True, frame
        else:
            print("Failed to capture image")
            return False, None
            
    finally:
        webcam.stop()


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--single":
            # Capture single image mode
            output_path = sys.argv[2] if len(sys.argv) > 2 else "capture.jpg"
            capture_single_image(output_path)
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python webcam_capture.py              - Run interactive app")
            print("  python webcam_capture.py --single     - Capture single image")
            print("  python webcam_capture.py --single <path> - Capture to specific path")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Run interactive app
        run_interactive_app()

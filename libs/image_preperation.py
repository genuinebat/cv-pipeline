import cv2
import os

input_dir = "1input_vids"
output_dir = "2input_imgs"

class VideoConverter:
    _valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

    def __init__(self, input_dir, output_dir):
        self._input_vids = input_dir
        self._output_vids = output_dir

    def convert_videos_to_images(self):
        for filename in os.listdir(self._input_vids):
            if not filename.lower().endswith(self._valid_extensions):
                continue

            video_path = os.path.join(self._input_vids, filename)
            video_name = os.path.splitext(filename)[0]
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Failed to open {filename}")
                continue

            saved_count = 0

            while True:
                success, frame = cap.read()
                
                if not success:
                    break
                    
                out_filename = f"{video_name}_frame_{saved_count:04d}.jpg"
                out_path = os.path.join(self._output_vids, out_filename)
                
                cv2.imwrite(out_path, frame)
                
                saved_count += 1

            cap.release()

            os.remove(video_path)

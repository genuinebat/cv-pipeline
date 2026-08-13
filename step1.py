import os
from image_preperation import VideoConverter 

if __name__ == "__main__":
    INPUT = "./1input_vids"
    OUTPUT = "./2input_imgs"

    num_vids = len(os.listdir(INPUT))
    vc = VideoConverter(input_dir=INPUT, output_dir=OUTPUT)
    vc.convert_videos_to_images()
    num_imgs = len(os.listdir(output_imgs))

    print(f"""
    ------------ Phase 1 Results ------------
    {num_vids} video{"" if num_vids == 1 else "s"}
    converted into
    {num_imgs} image{"" if num_vids == 1 else "s"} 
    -----------------------------------------
    """)
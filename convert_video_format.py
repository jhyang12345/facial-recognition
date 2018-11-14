import os, sys
from moviepy.editor import VideoFileClip, concatenate_videoclips

def main(argv):
    file_path = argv[-1]
    original_video = VideoFileClip(file_path)
    video_path = os.path.dirname(file_path)
    video_name = os.path.basename(file_path).split(".")[0]
    video_name = video_name + ".mp4"
    original_video.write_videofile(os.path.join(video_path, video_name))#, codec="libvpx")
    pass

if __name__ == '__main__':
    main(sys.argv[1:])

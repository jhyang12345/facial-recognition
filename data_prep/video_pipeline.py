import sys, os
from PIL import Image
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
from data_prep.image_pipeline import ArrayFeeder
from test import get_boolean_from_output
from config_helper import retrieve_option_model
from argparse import ArgumentParser

def cut_video_clip(video_path, start, end):
    original_video = VideoFileClip(video_path)
    subclip = original_video.subclip(start, end)
    video_name = os.path.basename(video_path).split(".")[0]
    video_name = video_name + "_sub.mp4"
    subclip.write_videofile(video_name)

class MakeClassifiedVideo:
    def __init__(self, video_path):
        self.video_path = video_path
        self.init_model()
        self.iterate_through_video()

    def init_model(self):
        self.model = retrieve_option_model("")
        self.model.load_model()


    def iterate_through_video(self):
        original_video = VideoFileClip(self.video_path)
        count = 0
        new_frames = []
        i = 0
        for frame in original_video.iter_frames():
            i += 1
            if i % 3 == 0:
                self.get_classified_frame(frame, i)
        print(count)

    def get_classified_frame(self, frame, i):
        array_feeder = ArrayFeeder(frame)
        input_data = array_feeder.input_data
        output_data = self.model.model.predict(input_data)
        location_values = get_boolean_from_output(output_data)
        new_frame = array_feeder.set_location_values(location_values)
        frame_path = os.path.join(".", "frames")
        frame_name = "{}_{}.jpg".format(os.path.join(frame_path, "frame"), i)
        Image.fromarray(np.uint8(new_frame)).save(frame_name)
        print("New frame extracted: {}".format(i))

def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cut", dest="cut")
    parser.add_argument("-s", "--start", dest="start")
    parser.add_argument("-e", "--end", dest="end")
    parser.add_argument("-m", "--make", dest="make")
    args = parser.parse_args()
    if args.cut:
        path = args.cut
        start = int(args.start)
        end = int(args.end)
        cut_video_clip(path, start, end)
    elif args.make:
        path = args.make
        MakeClassifiedVideo(path)
    # video_path = sys.argv[-1]
    # MakeClassifiedVideo(video_path)


if __name__ == '__main__':
    main()

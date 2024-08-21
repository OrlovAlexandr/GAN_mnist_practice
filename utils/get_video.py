from pathlib import Path

import cv2
import imageio.v3 as iio
from config import SequenceType
from tqdm import tqdm


def save_sequence(
        images_path: Path,
        sequence_type: SequenceType = SequenceType.VIDEO,
        every_n_image: int = 1,
        fps: int = 1,
):
    images = sorted(images_path.glob('*.png'), key=lambda x: x.stat().st_mtime)
    if images:
        last_frame = cv2.imread(str(images[-1]))
        height, width, layers = last_frame.shape

        images = images[::every_n_image]

        if sequence_type == SequenceType.VIDEO:
            video = cv2.VideoWriter(
                str(images_path.parent / 'video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps=fps,
                frameSize=(width, height),
            )
            for image in tqdm(images):
                frame = cv2.imread(str(image))
                cv2.imshow('frame', frame)
                video.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
            video.release()

        elif sequence_type == SequenceType.GIF:
            output_gif = images_path.parent / 'seq_io.gif'
            images_io = [iio.imread(str(image)) for image in images]
            iio.imwrite(str(output_gif), images_io, extension='.gif', fps=fps, loop=0, subrectangles=True)

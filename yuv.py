import numpy as np
import matplotlib.pyplot as plt

# yuv to rgb table
yuv_rgb_table = np.array([
    [1.0, 0.0, 1.4022],
    [1.0, -0.3456, -0.7145],
    [1.0, 1.7710, 0.0],
])


# Convert yuv matrix to rgb
def convert_yuv_to_rgb(yuv):
    yuv = np.double(yuv)
    yuv[:, 1: 3] = yuv[:, 1: 3] - 127
    rgb = (np.matmul(yuv_rgb_table, yuv.transpose())).transpose()
    rgb = np.clip(rgb, 0, 255)
    rgb = np.uint8(rgb)
    return rgb


# load_yuv: load num_frames of video from file path
# you must set width and height of your video
def load_yuv(file, width, height, frames_idx=[]):
    fio = open(file, "rb")
    sub_sample = np.array([[1, 1], [1, 1]])
    size_frame = int(1.5 * width * height)
    frames = []
    for fr in frames_idx:
        img_yuv = np.zeros((height, width, 3))  # y, u, v
        fio.seek((fr - 1) * size_frame)

        # read Y component
        buf = fio.read(width * height)
        buf = np.frombuffer(buf, dtype=np.uint8)
        img_yuv[:, :, 0] = buf.reshape((height, width))

        # read U component
        buf = fio.read(int(width / 2 * height / 2))
        buf = np.frombuffer(buf, dtype=np.uint8)
        img_yuv[:, :, 1] = np.kron(buf.reshape((int(height / 2), int(width / 2))), sub_sample)

        # read V component
        buf = fio.read(int(width / 2 * height / 2))
        buf = np.frombuffer(buf, dtype=np.uint8)
        img_yuv[:, :, 2] = np.kron(buf.reshape((int(height / 2), int(width / 2))), sub_sample)

        # convert YUV to RGB
        img_yuv = img_yuv.reshape((width * height, 3))
        img_rgb = convert_yuv_to_rgb(img_yuv)
        img_rgb = img_rgb.reshape((height, width, 3))

        frames.append(img_rgb)

    fio.close()
    return np.array(frames)

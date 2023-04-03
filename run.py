import os
import sys
import glob
import time
import tarfile
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path, input_size):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    self.INPUT_SIZE = input_size

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.compat.v1.Session(graph=self.graph)
    return

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.resize(target_size, Image.ANTIALIAS)

    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  return colormap

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--model', type=str, default='deeplab_cityscapes_xception71_trainvalfine_2018_09_08.tar.gz')
  parser.add_argument('--input', type=str, default='rgb_1024_512_all')
  parser.add_argument('--output', type=str, default='rgb_1024_512_all_sem')
  parser.add_argument('--size', type=int, default=1024)
  parser.add_argument('--num_seg', type=int, default=1)
  parser.add_argument('--seg', type=int, default=0)
  args = parser.parse_args()

  color_map = create_cityscapes_label_colormap()
  os.popen('mkdir -p ' + args.output)

  model = DeepLabModel(args.model, args.size)
  print('Model loaded successfully!')

  im_files = glob.glob(args.input + '/*.jpg') + glob.glob(args.input + '/*.png')
  start = int(len(im_files) * args.seg / args.num_seg)
  end = len(im_files) if (args.seg == args.num_seg - 1) else int(len(im_files) * (args.seg+1) / args.num_seg)
  for im_file in tqdm(sorted(im_files)[start: end]):
    basename = os.path.basename(im_file).replace('.jpg', '.png')
    im = Image.open(im_file)
    resized_image, seg_map = model.run(im)

    seg_rgb = color_map[seg_map.flatten()].reshape(seg_map.shape + (3,))
    alpha = (np.array(resized_image) * 0.618 + seg_rgb * 0.382).astype(np.uint8)

    res = Image.fromarray(seg_map.astype(np.uint8)).convert('L')
    res.putpalette(color_map.flatten())
    res.save(args.output + '/' + basename)

    Image.fromarray(alpha).save('rgb_1024_512_all_vis/' + basename)


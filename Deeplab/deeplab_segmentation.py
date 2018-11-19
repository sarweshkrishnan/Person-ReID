#@title Imports

import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import time
import datetime
import random

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFilter

import tensorflow as tf
import cv2

#@title Helper methods

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    # resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    resize_ratio = 1.0
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

def main():
    #@title Select and download models {display-mode: "form"}

    # MODEL_NAME = 'xception_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

    # _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    # _MODEL_URLS = {
    #     'mobilenetv2_coco_voctrainaug':
    #         'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    #     'mobilenetv2_coco_voctrainval':
    #         'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    #     'xception_coco_voctrainaug':
    #         'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    #     'xception_coco_voctrainval':
    #         'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    # }
    # _TARBALL_NAME = 'deeplab_model.tar.gz'

    # model_dir = tempfile.mkdtemp()
    # tf.gfile.MakeDirs(model_dir)

    # download_path = os.path.join(model_dir, _TARBALL_NAME)
    # print('downloading model, this might take a while...')
    # urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
    #                 download_path)
    # print('download completed! loading DeepLab model...')

    # MODEL = DeepLabModel(download_path)

    print('Loading DeepLab model...')
    MODEL = DeepLabModel('C:\\Users\\John\\Documents\\TAMU\\CSCE625\\DL\\Project_Repo\\Deeplab\\deeplabv3_pascal_trainval_2018_01_04.tar.gz')

    print('model loaded successfully!')

    #@title Run on sample images {display-mode: "form"}

    SAMPLE_IMAGE = 'image1'  # @param ['image1', 'image2', 'image3']
    IMAGE_URL = ''  #@param {type:"string"}

    _SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/'
                'deeplab/g3doc/img/%s.jpg?raw=true')

    def run_visualization(url):
        """Inferences DeepLab model and visualizes result."""
        try:
            # f = urllib.request.urlopen(url)
            # jpeg_str = f.read()
            # original_im = Image.open(BytesIO(jpeg_str))
            print(url)
            original_im = Image.open(url)
            original_im_enchance = original_im.filter(ImageFilter.DETAIL)

            # 1,150,5

            # original_im_enchance = original_im.filter(ImageFilter.UnsharpMask(radius=3, percent=200, threshold=7))
            # original_im = original_im.filter(ImageFilter.MinFilter)

            # kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
            # original_im = cv2.filter2D(original_im, -1, kernel)

            # img = img/255.0
            # img = cv2.pow(img,0.6)

        except IOError:
            print('Cannot retrieve image. Please check file: ' + url)
            return

        print('running deeplab on image %s...' % url)
        resized_im, seg_map = MODEL.run(original_im_enchance)
        # resized_im, seg_map = MODEL.run(original_im)

        seg_image = Image.fromarray(label_to_color_image(seg_map).astype(np.uint8).astype('uint8'), 'RGB')
        # print(np.shape(seg_image))
        # print(type(seg_image))
        # print(type(resized_im))
        # print(np.shape(convert_to_cv2_img(resized_im)))
        # print(np.shape(convert_to_cv2_img(original_im)))
        # print(np.shape(convert_to_cv2_img(seg_image)))

        return process_image(convert_to_cv2_img(original_im), convert_to_cv2_img(seg_image), convert_to_cv2_img(resized_im))
        # vis_segmentation(resized_im, seg_map)

    def convert_to_cv2_img(pil_image):
      pil_image = pil_image.convert('RGB')
      open_cv_image = np.array(pil_image)
      # Convert RGB to BGR
      open_cv_image = open_cv_image[:, :, ::-1].copy()
      return open_cv_image

    def process_image(original_im, mask_im, resize_im):
      try:
        # get filename and kernel size values from command line
        img = original_im
        mask = mask_im
        k = 7

        # read and display the original image
        # cv2.namedWindow("original", cv2.WINDOW_NORMAL)
        # cv2.imshow("original", img)

        # cv2.namedWindow("resize_im", cv2.WINDOW_NORMAL)
        # cv2.imshow("resize_im", resize_im)

        # blur and grayscale before thresholding
        blur = cv2.cvtColor(mask_im, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(blur, (k, k), 0)

        # perform adaptive thresholding
        (t, maskLayer) = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY +
          cv2.THRESH_OTSU)

        # make a mask suitable for color images
        mask = cv2.merge([maskLayer, maskLayer, maskLayer])

        # cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        # cv2.imshow("mask", mask)

        image_rgb_nobg = cv2.bitwise_and(img, mask)
        # cv2.namedWindow("selected", cv2.WINDOW_NORMAL)
        # cv2.imshow("selected", image_rgb_nobg)
        # cv2.waitKey(5000)

        print ( mask[np.where((mask == [255,255,255]).all(axis = 2))].size )
        print ( mask[np.where((mask == [0,0,0]).all(axis = 2))].size )
        print ( mask.size )
        print ( 1.0 * mask[np.where((mask == [255,255,255]).all(axis = 2))].size/mask.size )

        # this is to check threshold of clear pixels from the deeplab network
        # in case where no humans are segmented or very little is segmented so we return the original image
        white_per_threshold =  1.0 * mask[np.where((mask == [255,255,255]).all(axis = 2))].size/mask.size
        if white_per_threshold > 0.28:
          random.seed(datetime.datetime.now())
          row, column, channels = image_rgb_nobg.shape
          # recolor black background with random pixels
          for i in range(row):
            for j in range(column):
              if np.any(image_rgb_nobg[i,j] == [0,0,0]):
                B_random = random.randint(0, 255)
                G_random = random.randint(0, 255)
                R_random = random.randint(0, 255)
                image_rgb_nobg[i, j] = [B_random, G_random, R_random]

          return image_rgb_nobg
        else:
          return img

        # return image_rgb_nobg
      except IOError:
        print('Cannot open image!')
        return

    def remove_background(dir_path = 'C:\\Users\\John\\Documents\\TAMU\\CSCE625\\DL\\Market-1501-v15.09.15\\Market-1501-v15.09.15\\pytorch'):
      if not os.path.isdir(dir_path):
        return print("Error: No directory found")

      for root, dirs, files in os.walk(dir_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            image_no_bg = run_visualization(os.path.join(root, name))
            cv2.imwrite(root+"\\"+name, image_no_bg)

    remove_background('C:\\Users\\John\\Documents\\TAMU\\CSCE625\\DL\\Project_Repo\\Deeplab\\RANDVALSET')

if __name__ == '__main__':
  start_time = time.time()
  main()
  print("--- %s seconds ---" % (time.time() - start_time))

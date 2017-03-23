import numpy as np
import os
import caffe
import cv2
import argparse


def get_mean_imagenet(mean_image_path):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_image_path, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    arr = arr.reshape(3, 256, 256)
    arr = np.transpose(arr,[1,2,0])
    return arr


def get_sherlock_vector(net, blob, prototxt, model):

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    # im = caffe.io.load_image(image)
    result = np.empty([0, 3, 224, 224])
    for i in blob:
        result = np.append(result, [transformer.preprocess('data', i)], axis=0)
    net.blobs['data'].reshape(result.shape[0], result.shape[1],result.shape[2],result.shape[3])
    net.blobs['data'].data[...] = result  # transformer.preprocess('data', blob)
    net.reshape()

    print "Running images through Sherlock"

    out = net.forward()
    # print out

    # print net.blobs['image_lstlayer_PO'].data.shape
    vec_600 = net.blobs['image_lstlayer_PO'].data.copy()
    vec_300 = net.blobs['image_lstlayer_S'].data.copy()
    return np.concatenate([vec_300, vec_600], axis=1)


def im_list_to_blob(ims):
    """Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def prep_im_for_blob(im, mean_image, width, height):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32)
    im = cv2.resize(im, (256, 256))
    im = im - mean_image
    cropped_image = im[16:240, 16:240]
    return cropped_image


def get_images(path):
    images = []
    for filename in os.listdir(path):
        images.append(os.path.join(path, filename))
    return images


def get_image(path):
    im = caffe.io.load_image(path)
    return im

gpu_id = 2
parser = argparse.ArgumentParser()
parser.add_argument('-prototxt', type=str, default='/home/rohit/Downloads/deploy.prototxt')
parser.add_argument('-model', type=str, default='/home/rohit/Downloads/sherlock.caffemodel')
parser.add_argument('-path', type=str, default='/media/rohit/New Volume/train2014')
parser.add_argument('-mean_image', type=str, default='/home/rohit/Downloads/caffe/imagenet_mean.binaryproto')
args = parser.parse_args()

net = caffe.Net(args.prototxt, args.model, caffe.TEST)
caffe.set_mode_gpu()
caffe.set_device(gpu_id)

images = get_images(args.path)
sherlock_features = []
image_list = []
batch_size = 60
batch_number = 1
counter = 0
num = 0
i = 0

id_map_file = open('id_map.txt','w',0)

print len(images)

while i < len(images):
    num += 1
    filename = os.path.basename(images[i])
    image_id = filename.replace('COCO_train2014_','').replace('.jpg','').lstrip('0')
    id_map_file.write(image_id + " " + str(num) + "\n" )
    image_list.append(prep_im_for_blob(get_image(images[i]), get_mean_imagenet(args.mean_image), 224, 224))
    if (i + 1) % (batch_size) == 0:
        print num
        blob = im_list_to_blob(image_list)
        image_list = []
        sherlock_features.append(get_sherlock_vector(net, blob, args.prototxt, args.model))
        print "batch done", batch_number
        batch_number += 1
    i += 1
print len(image_list)
blob = im_list_to_blob(image_list)
sherlock_features.append(get_sherlock_vector(net, blob, args.prototxt, args.model))
print "Length of sherlock features:",len(sherlock_features)
sherlock_features_numpy = np.array(sherlock_features[:-1])
sherlock_features_numpy = sherlock_features_numpy.reshape(
    sherlock_features_numpy.shape[0] * sherlock_features_numpy.shape[1], 900)
sherlock_features_numpy = np.append(sherlock_features_numpy,sherlock_features[-1],axis=0)

print sherlock_features_numpy.shape

np.save('sherlock_features.npy', sherlock_features_numpy)

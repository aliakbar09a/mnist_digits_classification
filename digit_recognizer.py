import input_data
import prep_data
import cv2

mnist = input_data.read_data_sets("data/", one_hot=False)
data = mnist.train.next_batch(5000)



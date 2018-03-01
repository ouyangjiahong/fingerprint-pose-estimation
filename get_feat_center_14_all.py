#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import struct

CLASSES = ('__background__', '0', '15', '30', '45', '60', '75', '90', '105', '120', '135', '150', '165', '180', '195', '210', '225', '240', '255', '270', '285', '300', '315', '330', '345')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final_24.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, img_name,class_name, dets, inds, thresh=0.5):
    """Draw detected bounding boxes."""
    #inds = np.where(dets[:, -1] >= thresh)[0]
    #if len(inds) == 0:
    #    return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    dst_img_dir = 'result_test/'
    plt.savefig(dst_img_dir + img_name)

def merge(bbox, cls):
    """merge the bbox from different class and get the angle"""
    THRESH = 0.2
    OVERLAP_THRESH = 0.3
    num = bbox.shape[0]
    index = np.argsort(bbox[:, 4])
    #print bbox
    #print index
    obj = []
    obj_flag = []
    for i in range(0, num):
	flag = False
	idx = index[num-1-i]
	if i ==0:
	    obj.append([bbox[idx,0],bbox[idx,1],bbox[idx,2],bbox[idx,3],bbox[idx,4],int(cls[idx])])
	    obj_flag.append(False)
	    continue	
	if bbox[idx, 4] < THRESH:
	    break
	else:
	    for j in range(0,len(obj)):
		ob = obj[j]
		sx = max(ob[0], bbox[idx,0])
		sy = max(ob[1], bbox[idx,1])
		ex = min(ob[2], bbox[idx,2])
		ey = min(ob[3], bbox[idx,3])
		area1 = (ob[2]-ob[0])*(ob[3]-ob[1])
		area2 = (bbox[idx,2]-bbox[idx,0])*(bbox[idx,3]-bbox[idx,1]) 
                overlap = (ex-sx)*(ey-sy)                     
		if sx<ex and sy<ey and overlap>OVERLAP_THRESH*area1 and overlap>OVERLAP_THRESH*area2: #same
		    flag = True
		    if obj_flag[j] == True:  #match but already out
                        break
		   #interpolate
		    cls1 = int(ob[5])
		    cls2 = int(cls[idx])
		    if abs(cls1-cls2)<=30 or abs(abs(cls1-cls2)-360)<=30:  #<30
			prob = ob[4]+bbox[idx,4]
			if abs(cls1-cls2)>30:
			    if cls1>cls2:
				cls1 = cls1 - 360
			    else:
				cls2 = cls2 - 360
			ang = (ob[4]*cls1+bbox[idx,4]*cls2)/prob
			prob = ob[4]
			if ang < 0:
			    ang = ang+360
		    else:  # >30
			prob = ob[4]
			ang = cls1
		    obj[j][4]=prob               
		    obj[j][5]=ang
		    obj_flag[j] = True
		    
	    if flag == False:   #not match new                
                obj_flag.append(False)                        
		obj.append([bbox[idx,0],bbox[idx,1],bbox[idx,2],bbox[idx,3],bbox[idx,4],int(cls[idx])])
    return obj	
    


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im_file = '/media/med4T1/oyjh/data/nist14_all_raw_jpg/' + image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    #feat_dir = 'feat_14/'
    image_num = image_name.split('.')[0]
    #fc7_name = feat_dir + image_num + '.txt'
    #bbox_dir = 'bbox_14/'
    #bbox_name = bbox_dir + image_num + '.txt'
    pose_dir = 'pose_14_all/'
    pose_name = pose_dir + image_num + '.txt'

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal') 
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    bbox_data = []
    cls_data = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        #print 'cls_ind' , cls_ind
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
	keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :] #comment if don't need nms
        #vis_detections(im, image_name, cls, dets, thresh=CONF_THRESH)
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            tmplen = len(dets[:, -1])
            index = np.argsort(dets[:, -1])
            inds = [index[tmplen - 1]]
	    #f_err = file('low_poss.txt', 'a+')
    	    #f_err.write(image_num)
	    #f_err.write('\n')
	    #f_err.close()
	
        # with open(fc7_name, 'wb') as f:
        #     feat = net.blobs['fc7'].data[inds]
        #     np.savetxt(fc7_name, feat)
	for i in inds:
            #print 'i in inds'
	    bbox = dets[i,:4]
	    score = dets[i, -1]
	    #print score
            bbox_cur = [bbox[0], bbox[1], bbox[2], bbox[3], score]
            bbox_data.append(bbox_cur)
	    cls_data.append(cls)
    bbox_data = np.array(bbox_data)
    #cls_data = np.array(cls_data)
    obj_out = merge(bbox_data, cls_data)
    #print 'obj_out:'
    #print obj_out

    bbox_out = []
    for bbox in obj_out:
	#ax.add_patch(
        #plt.Rectangle((bbox[0], bbox[1]),
        #               bbox[2] - bbox[0],
        #               bbox[3] - bbox[1], fill = False,
        #               edgecolor = 'red', linewidth = 3.5)
        #     )
	#ax.text(bbox[0], bbox[1] - 2,
	#     '{:.3f} {:.3f}'.format(bbox[4], bbox[5]),
        #     bbox=dict(facecolor='blue', alpha=0.5),  
        #     fontsize=14, color='white')

	center_x = (bbox[0] + bbox[2])/2
        center_y = (bbox[1] + bbox[3])/2
	tmpdata = [bbox[4], int(bbox[5]),  center_x, center_y]
        bbox_out.append(tmpdata)
    pose_tmp = bbox_out[0]    
    bbox_out = np.array(bbox_out)
    #np.savetxt(bbox_name, bbox_out, fmt = '%f')
    if pose_tmp[1]>180:
	pose_tmp[1] = pose_tmp[1] - 360
    pose_out = [pose_tmp[2],pose_tmp[3],pose_tmp[1],pose_tmp[0]]
    pose_out = np.array([pose_out])
    np.savetxt(pose_name,pose_out,fmt = '%f')
	
    #plt.axis('off')
    #plt.tight_layout()
    #plt.draw()
    #dst_img_dir = 'result_14/'
    #plt.savefig(dst_img_dir + im_name)
       #vis_detection(im, img_name, cls, dets, inds, CONF_THRESH) 

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    #im_names = ['012001.jpg', '013001.jpg', '014001.jpg',
    #            '015001.jpg']
    #test_set
    #im_names = ['038887.jpg', '035252.jpg', '042582.jpg', 
    #		'006217.jpg', '022492.jpg']    
    #downsample
    #im_names = ['017999.jpg', '035998.jpg', '053996.jpg',
    #		 '071980.jpg', '089972.jpg']
    #NIST27
    #im_names = ['13.jpg']
#    im_names = []
 #   for i in range(1, 259):
#	tmp_name = str(i) + '.jpg'
#	im_names.append(tmp_name)
    im_names = []
    with open('nist14_all_num_rest2.txt', 'r') as fi:
	while(True):
	    line = fi.readline().strip().split()
	    if not line:
		break;
	    im_names.append(line[0] + '.jpg')
    print 'read imagelist done'
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        #print 'Demo for data/demo/{}'.format(im_name)
        print 'extract feature for ' + im_name
	demo(net, im_name)

    plt.show()

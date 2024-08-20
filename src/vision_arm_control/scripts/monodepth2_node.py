#!/usr/bin/env python3
import sys
import os
import rospy
import torch
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# Append the monodepth2 path
sys.path.append(os.path.expanduser('~/catkin_ws/src/monodepth2'))

# Assuming you have these modules in your project
from monodepth2.networks.resnet_encoder import ResnetEncoder
from monodepth2.networks.depth_decoder import DepthDecoder

class Monodepth2Node:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('monodepth2_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Load Monodepth2 model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = ResnetEncoder(18, False).to(self.device)
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4)).to(self.device)

        # Load model weights
        weights_path = rospy.get_param('~weights_path', '/home/edr/catkin_ws/src/vision_arm_control/weights/monodepth2_weights.pth')
        try:
            loaded_dict = torch.load(weights_path, map_location=self.device)
            self.encoder.load_state_dict({k: v for k, v in loaded_dict.items() if k.startswith('encoder')})
            self.depth_decoder.load_state_dict({k: v for k, v in loaded_dict.items() if k.startswith('depth')})
            
            # Add detailed logging
            rospy.loginfo(f"Loaded dictionary keys: {loaded_dict.keys()}")
            rospy.loginfo(f"Encoder state dict keys: {self.encoder.state_dict().keys()}")
            rospy.loginfo(f"Decoder state dict keys: {self.depth_decoder.state_dict().keys()}")
        except Exception as e:
            rospy.logerr(f"Failed to load model weights: {e}")

        # Subscribe to input image topic
        self.image_sub = rospy.Subscriber(rospy.get_param('~input_topic', '/camera/image_raw'), Image, self.image_callback)

        # Publisher for depth map
        self.depth_pub = rospy.Publisher(rospy.get_param('~output_topic', '/monodepth2/depth'), Image, queue_size=1)

        # Set processing rate
        self.rate = rospy.Rate(rospy.get_param('~processing_rate', 10))  # 10 Hz default

        self.input_height = rospy.get_param('~input_height', 192)
        self.input_width = rospy.get_param('~input_width', 640)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Preprocess image
        input_image = self.preprocess(cv_image)

        # Compute depth
        with torch.no_grad():
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (cv_image.shape[0], cv_image.shape[1]), mode="bilinear", align_corners=False)
        
        depth_map = 1 / disp_resized.squeeze().cpu().numpy()

        # Normalize depth map for visualization
        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
        depth_map = (depth_map * 255).astype(np.uint8)

        # Publish depth map
        try:
            depth_msg = self.bridge.cv2_to_imgmsg(depth_map, "mono8")
            self.depth_pub.publish(depth_msg)
        except CvBridgeError as e:
            rospy.logerr(e)

    def preprocess(self, img):
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return img

if __name__ == '__main__':
    node = Monodepth2Node()
    rospy.spin()

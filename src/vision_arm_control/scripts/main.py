#!/usr/bin/env python3

import sys
import os
import rospy
import torch
import cv2
import numpy as np
import mediapipe as mp
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

# Append the monodepth2 path
sys.path.append(os.path.expanduser('~/catkin_ws/src/monodepth2'))

# Import monodepth2 modules
from monodepth2.networks.resnet_encoder import ResnetEncoder
from monodepth2.networks.depth_decoder import DepthDecoder

class VisionArmControl:
    def __init__(self):
        rospy.init_node('vision_arm_control', anonymous=True)
        
        self.is_running = True
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback, queue_size=1)
        self.depth_pub = rospy.Publisher("/monodepth2/depth", Image, queue_size=1)
        
        self.pose_estimator = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.arm_controller = PandaArmController()

        # Initialize Monodepth2 model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = ResnetEncoder(18, False).to(self.device)
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4)).to(self.device)

        weights_path = rospy.get_param('~monodepth2_weights_path', '')
        if weights_path:
            self.load_weights(weights_path)
        
        rospy.on_shutdown(self.shutdown_hook)

    def load_weights(self, weights_path):
        try:
            loaded_dict = torch.load(weights_path, map_location=self.device)
            encoder_dict = {k: v for k, v in loaded_dict.items() if k.startswith('encoder')}
            decoder_dict = {k: v for k, v in loaded_dict.items() if k.startswith('depth')}
            
            self.encoder.load_state_dict(encoder_dict)
            self.depth_decoder.load_state_dict(decoder_dict)
            
            rospy.loginfo("Model weights loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load model weights: {e}")

    def image_callback(self, data):
        if not self.is_running:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        landmarks = self.estimate_pose(cv_image)
        if landmarks:
            target_pose = self.map_human_pose_to_robot(landmarks)
            self.arm_controller.move_arm(target_pose)

        depth_map = self.estimate_depth(cv_image)
        self.publish_depth(depth_map) #this is causing my error. Not sure why its saying something about not havind an encoder
        # i downloaded the kiti 
        # Display the image and depth map
        cv2.imshow("Camera Feed", cv_image)
        cv2.imshow("Depth Map", depth_map)
        cv2.waitKey(1)

    def estimate_pose(self, frame):
        results = self.pose_estimator.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return results.pose_landmarks.landmark if results.pose_landmarks else None

    def map_human_pose_to_robot(self, landmarks):
        # Implement your mapping logic here
        # This is a placeholder implementation
        target_pose = PoseStamped()
        # Set target_pose based on landmarks
        return target_pose

    def estimate_depth(self, image):
        with torch.no_grad():
            input_image = self.preprocess(image)
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)
            disp = outputs[("disp", 0)]
            return disp.squeeze().cpu().numpy()

    def preprocess(self, img):
        img = cv2.resize(img, (640, 192))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    def publish_depth(self, depth_map):
        try:
            depth_msg = self.bridge.cv2_to_imgmsg((depth_map * 255).astype(np.uint8), "mono8")
            self.depth_pub.publish(depth_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Error publishing depth map: {e}")

    def shutdown_hook(self):
        rospy.loginfo("Shutting down...")
        self.is_running = False
        cv2.destroyAllWindows()

    def run(self):
        rospy.spin()

class PandaArmController:
    def __init__(self):
        self.joint_pub = rospy.Publisher('/franka_state_controller/joint_commands', JointState, queue_size=10)
        self.panda_chain = self.setup_ikpy_chain()

    def setup_ikpy_chain(self):
        # This is a simplified chain. You'll need to adjust these values based on your actual robot.
        return Chain(name='panda', links=[
            OriginLink(),
            URDFLink(
                name="panda_link1",
                origin_translation=[0, 0, 0.333],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
            ),
            URDFLink(
                name="panda_link2",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
            ),
            # Add more links as needed
        ])

    def move_arm(self, target_pose):
        # Convert PoseStamped to a 4x4 transformation matrix
        target_matrix = np.eye(4)
        target_matrix[:3, 3] = [target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z]
        # You'll need to convert the quaternion to a rotation matrix here

        # Compute IK
        ik_solution = self.panda_chain.inverse_kinematics_frame(target_matrix)

        # Publish joint commands
        joint_state = JointState()
        joint_state.position = ik_solution[1:]  # Exclude the base rotation
        self.joint_pub.publish(joint_state)

if __name__ == '__main__':
    vision_arm_control = VisionArmControl()
    vision_arm_control.run()

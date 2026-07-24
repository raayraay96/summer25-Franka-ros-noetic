#!/usr/bin/env python3
import sys
import os
import rospy
import torch
import cv2
import numpy as np
import mediapipe as mp
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState, CompressedImage
from geometry_msgs.msg import PoseStamped
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from tf.transformations import quaternion_from_euler

# Append the monodepth2 path 
sys.path.append(os.path.expanduser('~/catkin_ws/src/monodepth2')) 

# Import monodepth2 modules
from monodepth2.networks import ResnetEncoder, DepthDecoder

class PandaArmController:
    def __init__(self):
        self.joint_pub = rospy.Publisher('/franka_state_controller/joint_commands', JointState, queue_size=10)
        self.panda_chain = self.setup_ikpy_chain()

    def setup_ikpy_chain(self):
        return Chain(name='panda', links=[
            OriginLink(),
            URDFLink(name="panda_link1", origin_translation=[0, 0, 0.333], origin_orientation=[0, 0, 0], rotation=[0, 0, 1]),
            URDFLink(name="panda_link2", origin_translation=[0, 0, 0], origin_orientation=[0, 0, 0], rotation=[0, 1, 0]),
            # ... add other links
        ])

    def move_arm(self, target_pose):
        target_frame = np.eye(4)
        target_frame[:3, 3] = [target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z]
        # ... populate target_frame with orientation from target_pose
        
        try:
            ik_solution = self.panda_chain.inverse_kinematics_frame(target_frame)
            joint_state = JointState()
            joint_state.position = ik_solution[1:]  # Exclude the base rotation
            self.joint_pub.publish(joint_state)
            rospy.loginfo(f"Moving arm to: {target_pose}")
        except Exception as e:
            rospy.logwarn(f"IK solution failed or arm movement error: {e}")

class VisionArmControl:
    def __init__(self):
        rospy.init_node('vision_arm_control', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.image_callback, queue_size=1, buff_size=2**24) 
        self.depth_pub = rospy.Publisher("/monodepth2/depth", Image, queue_size=1)
        
        # Initialize pose estimator
        self.mp_pose = mp.solutions.pose
        self.pose_estimator = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Initialize other components (encoder, depth_decoder, etc.)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = ResnetEncoder(18, False)
        self.depth_decoder = DepthDecoder(self.encoder.num_ch_enc, scales=range(4))
        
        # Load pre-trained weights if available
        encoder_path = rospy.get_param('~encoder_weights_path', '')
        depth_decoder_path = rospy.get_param('~depth_weights_path', '')

        try:
            if encoder_path and os.path.exists(encoder_path):
                state_dict = torch.load(encoder_path, map_location=self.device)
                # Filter out unexpected keys
                filtered_dict = {k: v for k, v in state_dict.items() if k in self.encoder.state_dict()}
                self.encoder.load_state_dict(filtered_dict, strict=False)
                rospy.loginfo(f"Loaded encoder weights from {encoder_path}")
            else:
                rospy.logwarn(f"Encoder weights not found at {encoder_path}")

            if depth_decoder_path and os.path.exists(depth_decoder_path):
                self.depth_decoder.load_state_dict(torch.load(depth_decoder_path, map_location=self.device))
                rospy.loginfo(f"Loaded depth decoder weights from {depth_decoder_path}")
            else:
                rospy.logwarn(f"Depth decoder weights not found at {depth_decoder_path}")
        except Exception as e:
            rospy.logerr(f"Error loading weights: {e}")

        self.encoder.to(self.device).eval()
        self.depth_decoder.to(self.device).eval()

        self.arm_controller = PandaArmController()
        
        rospy.loginfo("VisionArmControl initialized successfully")

    def image_callback(self, data):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            rospy.loginfo(f"Received image. Shape: {cv_image.shape}")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        landmarks = self.estimate_pose(cv_image)
        if landmarks:
            rospy.loginfo("Pose landmarks detected")
            target_pose = self.map_human_pose_to_robot(landmarks)
            if target_pose:
                self.arm_controller.move_arm(target_pose)

        depth_map = self.estimate_depth(cv_image)
        if depth_map is not None:
            rospy.loginfo(f"Depth map generated. Shape: {depth_map.shape}")
            self.publish_depth(depth_map)
            cv2.imshow('Depth Map', depth_map)
        else:
            rospy.logwarn("Failed to generate depth map")

        cv2.imshow('Original', cv_image)
        cv2.waitKey(1)

    def estimate_pose(self, frame):
        results = self.pose_estimator.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            rospy.loginfo("Pose landmarks detected")
        else:
            rospy.loginfo("No pose landmarks detected")
        return results.pose_landmarks.landmark if results.pose_landmarks else None

    def map_human_pose_to_robot(self, landmarks):
        try:
            target_pose = PoseStamped()
            target_pose.header.frame_id = "base_link"  
            target_pose.pose.position.x = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x
            target_pose.pose.position.y = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            target_pose.pose.position.z = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].z

            # Example: Set orientation 
            roll, pitch, yaw = 0.0, 0.0, 0.0
            target_pose.pose.orientation.x, target_pose.pose.orientation.y, target_pose.pose.orientation.z, target_pose.pose.orientation.w = quaternion_from_euler(roll, pitch, yaw)

            return target_pose
        except Exception as e:
            rospy.logwarn(f"Error mapping human pose: {e}")
            return None 

    def estimate_depth(self, image):
        rospy.loginfo("Starting depth estimation")
        try:
            with torch.no_grad():
                input_tensor = self.preprocess(image)
                features = self.encoder(input_tensor)
                outputs = self.depth_decoder(features)
                disp = outputs[("disp", 0)]
                
                disp_resized = torch.nn.functional.interpolate(
                    disp, (image.shape[0], image.shape[1]), mode="bilinear", align_corners=False
                )
                depth_map = 1 / disp_resized.squeeze().cpu().numpy()
                return depth_map
        except Exception as e:
            rospy.logerr(f"Error in depth estimation: {e}")
            return None

    def preprocess(self, img):
        img = cv2.resize(img, (640, 192))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        input_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return input_tensor

    def publish_depth(self, depth_map):
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_map_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

        try:
            depth_msg = self.bridge.cv2_to_imgmsg(depth_map_color, "bgr8") 
            self.depth_pub.publish(depth_msg)
            rospy.loginfo("Depth map published successfully") 
        except CvBridgeError as e:
            rospy.logerr(f"Error publishing depth map: {e}")

    def shutdown_hook(self):
        cv2.destroyAllWindows()

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        vision_arm_control = VisionArmControl()
        vision_arm_control.run()
    except rospy.ROSInterruptException:
        pass
#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

class CameraNode:
    def __init__(self):
        rospy.init_node('camera_node', anonymous=True)
        self.image_pub = rospy.Publisher("/camera/color/image_raw/compressed", CompressedImage, queue_size=10)
        self.cap = cv2.VideoCapture(0)
        rospy.loginfo("Camera node initialized")

    def run(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                rospy.loginfo("Camera captured a frame")
                try:
                    msg = CompressedImage()
                    msg.header.stamp = rospy.Time.now()
                    msg.format = "jpeg"
                    msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
                    self.image_pub.publish(msg)
                    rospy.loginfo("Published compressed image")
                except Exception as e:
                    rospy.logerr(f"Error publishing image: {e}")
            else:
                rospy.logwarn("Failed to capture frame")
            rate.sleep()
        
        self.cap.release()

if __name__ == '__main__':
    try:
        node = CameraNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
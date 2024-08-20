#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image

class CameraNode:
    def __init__(self):
        rospy.init_node('camera_node', anonymous=True)
        self.image_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow('ROS Camera', frame)
                try:
                    img_msg = Image()
                    img_msg.height = frame.shape[0]
                    img_msg.width = frame.shape[1]
                    img_msg.encoding = "bgr8"
                    img_msg.is_bigendian = False
                    img_msg.step = 3 * frame.shape[1]
                    img_msg.data = np.array(frame).tobytes()
                    self.image_pub.publish(img_msg)
                except Exception as e:
                    print(e)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                rospy.loginfo("Failed to capture frame")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = CameraNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

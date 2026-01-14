import hydra
import pyspacemouse
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8MultiArray


class SpaceMouseNode:
    def __init__(self):
        rospy.init_node("spacemouse_node", anonymous=True)
        self.spacemouse = pyspacemouse.open()
        self.pub_twist = rospy.Publisher("spacemouse/twist", Twist, queue_size=10)
        self.pub_buttons = rospy.Publisher("spacemouse/buttons", Int8MultiArray, queue_size=10)
        self.rate = rospy.Rate(100)

    def run(self):
        while not rospy.is_shutdown():
            state = self.spacemouse.read()
            if state is None:
                self.rate.sleep()
                continue

            msg = Twist()

            msg.linear.x = float(state.x)
            msg.linear.y = float(state.y)
            msg.linear.z = -float(state.z)
            msg.angular.y = float(state.roll)
            msg.angular.x = -float(state.pitch)
            msg.angular.z = float(state.yaw)

            buttons_msg = Int8MultiArray(data=list(state.buttons))

            self.pub_buttons.publish(buttons_msg)
            self.pub_twist.publish(msg)
            self.rate.sleep()


def main():
    try:
        node = SpaceMouseNode()
        rospy.loginfo("SpaceMouse node started")
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.spacemouse.close()
        rospy.loginfo("SpaceMouse node stopped")


if __name__ == "__main__":
    main()

import hydra
import rospy
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger, TriggerResponse
from xlib.algo.filter import MovingAverageFilter
from xlib.device.sensor import Ft300sSensor


class FtSensorNode:
    def __init__(self, config):
        rospy.init_node('ft_sensor_node', anonymous=True)
        self.ft_sensor = Ft300sSensor(config.ft_port, config.ft_timeout, config.ft_zero_reset)
        self.filter = MovingAverageFilter(config.ft_filter_window_size)
        self.pub = rospy.Publisher('ft_filtered', WrenchStamped, queue_size=10)
        self.rate = rospy.Rate(config.ft_read_freq)

        # Reset service
        self.reset_service = rospy.Service('ft_sensor_reset', Trigger, self._reset_callback)

    def _reset_callback(self, req):
        """Reset the F/T sensor bias."""
        try:
            self.ft_sensor.reset_bias()
            self.filter.reset()
            rospy.loginfo("F/T sensor bias reset successfully")
            return TriggerResponse(success=True, message="F/T sensor bias reset successfully")
        except Exception as e:
            rospy.logerr(f"Failed to reset F/T sensor bias: {e}")
            return TriggerResponse(success=False, message=str(e))

    def run(self):
        while not rospy.is_shutdown():
            ft_value = self.ft_sensor.get_force_torque()
            if ft_value is None:
                self.rate.sleep()
                continue

            self.filter.update(ft_value)
            if self.filter.size < self.filter.window_size:
                self.rate.sleep()
                continue

            filtered = self.filter.output

            # 发布 Wrench 消息
            msg = WrenchStamped()
            msg.header.stamp = rospy.Time.now()
            msg.wrench.force.x = float(filtered[0])
            msg.wrench.force.y = float(filtered[1])
            msg.wrench.force.z = float(filtered[2])
            msg.wrench.torque.x = float(filtered[3])
            msg.wrench.torque.y = float(filtered[4])
            msg.wrench.torque.z = float(filtered[5])

            self.pub.publish(msg)
            self.rate.sleep()

@hydra.main(config_path="../config/env", config_name="ft_sensor", version_base=None)
def main(config):
    try:
        node = FtSensorNode(config)
        rospy.loginfo("FT Sensor node started")
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.ft_sensor.close()
        rospy.loginfo("FT Sensor node stopped")


if __name__ == '__main__':
    main()

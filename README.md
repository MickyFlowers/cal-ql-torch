# cal-ql-torch

start camera node:

roslaunch realsense2_camera rs_camera.launch enable_depth:=false color_height:=480 color_width:=640 color_fps:=30

start ft_sensor_node:
python3 -m env.ft_sensor_node


start data_collection:
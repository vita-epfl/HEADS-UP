# HEADS-UP

This is a ROS2 package for processing dynamic object positions from a tracker and predicting the future positions with a Kalman filter. It can be used for pedestrian trajectory prediction.

After setting up a ROS2 workspace and building the dynamic_costmap package and the interfaces package you can use `ros2 run dynamic_costmap dynamic_costmap`.

The node subscribes to `/dynamic_objects` to get the object coordinates and publishes a string message to `/chatter_dynam` when a collision is detected.

This package is tested with ROS2 Humble and the ZED camera.

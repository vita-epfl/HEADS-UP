import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R
from global_view_interfaces.msg import DynamObjs
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from time import time
from std_msgs.msg import String
import cv2

SEMILOCAL = False # parameter to control whether we use the translation part of odom

USE_IMU = True # if True, then SEMILOCAL is ignored

MAX_TIME_SECONDS = 0.5

PUBLISH_COSTMAP = False

RADIUS = 1.0

print(f"{SEMILOCAL=} {USE_IMU=} {MAX_TIME_SECONDS=} {PUBLISH_COSTMAP=} {RADIUS=}")


class KalmanFilterOpenCV():

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                             [0, 1, 0, 1], 
                                             [0, 0, 1, 0], 
                                             [0, 0, 0, 1]], np.float32)

    def predict(self, coordinateX, coordinateY):
        measured = np.array([[np.float32(coordinateX)], [np.float32(coordinateY)]])

        self.kf.correct(measured)

        predicted = self.kf.predict()

        x, y = float(predicted[0]), float(predicted[1])
        return x, y


def numpy_to_occupancy_grid(arr, info=None, timestamp=None):
    if not len(arr.shape) == 2:
        raise TypeError('Array must be 2D')

    grid = OccupancyGrid()
    grid.header.frame_id = 'base_link'
    grid.header.stamp = timestamp

    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.data

    data = arr.ravel().astype(np.int8)
    grid.data = data.tolist()
    grid.info = info or MapMetaData()
    grid.info.height = arr.shape[0]
    grid.info.width = arr.shape[1]

    return grid


class DynamicCostmapNode(Node):

    def __init__(self):
        super().__init__('dynamic_costmap')

        qos_profile = QoSProfile(
           reliability=QoSReliabilityPolicy.BEST_EFFORT,
           history=QoSHistoryPolicy.KEEP_LAST,
           depth=1,
        )

        self.tracks = {}

        self.size_meters = 12
        self.resolution = 0.25
        map_size_cells = int(self.size_meters / self.resolution)

        self.local_bounded_costmap = np.zeros((map_size_cells, map_size_cells))

        self.last_pose = None

        if USE_IMU:
            self.imu_sub = self.create_subscription(
                Imu, '/zed/zed_node/imu/data', self.imu_callback, qos_profile)
        else:
            self.odom_sub = self.create_subscription(
                Odometry, '/zed/zed_node/odom', self.odom_callback, qos_profile)        

        self.detect_subscription = self.create_subscription(
            DynamObjs,
            '/dynamic_objects',
            self.dynamic_objects_detection_callback,
            qos_profile)

        self.costmap_pub = self.create_publisher(
            OccupancyGrid, '/costmap/dynamic_costmap', qos_profile)

        self.alert_string = self.create_publisher(
            String, '/chatter_dynam', 10)

        self.last_pose = None


    def predict_next_values(self, past_positions: list[list], future_steps: int) -> tuple[list[float], list[float]]:
        """
        function that gets a list of past positions

        each position is a list that contains [x, y, id, time]

        returns a tuple of lists, predicted_x and predicted_y
        """

        past_positions = [position[:2] for position in past_positions if np.isfinite(
            position[0]) and np.isfinite(position[1])]
        

        kf = KalmanFilterOpenCV()
        for coord in past_positions:
            predicted = kf.predict(coord[0], coord[1])

        predicted_x = []
        predicted_y = []
        predicted_x.append(predicted[0])
        predicted_y.append(predicted[1])

        for _ in range(future_steps - 1):
            predicted = kf.predict(predicted[0], predicted[1])
            predicted_x.append(predicted[0])
            predicted_y.append(predicted[1])

        return predicted_x, predicted_y


    def odom_callback(self, odom):
        self.last_pose = odom.pose.pose

        if SEMILOCAL:
            # ignore the translation part of the odometry
            self.last_pose.position.x = 0.0
            self.last_pose.position.y = 0.0
            self.last_pose.position.z = 0.0

    def imu_callback(self, imu):
        self.last_pose = Pose(
            position=Point(x=0.0, y=0.0, z=0.0),
            orientation=imu.orientation)

    def dynamic_objects_detection_callback(self, msg):
        if self.last_pose is not None:
            self.map_dynamic_objects(msg)

    def global_coordinates(self, obj, pose):
        coordinates = np.array([obj.x, obj.y, obj.z])

        positional_tf = np.array(
            [pose.position.x, pose.position.y, pose.position.z])
        quat = pose.orientation
        rotational_tf = R.from_quat([quat.x, quat.y, quat.z, quat.w])

        converted = positional_tf + \
            np.dot(coordinates, rotational_tf.as_matrix().T)

        return converted

    def draw_and_publish_costmap(self, points_to_visualize, pose, stamp):
        output = np.zeros((self.local_bounded_costmap.shape[0], self.local_bounded_costmap.shape[1]))
        possible_weights = {p['weight'] for p in points_to_visualize}
        for weight in possible_weights:
            x_values = [p['x'] for p in points_to_visualize if p['weight'] == weight]
            y_values = [p['y'] for p in points_to_visualize if p['weight'] == weight]
            output += weight * self.get_costmap(np.array(x_values), np.array(y_values), pose)

        map_output = np.rot90(output)
        map_output = np.where(
                map_output >= 1, map_output, 0)

        info = MapMetaData()
        info.height = map_output.shape[0]
        info.width = map_output.shape[1]
        info.resolution = 0.25
        info.map_load_time = stamp
        info.origin = Pose()
        info.origin.position.x = -6.0
        info.origin.position.y = -6.0
        occupancy_grid = numpy_to_occupancy_grid(
            map_output, info=info, timestamp=stamp)
        self.costmap_pub.publish(occupancy_grid)

    def get_costmap(self, pointx, pointy, pose):
        """
        returns the costmap array with values 0 or 1
        """

        converted = np.zeros((pointx.shape[0], 3))
        converted[:, 0] = pointx
        converted[:, 1] = pointy

        positional_tf = np.array(
            [pose.position.x, pose.position.y, pose.position.z])

        # we now have the converted point cloud to the global frame
        result = converted - positional_tf

        x = result[:, 0]
        y = result[:, 1]
        new_costmap_before_filter, xedges, yedges = np.histogram2d(
            x, y, bins=(np.arange(-6, 6+0.25, 0.25), np.arange(-6, 6+0.25, 0.25)))

        current_costmap = np.where(
            new_costmap_before_filter > 0, 1, 0)

        output = np.fliplr(current_costmap)
        return output
    
    def alert_obj(self, prediction):
        if check_collision(trajectory=prediction, slam_pose={'x': self.last_pose.position.x, 'y': self.last_pose.position.y}):
            return 'pred'

    def map_dynamic_objects(self, dynamic_objects_msg):
        start = time()
        pose = self.last_pose
        current_time = dynamic_objects_msg.header.stamp.sec + dynamic_objects_msg.header.stamp.nanosec * 1e-9

        for obj in dynamic_objects_msg.objects:
            if obj.x == 0.0 and obj.y == 0.0:
                print("object coordinates x and y are 0.0, there might be a problem with the detection")

            coord = self.global_coordinates(obj, pose)

            if str(obj.id) not in self.tracks:
                self.tracks[str(obj.id)] = []

            if len(self.tracks[str(obj.id)]) > 50:
                del self.tracks[str(obj.id)][0]

            self.tracks[str(obj.id)].append(
                [coord[0], coord[1], obj.id, current_time])

        for key in list(self.tracks.keys()): # can change size during iteration
            latest_timestamp_for_this_pedestrian = self.tracks[key][-1][3]
            if current_time - latest_timestamp_for_this_pedestrian > MAX_TIME_SECONDS and key in self.tracks:
                del self.tracks[key]


        current_frame_ids = {str(obj.id) for obj in dynamic_objects_msg.objects}

        points_to_visualize = []

        for key in list(self.tracks.keys()):
            # only show pedestrians that are detected in this frame
            if key in current_frame_ids and len(self.tracks[key]): 

                last_occurence = self.tracks[key][-1]
                points_to_visualize.append({'id': key, 'x': last_occurence[0], 'y': last_occurence[1], 'weight': 50, 'type': 'current'})

                seconds_timestamps = np.array([item[3] for item in self.tracks[key]])
                if len(self.tracks[key]) > 50:
                    del self.tracks[key][0]
                if len(seconds_timestamps) >= 2:
                    time_range_seconds = seconds_timestamps.max() - seconds_timestamps.min()
                    if time_range_seconds > 0:
                        historical_fps = (len(seconds_timestamps) - 1) / time_range_seconds
                        steps_for_3_seconds = round(historical_fps * 3)
                        print(f"historical points for this pedestrian {time_range_seconds=:.2f} {len(seconds_timestamps)=} or {historical_fps=:.2f} {steps_for_3_seconds=}")
                        
                        next_predictions_x, next_predictions_y = self.predict_next_values(self.tracks[key], future_steps=steps_for_3_seconds)
                        for x, y in zip(next_predictions_x, next_predictions_y):
                            points_to_visualize.append({'id': key, 'x': x, 'y': y, 'weight': 40, 'type': 'prediction'})

                        alert_str = self.alert_obj(prediction={'x': np.array(next_predictions_x), 'y': np.array(next_predictions_y)})
                        if alert_str:
                            str_msg = String()
                            str_msg.data = alert_str
                            self.alert_string.publish(str_msg)
                            print('sent alert')


        if PUBLISH_COSTMAP:
            current_and_future_points = [point for point in points_to_visualize if point['type'] in ['current', 'prediction']]
            self.draw_and_publish_costmap(current_and_future_points, pose, stamp=dynamic_objects_msg.image_header.stamp)

        tracked_objects = len(self.tracks)
        max_history_points = max((len(x) for x in self.tracks.values()), default=0)
        if tracked_objects > 50 or max_history_points > 40:
            print(f'warning, too many tracked objects {tracked_objects=}, {max_history_points=}')

        print(f"processing time {time() - start:.2f} sec")


def check_collision(trajectory, slam_pose):
    points_inside_circle = (trajectory['x'] - slam_pose['x'])**2 + (trajectory['y'] - slam_pose['y'])**2 <= RADIUS**2
    return np.any(points_inside_circle)


def main(args=None):
    rclpy.init(args=args)
    dynamic_costmap_node = DynamicCostmapNode()
    rclpy.spin(dynamic_costmap_node)
    dynamic_costmap_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

import numpy

from numba import float32, int32, jit
from numba.experimental import jitclass

from basic_math_tools import clip, float_to_str


distance_precision = 2
velocity_precision = 3
angle_precision = 1
angular_velocity_precision = 1
thrust_precision = 1


position_command_spec = dict(x=float32,
                             y=float32,
                             z=float32,
                             yaw=float32)

velocity_command_spec = dict(vx=float32,
                             vy=float32,
                             vz=float32,
                             yaw_rate=float32)

attitude_command_spec = dict(roll=float32,
                             pitch=float32,
                             yaw_rate=float32,
                             thrust=int32)

coordinates_3d_spec = dict(x=float32,
                           y=float32,
                           z=float32)

coordinates_angles_6d_spec = dict(x=float32,
                                  y=float32,
                                  z=float32,
                                  roll=float32,
                                  pitch=float32,
                                  yaw=float32)


@jitclass(position_command_spec)
class PositionCommand:
    """
    :param x: X Coordinate in the world frame (m)
    :param y: Y Coordinate in the world frame (m)
    :param z: Z Coordinate in the world frame (m)
    :param yaw: Yaw Angle (°)
    """
    def __init__(self, x: float = 0, y: float = 0, z: float = 0, yaw: float = 0):
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.yaw: float = yaw

    def __str__(self):
        return ('PositionCommand(' +
                'x=' + float_to_str(self.x, distance_precision) + 'm, ' +
                'y=' + float_to_str(self.y, distance_precision) + 'm, ' +
                'z=' + float_to_str(self.z, distance_precision) + 'm, ' +
                'yaw=' + float_to_str(self.yaw, angle_precision) + '°)')

    def saturate(self):
        pass


@jitclass(velocity_command_spec)
class VelocityCommand:
    """
    :param vx: X Velocity in the world frame (m/s), -1m/s <= vx <= 1m/s
    :param vy: Y Velocity in the world frame (m/s), -1m/s <= vy <= 1m/s
    :param vz: Z Velocity in the world frame (m/s), -1m/s <= vz <= 1m/s
    :param yaw_rate: Yaw rate (°/s), -180°/s <= yaw_rate <= 180°/s
    """
    def __init__(self, vx: float = 0, vy: float = 0, vz: float = 0, yaw_rate: float = 0):
        self.vx: float = vx
        self.vy: float = vy
        self.vz: float = vz
        self.yaw_rate: float = yaw_rate

    def __str__(self):
        return ('VelocityCommand(' +
                'vx=' + float_to_str(self.vx, velocity_precision) + 'm/s, ' +
                'vy=' + float_to_str(self.vy, velocity_precision) + 'm/s, ' +
                'vz=' + float_to_str(self.vz, velocity_precision) + 'm/s, ' +
                'yaw_rate=' + float_to_str(self.yaw_rate, angular_velocity_precision) + '°/s)')

    def saturate(self):
        self.vx = clip(self.vx, float32(-1), float32(1))
        self.vy = clip(self.vy, float32(-1), float32(1))
        self.vz = clip(self.vz, float32(-1), float32(1))
        self.yaw_rate = clip(self.yaw_rate, float32(-180), float32(180))
        

@jitclass(attitude_command_spec)
class AttitudeCommand:
    """
    :param roll: Roll angle (°), -20° <= roll <= 20°
    :param pitch: Pitch angle (°), -20° >= pitch <= 20°
    :param yaw_rate: Yaw rate (°/s), -180°/s <= yaw_rate <= 180°/s
    :param thrust: Thrust (PWM integer), 0 <= thrust <= 65535
    """
    def __init__(self, roll: float = 0, pitch: float = 0, yaw_rate: float = 0, thrust: int = 0):
        self.roll: float = roll
        self.pitch: float = pitch
        self.yaw_rate: float = yaw_rate
        self.thrust: int = thrust

    def __str__(self):
        return ('AttitudeCommand(' +
                'roll=' + float_to_str(self.roll, angle_precision) + '°, ' +
                'pitch=' + float_to_str(self.pitch, angle_precision) + '°, ' +
                'yaw_rate=' + float_to_str(self.yaw_rate, angular_velocity_precision) + '°/s, ' +
                'thrust=' + float_to_str(self.thrust, thrust_precision) + '[PWM])')

    def saturate(self):
        self.roll = clip(self.roll, float32(-20), float32(20))
        self.pitch = clip(self.pitch, float32(-20), float32(20))
        self.thrust = clip(self.thrust, int32(0), int32(65535))
        self.yaw_rate = clip(self.yaw_rate, float32(-180), float32(180))


@jitclass(coordinates_3d_spec)
class Coordinates3D:
    """
    :param x: X Coordinate in the world frame (m)
    :param y: Y Coordinate in the world frame (m)
    :param z: Z Coordinate in the world frame (m)
    """
    def __init__(self, x: float32 = 0, y: float32 = 0, z: float32 = 0):
        self.x: float32 = x
        self.y: float32 = y
        self.z: float32 = z

    def __str__(self):
        return ('Coordinates(' +
                'x=' + float_to_str(self.x, distance_precision) + 'm, ' +
                'y=' + float_to_str(self.y, distance_precision) + 'm, ' +
                'z=' + float_to_str(self.z, distance_precision) + 'm)')


@jitclass(coordinates_angles_6d_spec)
class CoordinatesAngles6D:
    """
    :param x: X Coordinate in the world frame (m)
    :param y: Y Coordinate in the world frame (m)
    :param z: Z Coordinate in the world frame (m)
    :param roll: Roll angle (°)
    :param pitch: Pitch angle (°)
    :param yaw: Yaw angle (°)
    """
    def __init__(self, x: float32 = 0, y: float32 = 0, z: float32 = 0, roll: float32 = 0, pitch: float32 = 0, yaw: float32 = 0):
        self.x: float32 = x
        self.y: float32 = y
        self.z: float32 = z
        self.roll: float32 = roll
        self.pitch: float32 = pitch
        self.yaw: float32 = yaw

    def log_coordinates(self) -> list[float32]:
        log_table = [self.x, self.y, self.z,
                     self.roll, self.pitch, self.yaw]
        return log_table

    def __str__(self):
        return ('Coordinates(' +
                'x=' + float_to_str(self.x, distance_precision) + 'm, ' +
                'y=' + float_to_str(self.y, distance_precision) + 'm, ' +
                'z=' + float_to_str(self.z, distance_precision) + 'm)' +
                ' | Angles(' +
                'roll=' + float_to_str(self.roll, angle_precision) + '°, ' +
                'pitch=' + float_to_str(self.pitch, angle_precision) + '°, ' +
                'yaw=' + float_to_str(self.yaw, angle_precision) + '°)')


@jit
def distance_xy(position_1: Coordinates3D | CoordinatesAngles6D | PositionCommand,
                position_2: Coordinates3D | CoordinatesAngles6D | PositionCommand) -> float32:
    d = numpy.sqrt((position_1.x - position_2.x) ** 2
                   + (position_1.y - position_2.y) ** 2)
    return d


@jit
def distance_xyz(position_1: Coordinates3D | CoordinatesAngles6D | PositionCommand,
                 position_2: Coordinates3D | CoordinatesAngles6D | PositionCommand) -> float32:
    d = numpy.sqrt((position_1.x - position_2.x) ** 2
                   + (position_1.y - position_2.y) ** 2
                   + (position_1.z - position_2.z) ** 2)
    return d


def test_module():
    p1 = Coordinates3D(x=1, y=1, z=1)
    p2 = Coordinates3D()
    d = distance_xyz(p1, p2)
    print(d, type(d))


if __name__ == '__main__':
    test_module()
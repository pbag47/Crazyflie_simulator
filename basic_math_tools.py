import numpy

from numba import float32, jit, optional
from numba.experimental import jitclass


integrator_spec = dict(initial_value=float32,
                       previous_output=float32,
                       previous_timestamp=float32)
@jitclass(integrator_spec)
class Integrator:
    def __init__(self,
                 initial_value: float32,
                 ):
        self.initial_value: float32 = initial_value
        self.previous_output: float32 = self.initial_value
        self.previous_timestamp: float32 = 0

    def integrate(self,
                  input_value: float32,
                  timestamp: float32,
                  ) -> float32:
        delta_t = timestamp - self.previous_timestamp
        output = delta_t * input_value + self.previous_output
        self.previous_timestamp = timestamp
        self.previous_output = output
        return output

    def reset(self,
              timestamp: float32,
              initial_value: optional(float32) = None,
              ):
        if initial_value is not None:
            self.initial_value = initial_value
        self.previous_output = self.initial_value
        self.previous_timestamp = timestamp


differentiator_spec = dict(previous_input=float32,
                           previous_timestamp=float32)
@jitclass(differentiator_spec)
class Differentiator:
    def __init__(self):
        self.previous_input = 0
        self.previous_timestamp = 0

    def differentiate(self,
                      input_value: float32,
                      timestamp: float32,
                      ) -> float32:
        delta_t = timestamp - self.previous_timestamp
        if delta_t > 0:
            output = (input_value - self.previous_input) / delta_t
        else:
            output = 0
        self.previous_input = input_value
        self.previous_timestamp = timestamp
        return output


pid_spec = dict(kp=float32,
                ki=float32,
                kd=float32,
                output_at_steady_state=float32,
                min_output=optional(float32),
                max_output=optional(float32),
                integrator=Integrator.class_type.instance_type,
                differentiator=Differentiator.class_type.instance_type)
@jitclass(pid_spec)
class PID:
    def __init__(self,
                 kp: float32 = 1,
                 ki: float32 = 0.1,
                 kd: float32 = 0.4,
                 output_at_steady_state: float32 = 0,
                 min_output: None | float = None,
                 max_output: None | float = None):
        self.kp: float = kp
        self.ki: float = ki
        self.kd: float = kd
        self.output_at_steady_state = output_at_steady_state
        self.min_output: None | float = min_output
        self.max_output: None | float = max_output
        self.integrator = Integrator(0)
        self.differentiator = Differentiator()

    def pid(self,
            error: float32,
            timestamp: float32,
            ) -> float32:
        # PID
        p = self.kp * error
        i = self.integrator.integrate(self.ki * error, timestamp)
        d = self.kd * self.differentiator.differentiate(error, timestamp)
        output = p + i + d + self.output_at_steady_state

        # Saturation
        output = clip(output, self.min_output, self.max_output)
        # if self.min_output is not None:
        #     if output <= self.min_output:
        #         output = self.min_output
        # if self.max_output is not None:
        #     if output >= self.max_output:
        #         output = self.max_output

        return output

    def reset(self, timestamp: float):
        self.integrator.reset(timestamp)


@jit
def find_min_angle_difference(
        angle_1: float32,
        angle_2: float32,
        ) -> float32:
    """
    :param angle_1: Angle 1 in the world frame (°)
    :param angle_2: Angle 2 in the world frame (°)
    :return: Smallest relative angle sector between Angles 1 and 2 (°)
    """
    angle_1 = angle_1 % 360
    if angle_1 > numpy.pi:
        angle_1 = angle_1 - 360

    angle_2 = angle_2 % 360
    if angle_2 > numpy.pi:
        angle_2 = angle_2 - 360

    yaw_errors = numpy.array([angle_1 - angle_2,
                              angle_1 - angle_2 + 360,
                              angle_1 - angle_2 - 360])
    index = numpy.argmin(numpy.absolute(yaw_errors))
    minimum_difference = yaw_errors[index]
    return minimum_difference


@jit
def clip(value, lower_limit, upper_limit):
    if lower_limit is not None:
        if value < lower_limit:
            value = lower_limit
    if upper_limit is not None:
        if value > upper_limit:
            value = upper_limit
    return value

@jit
def float_to_str(value, precision=2) -> str :
    if numpy.isnan(value) :
        return 'NaN'
    s = str(int(numpy.floor(value))) + '.'
    digits = value % 1
    for _ in range(precision) :
        digits *= 10
        s += str(int(numpy.floor(digits)))
    return s


def test_module():
    corrector = PID()
    current_system_state = 0
    system = Integrator(current_system_state)
    iterations = 50
    for i in range(iterations):
        desired_system_state = 1
        error = desired_system_state - current_system_state
        command = corrector.pid(error, i)
        current_system_state = system.integrate(command, i)
        print(current_system_state)


if __name__ == '__main__':
    test_module()
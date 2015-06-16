
import math

class Point2D(object):

    def __init__(self, x=0, y=0):
        self._x = x
        self._y = y

    def set_x(self, x):
        self._x = x

    def get_x(self):
        return self._x

    def set_y(self, y):
        self._y = y

    def get_y(self):
        return self._y

    def set_pos(self, x=0, y=0):
        self.set_x(x)
        self.set_y(y)

    def get_squared_distance(self, point2d):
        x_delta = self._x - point2d.get_x()
        y_delta = self._y - point2d.get_y()
        sqr_dist = x_delta * x_delta + y_delta * y_delta

        return sqr_dist

    def get_euclidean_distance(self, point2d):
        sqr_dist = self.get_squared_distance(point2d)

        return math.sqrt(sqr_dist)

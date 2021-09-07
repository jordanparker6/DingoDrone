"""A collection of drone drivers and command mappings"""
import sys
import logging
from djitellopy import Tello

log = logging.getLogger(__name__)

class TelloDriver(Tello):
    """A driver implementation for the Tello drone"""
    def connect(self):
        try:
            super().connect()
            log.info(f"Tello driver connected... Battery life {self.get_battery()}%")
        except Exception as e:
            log.error(e)
            sys.exit()

    def __init__(self):
        super().__init__()
        self.command_mappings = {
            "takeoff": self.takeoff,
            "land": self.land,
            "backflip": self.flip_back,
            #"throwfly": self.initiate_throw_takeoff,
            "spin": lambda: self.rotate_clockwise(360)
        }
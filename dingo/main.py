import logging
import time
from djitellopy import Tello

import config
from utils import setup_logging
from drone import DingoDrone
from drone.io.audio import AudioRecorder
from drone.control.speech2command import Speech2Command

setup_logging()
log = logging.getLogger(__name__)

def configure_drone(config):
    controller = Speech2Command()
    driver = Tello()
    drone = DingoDrone(driver=driver, controller=controller, config=config)
    return drone

def main():
    drone = configure_drone(config)
    drone.start()

if __name__ == "__main__":
    main()

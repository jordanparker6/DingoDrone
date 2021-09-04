import logging

from drone.control.speech2command import Speech2Command

log = logging.getLogger(__name__)

class DingoDrone:
    """Main drone class"""
    def __init__(self, driver, controller, **config):
        super().__init__()
        self.config = config
        self.driver = driver
        self.controller = controller
        self.command_mappings = {
            "takeoff": driver.takeoff,
            "land": driver.land,
        }

    def start(self):
        log.info("Starting DingoDrone...")
        #self.connect()
        self.controller.start()
        while True:
            for command in self.controller.commands:
                self.exec(command)

    def exec(self, command):
        try:
            log.info(f"Executing command: ::: {command.name}")
            self.command_mappings[command.name]()
        except KeyError:
            log.error("Command not implemented", exec_info=True, extra={ "name": command.name })

    def capture(self, show=True):
        reader = self.get_frame_read()
        frame = reader.frame
        img = cv2.resize(frame, (self.config.image.width, self.config.image.height))
        if show:
            cv2.imshow("DingoFrame", img)
        return img

    def stop():
        self.controller.stop()
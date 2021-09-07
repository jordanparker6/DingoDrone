import logging

from drone.io.video import VideoRecorder

log = logging.getLogger(__name__)

class DingoDrone:
    """Main drone class"""
    def __init__(self, driver, controller, config):
        super().__init__()
        self.config = config
        self.driver = driver
        self.video = VideoRecorder(driver, **config["image"])
        self.controller = controller

    def start(self):
        log.info("Starting DingoDrone...")
        self.driver.connect()
        self.driver.takeoff()
        self.driver.flip_back()
        #self.controller.recorder.plot()
        self.controller.start()
        while True:
            for command in self.controller.commands:
                print(command)
                try:
                    self.exec(command)
                except Exception as e:
                    log.error(e)

    def exec(self, command):
        try:
            log.info(f"Executing command ::: {command.name}")
            self.driver.command_mappings[command.name]()
        except KeyError:
            log.error("Command not implemented", exec_info=True, extra={ "name": command.name })

    def stop(self):
        self.controller.stop()
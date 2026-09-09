import queue
import pyttsx3


class Speaker:
    def __init__(self, rate=160):
        self.queue = queue.Queue()
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.startLoop(False)

    def speak(self, text):
        if not text or not text.strip():
            return
        self.engine.say(text)

    def update(self):
        self.engine.iterate()

    def shutdown(self):
        self.engine.endLoop()
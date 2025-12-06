import sys

class Tee:
    """
    A helper class to redirect stdout/stderr to both a file and the console.
    """
    def __init__(self, file_obj, console_obj):
        self.file = file_obj
        self.console = console_obj
    
    def write(self, message):
        self.file.write(message)
        self.console.write(message)
    
    def flush(self):
        self.file.flush()
        self.console.flush()
    
    def isatty(self):
        return self.console.isatty()

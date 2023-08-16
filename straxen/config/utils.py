import warnings

class URLConfigWarning(Warning):
    def __init__(selfself, message, custom_attribute=None):
        self.message = message
        self.custom_attribute = custom_attribute

    def __str__(self):
        return self.message
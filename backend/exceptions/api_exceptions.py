class APIException(Exception):
    def __init__(self, message):
        self.message = message


class CannotConnect(APIException):
    def __init__(self, message="Cannot connect to the API"):
        super().__init__(message)


class CannotParseByte(APIException):
    def __init__(self, message="Cannot parse the byte"):
        super().__init__(message)

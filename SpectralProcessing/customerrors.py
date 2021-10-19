class FilePathTypeError(Exception):
    """Exception raised for errors in the input file_path.

    Attributes:
        file_path -- file path(s) to the desiered file(s) to read
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def __str__(self):
        return f'Argument file_path is type {type(self.file_path)}, but only accepts type "list" or "string"'

class LableTypeError(Exception):
    """Exception raised for errors in the input sample_ID.

    Attributes:
        sample_ID -- labels for the file(s)
    """

    def __init__(self, sample_ID):
        self.sample_ID = sample_ID

    def __str__(self):
        return f'Argument sample_ID is type {type(self.sample_ID)}, but only accepts type "list" or "string"'

class LabelSizeMissmachError(Exception):
    """
    Exception raised for a size missmach with the file_path and sample_ID
    arguments.

    Attributes:
        file_path -- file path(s) to the desiered file(s) to read
    """

    def __init__(self, file_path, sample_ID):
        self.file_path = file_path
        self.sample_ID = sample_ID

    def __str__(self):
        return f'Arguments file_path and sample_ID must be the same length, but were length {len(self.file_path)} and {len(self.sample_ID)} respectivly'

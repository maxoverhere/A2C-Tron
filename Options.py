import argparse

class Options:
    def __init__(self):
        self.options = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--gui',
                                 help='Use GUI or not',
                                 type=bool,
                                 default=False)
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
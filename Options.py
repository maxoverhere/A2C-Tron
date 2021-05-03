import argparse

class Options:
    def __init__(self):
        self.options = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--gui',
                                 help='Use GUI or not',
                                 type=bool,
                                 default=False)

        self.parser.add_argument('--epochs',
                                 help='Epochs to train',
                                 type=int,
                                 default=50000)
        
        self.parser.add_argument('--model_name',
                                 help='Name of the model',
                                 type=str,
                                 default='default')

        self.parser.add_argument('--model_type',
                                 help='DQN or A2C?',
                                 type=str,
                                 choices=['DQN', 'A2C'],
                                 default='A2C')

        self.parser.add_argument('--depth',
                                 help='Range of depth sensor for the bot',
                                 type=int,
                                 default=2)



    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
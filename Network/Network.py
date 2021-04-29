import torch
import torch.nn as nn
import torch.nn.functional as F

class TronNet(nn.Module):
    def __init__(self):
        super(TronNet, self).__init__()

        ## For Board
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 2, 3, padding=1)

        ## For DLC
        self.fc0 = nn.Linear(4, 32)
    
        ## For combined Board & DLC
        self.fc1 = nn.Linear(16 * 16 * 2 + 32, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024,  64)
        self.fc4 = nn.Linear( 64,   4)
        self.critic = nn.Linear(64, 1)

    ## Board, dynamic game board size matrix
    ## dlc  , any vector of size 4
    def forward(self, board, dlc):
        # board = torch.zeros_like(board)
        board = F.relu(self.conv1(board))
        board = F.relu(self.conv2(board))
        board = F.interpolate(board, size=(16, 16), 
                              mode='bicubic', 
                              align_corners=False)            
        board = board.view(-1, 2 * 16 * 16)
        dlc = F.relu(self.fc0(dlc))
        combined = torch.cat([board, dlc], dim=-1)        
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        combined = F.relu(self.fc3(combined))

        critic   = self.critic(combined)
        combined = F.softmax(self.fc4(combined), dim=-1)
        
        # print('last', combined, critic)
        return combined, critic




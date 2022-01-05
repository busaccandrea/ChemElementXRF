import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader as DataLoader

class OneElementCNN(nn.Module):
    def __init__(self, input_size, kernel_size=5, n_features=128):
        super(OneElementCNN, self).__init__()

        self.in_size = input_size
        self.hidden = n_features
        self.kernel_size = kernel_size

        # NETWORK DEFINITION
        # first layer
        self.conv_0 = nn.Conv1d(in_channels=1, out_channels=25, kernel_size=self.kernel_size, padding=int(kernel_size/2))
        self.maxpool_0 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.relu = nn.ReLU()

        # second layer
        self.conv_1 = nn.Conv1d(in_channels=25, out_channels=12, kernel_size=self.kernel_size, padding=int(kernel_size/2))
        self.maxpool_1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # third layer
        self.conv_2 = nn.Conv1d(in_channels=12, out_channels=6, kernel_size=self.kernel_size, padding=int(kernel_size/2))
        self.maxpool_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # fourth layer
        self.conv_3 = nn.Conv1d(in_channels=6, out_channels=1, kernel_size=self.kernel_size, padding=int(kernel_size/2))
        
        # last layer
        self.fc = nn.Linear(64, 1)

        self.feature_extractor = nn.Sequential(
            #first layer
            self.conv_0,
            self.maxpool_0,
            self.relu,
            
            # second layer,
            self.conv_1,
            self.maxpool_1,
            self.relu,
            
            # third layer,
            self.conv_2,
            self.maxpool_2,
            self.relu,
            
            # fourth layer,
            self.conv_3,
            self.relu,
            
            # last layer
            self.fc
        )
        # random initialization of weights
        for layer in self.feature_extractor:
            layername = layer.__class__.__name__
            if layername == 'Conv1d':
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, data):
        output0 = self.conv_0(data)
        output0 = self.maxpool_0(output0)
        output0 = self.relu(output0)

        output1 = self.conv_1(output0)
        output1 = self.maxpool_1(output1)
        output1 = self.relu(output1)

        output2 = self.conv_2(output1)
        output2 = self.maxpool_2(output2)
        output2 = self.relu(output2)

        output3 = self.conv_3(output2)
        output3 = self.relu(output3)
        
        output = self.fc(output3)

        return output

class N_ElementCNN(nn.Module):
    def __init__(self, input_size=2048, kernel_size=5):
        super(N_ElementCNN, self).__init__()

        self.in_size = input_size
        self.kernel_size = kernel_size

        # NETWORK DEFINITION
        # first layer
        self.conv_0 = nn.Conv1d(in_channels=1, out_channels=25, kernel_size=self.kernel_size, padding=int(kernel_size/2))
        self.maxpool_1 = nn.MaxPool1d(kernel_size=4, stride=4) #reducing factor: 4
        self.maxpool_2 = nn.MaxPool1d(kernel_size=2, stride=2) #reducing factor: 2
        self.relu = nn.ELU()

        # second layer
        self.conv_1 = nn.Conv1d(in_channels=25, out_channels=12, kernel_size=self.kernel_size, padding=int(kernel_size/2))
        
        # third layer
        self.conv_2 = nn.Conv1d(in_channels=12, out_channels=6, kernel_size=self.kernel_size, padding=int(kernel_size/2))
        
        # fourth layer
        self.conv_3 = nn.Conv1d(in_channels=6, out_channels=1, kernel_size=self.kernel_size, padding=int(kernel_size/2))
        
        # last layer
        self.fc = nn.Linear(128, 10)

        self.feature_extractor = nn.Sequential(
            #first layer
            self.conv_0,
            self.maxpool_1,
            
            # second layer,
            self.conv_1,
            self.maxpool_2,
            
            # third layer,
            self.conv_2,
            self.maxpool_2,
            
            # fourth layer,
            self.conv_3,
            self.maxpool_2,
            
            # last layer
            self.fc,

            self.relu
        )
        # random initialization of weights
        for layer in self.feature_extractor:
            layername = layer.__class__.__name__
            if layername == 'Conv1d':
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, data):
        output0 = self.conv_0(data)
        # output0 = self.maxpool_2(output0)

        output1 = self.conv_1(output0)
        output1 = self.maxpool_2(output1)
        output1 = self.relu(output1)

        output2 = self.conv_2(output1)
        output2 = self.maxpool_2(output2)
        output2 = self.relu(output2)

        output3 = self.conv_3(output2)
        output3 = self.maxpool_2(output3)
        output3 = self.relu(output3)

        
        output = self.fc(output3)
        output = self.relu(output)

        return output
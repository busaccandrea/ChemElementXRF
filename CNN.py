import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader as DataLoader
from torch import flatten

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

class N_ElementCNN_1(nn.Module):
    def __init__(self, n_channels_start=128, n_outputs=10):
        super(N_ElementCNN_1, self).__init__()

        self.n_channels_start = n_channels_start
        self.n_outputs = n_outputs
        self.kernel_size = 5

        # LAYERS DECLARATION
        # first layer
        self.conv_0 = nn.Conv1d(in_channels=1, out_channels=self.n_channels_start, kernel_size=self.kernel_size, padding=int(self.kernel_size/2))
        self.maxpool_1 = nn.MaxPool1d(kernel_size=5, stride=5) #reducing factor: 4
        self.maxpool_2 = nn.MaxPool1d(kernel_size=8, stride=8) #reducing factor: 2
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

        # second layer
        in_channels = self.conv_0.out_channels
        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=self.kernel_size, padding=int(self.kernel_size/2))
        
        # third layer
        in_channels = self.conv_1.out_channels
        self.conv_2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=self.kernel_size, padding=int(self.kernel_size/2))
        
        # fourth layer
        in_channels = self.conv_2.out_channels
        self.conv_3 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=self.kernel_size, padding=int(self.kernel_size/2))
        
        # last layer
        in_size = self.conv_3.out_channels
        self.fc0 = nn.Linear(in_size, int(in_size/2))
        in_size = self.fc0.out_features
        self.fc1 = nn.Linear(in_size, self.n_outputs)

    
    def forward(self, data):
        output0 = self.conv_0(data)
        output0 = self.maxpool_1(output0)

        output1 = self.conv_1(output0)
        output1 = self.maxpool_1(output1)
        output1 = self.elu(output1)

        output2 = self.conv_2(output1)
        output2 = self.maxpool_1(output2)
        output2 = self.elu(output2)

        output3 = self.conv_3(output2)
        output3 = self.maxpool_1(output3)
        output3 = self.elu(output3)

        output3 = flatten(output3, start_dim=1)
        output_fc = self.fc0(output3)
        output_fc = self.fc1(output_fc)
        # output_fc = self.fc2(output_fc)
        output = self.relu(output_fc)
        return output
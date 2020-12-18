import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)
        self.conv3 = nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = 5)
        self.conv4 = nn.Conv2d(in_channels = 24, out_channels = 48, kernel_size = 5)
        self.conv5 = nn.Conv2d(in_channels = 48, out_channels = 96, kernel_size = 5)
        self.conv6 = nn.Conv2d(in_channels = 96, out_channels = 192, kernel_size = 5)
        self.conv7 = nn.Conv2d(in_channels = 192, out_channels = 384, kernel_size = 5)
        self.conv8 = nn.Conv2d(in_channels = 384, out_channels = 768, kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 768 * 2 * 2, out_features = 96)
        self.fc2 = nn.Linear(in_features = 96, out_features = 48)
        self.fc3 = nn.Linear(in_features = 48, out_features = 24)
        self.fc4 = nn.Linear(in_features = 24, out_features = 12)
        self.fc5 = nn.Linear(in_features = 12, out_features = 6)
        self.out = nn.Linear(in_features = 6, out_features = 2)

    def forward(self, x):
        # input layer    
        x = x
        
        # convolution layer 1 (hidden layer)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 5, stride = 1, padding = 1)

        # convolution layer 2 (hidden layer)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 5, stride = 1, padding = 1)            
                
        # convolution layer 3 (hidden layer)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 5, stride = 1, padding = 1)

        # convolution layer 4 (hidden layer)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 5, stride = 1, padding = 1)

        # convolution layer 5 (hidden layer)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 5, stride = 1, padding = 1)

        # convolution layer 6 (hidden layer)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 5, stride = 1, padding = 1)
        
        # convolution layer 7 (hidden layer)
        x = self.conv7(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 5, stride = 1, padding = 1)

        # convolution layer 8 (hidden layer)
        x = self.conv8(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 5, stride = 1, padding = 1)

        # linear layer 1 (hidden layer)
        x = x.reshape(-1, 768 * 2 * 2) # flatening / reshaping operation
        x = self.fc1(x)
        x = F.relu(x)

        # linear layer 2 (hidden layer)
        x = self.fc2(x)
        x = F.relu(x)

        # linear layer 3 (hidden layer)
        x = self.fc3(x)
        x = F.relu(x)

        # linear layer 4 (hidden layer)
        x = self.fc4(x)
        x = F.relu(x)

        # linear layer 5 (hidden layer)
        x = self.fc5(x)
        x = F.relu(x)

        # linear layer  (output layer)
        x = self.out(x)
        #x = F.softmax(x, dim = 1)

        
        return x


class Predictions():
    def __init__(self):
        pass

    def Correct(self, predictions, labels):
        return predictions.argmax(dim = 1).eq(labels).sum().item()

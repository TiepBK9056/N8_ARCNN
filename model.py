import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionalSpeechPredictor(nn.Module):
    def __init__(self, hparams):
        super(EmotionalSpeechPredictor, self).__init__()

        self.num_classes = len(hparams.data.classes)
        self.in_channel = hparams.model.in_channel
        self.num_kernel1 = hparams.model.num_kernel1
        self.num_kernel2 = hparams.model.num_kernel2
        self.cell_units = hparams.model.cell_units
        self.hidden_dim_cnn = hparams.model.hidden_dim_cnn
        self.hidden_dim_fc = hparams.model.hidden_dim_fc
        self.kernel_size = hparams.model.kernel_size
        self.max_pooling_size = hparams.model.max_pooling_size
        self.dropout_prob = hparams.model.dropout_prob
        self.num_filter_bank = hparams.data.num_filter_bank
        self.num_layers_lstm = hparams.model.num_layers_lstm
        self.max_length = hparams.data.max_length

        # Define CNN layer [filter_height, filter_width, in_channels, out_channels]
        self.conv1 = nn.Conv2d(self.in_channel, self.num_kernel1, self.kernel_size, padding='same')     # [5, 3, 3, 128]
        self.conv2 = nn.Conv2d(self.num_kernel1, self.num_kernel2, self.kernel_size, padding='same')    # [5, 3, 128, 256]
        self.conv3 = nn.Conv2d(self.num_kernel2, self.num_kernel2, self.kernel_size, padding='same')    # [5, 3, 256, 256]
        self.conv4 = nn.Conv2d(self.num_kernel2, self.num_kernel2, self.kernel_size, padding='same')    # [5, 3, 256, 256]
        self.conv5 = nn.Conv2d(self.num_kernel2, self.num_kernel2, self.kernel_size, padding='same')    # [5, 3, 256, 256]
        self.conv6 = nn.Conv2d(self.num_kernel2, self.num_kernel2, self.kernel_size, padding='same')    # [5, 3, 256, 256]

        if self.num_filter_bank % self.max_pooling_size[1] != 0:
            raise ValueError("{} is not divisible by {}".format(
                self.num_filter_bank, self.max_pooling_size))
        
        self.input_linear_dim = (self.num_filter_bank // self.max_pooling_size[1]) * self.num_kernel2
        self.linear_cnn_to_lstm = nn.Linear(self.input_linear_dim, self.hidden_dim_cnn)      # [(40//2)*256, 768]
        self.bn = nn.BatchNorm1d(self.hidden_dim_cnn)

        self.relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout2d(self.dropout_prob)
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=self.hidden_dim_cnn, hidden_size=self.cell_units,
                              batch_first=True, num_layers=self.num_layers_lstm, bidirectional=True)

        # Define Attention layer
        self.a_fc1 = nn.Linear(2*self.cell_units, 1)
        self.a_fc2 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # Define Fully-Connected layer
        self.fc1 = nn.Linear(2*self.cell_units, self.hidden_dim_fc)
        self.fc2 = nn.Linear(self.hidden_dim_fc, self.num_classes)

    def forward(self, x):
        # x -> (B, C, T, F)
        # CNN
        layer1 = self.relu(self.conv1(x))
        layer1 = F.max_pool2d(layer1, kernel_size=self.max_pooling_size)
        layer1 = self.dropout(layer1)

        layer2 = self.relu(self.conv2(layer1))
        layer2 = self.dropout(layer2)

        layer3 = self.relu(self.conv3(layer2))
        layer3 = self.dropout(layer3)

        layer4 = self.relu(self.conv4(layer3))
        layer4 = self.dropout(layer4)

        layer5 = self.relu(self.conv5(layer4))
        layer5 = self.dropout(layer5)

        layer6 = self.relu(self.conv6(layer5))
        layer6 = self.dropout(layer6)               # (B, C, T, F)

        # LSTM
        layer6 = layer6.permute(0, 2, 3, 1)     # (B, T, F, C)
        time_step = self.max_length // self.max_pooling_size[0]
        layer6 = layer6.reshape(-1, time_step, self.input_linear_dim)   # (B, T, F*C)
        layer6 = layer6.reshape(-1, self.input_linear_dim)      # (B*T, F*C)

        linear1 = self.relu(self.bn(self.linear_cnn_to_lstm(layer6)))
        linear1 = linear1.reshape(-1, time_step, self.hidden_dim_cnn)   # (B, T, 768)

        out_lstm, _ = self.lstm(linear1)        # (B, T, 128*2)

        # Attention
        v = self.sigmoid(self.a_fc1(out_lstm))      # (B, T, 1)
        alphas = self.softmax(self.a_fc2(v).squeeze())      # (B, T)
        res_att = (alphas.unsqueeze(2) * out_lstm).sum(axis=1)      # (B, 128*2)

        # Fully-Connected
        fc_1 = self.relu(self.fc1(res_att))
        fc_1 = self.dropout(fc_1)
        logits = self.fc2(fc_1)
        logits = self.softmax(logits)

        return logits
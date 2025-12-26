import torch
import torch.nn as nn

class CRNN_Separator(nn.Module):
    # [수정] num_layers 인자 추가
    def __init__(self, input_channels=2, hidden_size=256, n_bins=512, num_stems=4, num_layers=3):
        super(CRNN_Separator, self).__init__()
        
        self.input_channels = input_channels
        self.num_stems = num_stems 

        # 1. Conv Encoder
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1)) 
        self.relu = nn.ReLU()

        # 2. Bi-LSTM (Bottleneck)
        self.compressed_freq = n_bins // 4
        lstm_input_size = 64 * self.compressed_freq
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers, # [수정] 변수 사용
            batch_first=True,
            bidirectional=True
        )
        
        self.linear = nn.Linear(hidden_size * 2, lstm_input_size)

        # 3. Decoder
        self.up_sample = nn.Upsample(scale_factor=(4, 1), mode='bilinear', align_corners=False)
        self.conv_out = nn.Conv2d(64, input_channels * num_stems, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, ch, freq, time = x.size()
        
        c1 = self.relu(self.bn1(self.conv1(x)))
        p1 = self.pool1(c1)
        
        lstm_in = p1.permute(0, 3, 1, 2).reshape(batch, time, -1)
        lstm_out, _ = self.lstm(lstm_in)
        
        lstm_out = self.linear(lstm_out)
        dec_in = lstm_out.reshape(batch, time, 64, -1).permute(0, 2, 3, 1)
        
        dec_in = dec_in + p1 
        
        u1 = self.up_sample(dec_in) 
        mask = self.conv_out(u1)
        mask = self.sigmoid(mask)
        
        mask = mask.view(batch, self.num_stems, self.input_channels, freq, time)
        return mask
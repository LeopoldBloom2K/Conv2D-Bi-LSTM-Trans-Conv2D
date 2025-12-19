import torch
import torch.nn as nn

class CRNN_Separator(nn.Module):
    def __init__(self, input_channels=1, hidden_size=256, n_bins=1024):
        super(CRNN_Separator, self).__init__()
        
        # 1. Conv Encoder
        # Freq: 1024 -> 256 (MaxPool kernel=4)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1)) 
        self.relu = nn.ReLU()

        # 2. Bi-LSTM (Bottleneck)
        self.compressed_freq = n_bins // 4  # 1024 / 4 = 256
        lstm_input_size = 64 * self.compressed_freq
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=3, # 층을 좀 더 깊게
            batch_first=True,
            bidirectional=True
        )
        
        self.linear = nn.Linear(hidden_size * 2, lstm_input_size)

        # 3. Decoder
        self.up_sample = nn.Upsample(scale_factor=(4, 1), mode='bilinear', align_corners=False)
        self.conv_out = nn.Conv2d(64, input_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)
        batch, ch, freq, time = x.size()
        
        # --- Encoder ---
        c1 = self.relu(self.bn1(self.conv1(x)))
        p1 = self.pool1(c1) # (Batch, 64, Freq/4, Time)
        
        # --- LSTM Pre-processing ---
        # (Batch, 64, Freq', Time) -> (Batch, Time, 64*Freq')
        lstm_in = p1.permute(0, 3, 1, 2).reshape(batch, time, -1)
        
        # --- LSTM ---
        lstm_out, _ = self.lstm(lstm_in)
        
        # --- LSTM Post-processing ---
        lstm_out = self.linear(lstm_out)
        # (Batch, Time, 64*Freq') -> (Batch, 64, Freq', Time)
        dec_in = lstm_out.reshape(batch, time, 64, -1).permute(0, 2, 3, 1)
        
        # Skip Connection (인코더 특징 + LSTM 특징)
        dec_in = dec_in + p1 
        
        # --- Decoder ---
        u1 = self.up_sample(dec_in) # 주파수 4배 복원
        mask = self.conv_out(u1)
        mask = self.sigmoid(mask)
        
        return mask # 0~1 사이의 마스크 반환
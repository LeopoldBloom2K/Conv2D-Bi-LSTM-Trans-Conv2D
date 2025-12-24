import torch
import torch.nn as nn

class CRNN_Separator(nn.Module):
    def __init__(self, input_channels=2, hidden_size=256, n_bins=512, num_stems=4):
        """
        input_channels: 2 (Stereo 입력을 권장합니다)
        n_bins: 512 (n_fft=1024일 때)
        num_stems: 4 (Vocals, Drums, Bass, Other)
        """
        super(CRNN_Separator, self).__init__()
        
        self.input_channels = input_channels
        self.num_stems = num_stems 

        # 1. Conv Encoder
        # Freq: 512 -> 128 (MaxPool kernel=4)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1)) 
        self.relu = nn.ReLU()

        # 2. Bi-LSTM (Bottleneck)
        self.compressed_freq = n_bins // 4  # 512 / 4 = 128
        lstm_input_size = 64 * self.compressed_freq
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=3, 
            batch_first=True,
            bidirectional=True
        )
        
        self.linear = nn.Linear(hidden_size * 2, lstm_input_size)

        # 3. Decoder
        self.up_sample = nn.Upsample(scale_factor=(4, 1), mode='bilinear', align_corners=False)
        
        # [핵심 변경] 출력 채널이 8개가 됩니다. (2 Stereo * 4 Stems = 8)
        self.conv_out = nn.Conv2d(64, input_channels * num_stems, kernel_size=3, padding=1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, 2, Freq, Time)
        batch, ch, freq, time = x.size()
        
        # --- Encoder ---
        c1 = self.relu(self.bn1(self.conv1(x)))
        p1 = self.pool1(c1)
        
        # --- LSTM Pre-processing ---
        lstm_in = p1.permute(0, 3, 1, 2).reshape(batch, time, -1)
        
        # --- LSTM ---
        lstm_out, _ = self.lstm(lstm_in)
        
        # --- LSTM Post-processing ---
        lstm_out = self.linear(lstm_out)
        dec_in = lstm_out.reshape(batch, time, 64, -1).permute(0, 2, 3, 1)
        
        # Skip Connection
        dec_in = dec_in + p1 
        
        # --- Decoder ---
        u1 = self.up_sample(dec_in) 
        
        # 여기서 8채널 출력이 나옵니다.
        mask = self.conv_out(u1)
        mask = self.sigmoid(mask)
        
        # [구조 변경] (Batch, 8, Freq, Time) -> (Batch, 4, 2, Freq, Time)
        # 차원 순서: [배치, 악기(4), 채널(2), 주파수, 시간]
        mask = mask.view(batch, self.num_stems, self.input_channels, freq, time)
        
        return mask
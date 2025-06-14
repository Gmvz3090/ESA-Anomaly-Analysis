import torch
import torch.nn as nn

class AttentionGRUAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=1):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # --- Encoder ---
        enc_out, _ = self.encoder(x)  # [batch, time, hidden_dim]
        attn_weights = torch.softmax(self.attn(enc_out).squeeze(-1), dim=1).unsqueeze(-1)
        context = torch.sum(enc_out * attn_weights, dim=1)  # [batch, hidden_dim]
        z = self.latent(context)  # [batch, latent_dim]

        # --- Decoder ---
        h0 = self.latent_to_hidden(z).unsqueeze(0)  # [1, batch, hidden_dim]
        dec_input = torch.zeros(x.size(0), x.size(1), h0.size(-1), device=x.device)
        dec_out, _ = self.decoder_gru(dec_input, h0)  # [batch, time, hidden_dim]
        out = self.out(dec_out)  # [batch, time, input_dim]
        return out

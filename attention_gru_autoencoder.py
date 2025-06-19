import torch
import torch.nn as nn

class AttentionGRUAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Latent space
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )
        
        # Decoder initialization
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Decoder GRU
        self.decoder = nn.GRU(input_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def encode(self, x):
        """Encode input sequence to latent representation"""
        batch_size, seq_len, _ = x.shape
        
        # Pass through encoder
        enc_out, final_hidden = self.encoder(x) 
        
        # Compute attention weights
        attn_scores = self.attn(enc_out)
        attn_weights = torch.softmax(attn_scores, dim=1) 
        
        # Compute attended context vector
        context = torch.sum(enc_out * attn_weights, dim=1) 
        
        # Project to latent space
        latent = self.to_latent(context) 
        
        return latent, attn_weights, enc_out
    
    def decode(self, latent, target_length, target_input=None):
        """Decode latent representation back to sequence"""
        batch_size = latent.shape[0]
        
        h0 = self.latent_to_hidden(latent).unsqueeze(0)  
        if self.num_layers > 1:
            h0 = h0.repeat(self.num_layers, 1, 1)
        
        if target_input is not None:
            dec_out, _ = self.decoder(target_input, h0)
            output = self.output_proj(dec_out)
        else:
            outputs = []
            current_input = torch.zeros(batch_size, 1, self.input_dim, 
                                      device=latent.device, dtype=latent.dtype)
            hidden = h0
            
            for t in range(target_length):
                dec_out, hidden = self.decoder(current_input, hidden)
                step_output = self.output_proj(dec_out)
                outputs.append(step_output)
                current_input = step_output  # Use output as next input
            
            output = torch.cat(outputs, dim=1)
        
        return output
    
    def forward(self, x):
        """Forward pass through encoder-decoder"""
        batch_size, seq_len, input_dim = x.shape
        
        # Encode
        latent, attn_weights, enc_out = self.encode(x)
        
        if self.training:
            decoder_input = torch.zeros_like(x)
            decoder_input[:, 1:, :] = x[:, :-1, :]  # Shift right
            output = self.decode(latent, seq_len, decoder_input)
        else:
            output = self.decode(latent, seq_len, None)
        
        return output
    
    def get_latent_representation(self, x):
        """Get latent representation and attention weights"""
        with torch.no_grad():
            latent, attn_weights, _ = self.encode(x)
        return latent.cpu().numpy(), attn_weights.squeeze(-1).cpu().numpy()
    
    def reconstruction_error(self, x):
        """Calculate reconstruction error per sample"""
        with torch.no_grad():
            output = self.forward(x)
            mse = ((x - output) ** 2).mean(dim=(1, 2)) 
            return mse.cpu().numpy()


class ImprovedAttentionGRUAutoEncoder(AttentionGRUAutoEncoder):
    """
    Enhanced version with residual connections and layer normalization
    """
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=2):
        super().__init__(input_dim, hidden_dim, latent_dim, num_layers)
        
        self.layer_norm_enc = nn.LayerNorm(hidden_dim)
        self.layer_norm_dec = nn.LayerNorm(hidden_dim)
        self.use_residual = (input_dim == hidden_dim)
        if not self.use_residual and input_dim < hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
            self.use_residual = True
    
    def forward(self, x):
        """Forward pass with residual connections and layer norm"""
        batch_size, seq_len, input_dim = x.shape
        
        enc_out, _ = self.encoder(x)
        enc_out = self.layer_norm_enc(enc_out)
        if self.use_residual:
            if hasattr(self, 'residual_proj'):
                residual = self.residual_proj(x)
            else:
                residual = x
            enc_out = enc_out + residual
        
        attn_scores = self.attn(enc_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(enc_out * attn_weights, dim=1)
        latent = self.to_latent(context)
        
        # Decode
        h0 = self.latent_to_hidden(latent).unsqueeze(0)
        if self.num_layers > 1:
            h0 = h0.repeat(self.num_layers, 1, 1)
        
        if self.training:
            decoder_input = torch.zeros_like(x)
            decoder_input[:, 1:, :] = x[:, :-1, :]
            dec_out, _ = self.decoder(decoder_input, h0)
        else:
            outputs = []
            current_input = torch.zeros(batch_size, 1, self.input_dim, 
                                      device=x.device, dtype=x.dtype)
            hidden = h0
            
            for t in range(seq_len):
                dec_out_step, hidden = self.decoder(current_input, hidden)
                step_output = self.output_proj(dec_out_step)
                outputs.append(step_output)
                current_input = step_output
            
            dec_out = torch.cat(outputs, dim=1)
            output = dec_out
        
        if self.training:
            dec_out = self.layer_norm_dec(dec_out)
            output = self.output_proj(dec_out)
        
        return output

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class DecoderOnlyAudioTransformer(nn.Module):
    def __init__(self, input_dim=128, model_dim=256, num_heads=8, num_layers=6, max_length=1000):
        super(DecoderOnlyAudioTransformer, self).__init__()
        self.model_dim = model_dim
        self.max_length = max_length

        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, model_dim)

        # Positional encoding buffer
        self.register_buffer('positional_encoding', self.create_positional_encoding(max_length, model_dim))

        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=0.1
        )

        # Decoder-only Transformer
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer
        self.output_layer = nn.Linear(model_dim, input_dim)

    def create_positional_encoding(self, max_length, model_dim):
        """Generates sinusoidal positional encodings."""
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))
        pe = torch.zeros(max_length, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def generate_causal_mask(self, seq_len, device):
        """Generates a causal mask to prevent attention to future tokens."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
        return mask

    def forward(self, tgt):
        """
        Args:
            tgt (torch.Tensor): Target sequence (input spectrogram). Shape: (batch_size, seq_len, input_dim).

        Returns:
            output (torch.Tensor): Predicted spectrogram. Shape: (batch_size, seq_len, input_dim).
        """
        # Ensure tgt is (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = tgt.size()
        device = tgt.device

        # Embed and add positional encodings
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(device)
        tgt = self.input_embedding(tgt) + pos_encoding  # (batch_size, seq_len, model_dim)

        # Transpose for transformer input: (seq_len, batch_size, model_dim)
        tgt = tgt.transpose(0, 1)

        # Generate causal mask
        tgt_mask = self.generate_causal_mask(seq_len, device)

        # Transformer decoder forward pass (memory is set to None for decoder-only setup)
        output = self.transformer_decoder(tgt, memory=torch.zeros_like(tgt).to(device), tgt_mask=tgt_mask)

        # Transpose back to (batch_size, seq_len, model_dim)
        output = output.transpose(0, 1)

        # Output layer
        output = torch.sigmoid(self.output_layer(output)) # (batch_size, seq_len, input_dim)

 

        return output
    

    def inference(self, initial_input, max_length=100):
        """
        Auto-regressive inference for audio generation.

        Args:
            initial_input (torch.Tensor): Initial seed input of shape (batch_size, 1, input_dim).
            max_length (int): Maximum sequence length to generate.

        Returns:
            generated (torch.Tensor): Generated sequence of shape (batch_size, max_length, input_dim).
        """
        generated = initial_input

        for _ in range(max_length - generated.size(1)):
            batch_size, seq_len, _ = generated.size()

            # Embed and add positional encodings
            pos_encoding = self.positional_encoding[:, :seq_len, :].to(generated.device)
            tgt_emb = self.input_embedding(generated) + pos_encoding

            # Transpose for transformer input: (seq_len, batch_size, model_dim)
            tgt_emb = tgt_emb.transpose(0, 1)

            # Generate causal mask
            tgt_mask = self.generate_causal_mask(seq_len, generated.device)

            # Transformer decoder forward pass
            output = self.transformer_decoder(tgt_emb, memory=torch.zeros_like(tgt_emb), tgt_mask=tgt_mask)
            next_sample = torch.sigmoid(self.output_layer(output[-1]))

            # Append the new sample to the generated sequence
            next_sample = next_sample.unsqueeze(1)  # (batch_size, 1, input_dim)
            generated = torch.cat((generated, next_sample), dim=1)

            # Stop if the maximum length is reached
            if generated.size(1) >= max_length:
                break

        return generated
    
    def inference_train(self, initial_input, max_length=100):
            """
            Auto-regressive inference for audio generation.

            Args:
                initial_input (torch.Tensor): Initial seed input of shape (batch_size, 1, input_dim).
                max_length (int): Maximum sequence length to generate.

            Returns:
                generated (torch.Tensor): Generated sequence of shape (batch_size, max_length, input_dim).
            """
            generated = initial_input

            for _ in range(max_length - generated.size(1)):
                batch_size, seq_len, _ = generated.size()

                # Embed and add positional encodings
                pos_encoding = self.positional_encoding[:, :seq_len, :].to(generated.device)
                tgt_emb = self.input_embedding(generated) + pos_encoding

                # Transpose for transformer input: (seq_len, batch_size, model_dim)
                tgt_emb = tgt_emb.transpose(0, 1)

                # Generate causal mask
                tgt_mask = self.generate_causal_mask(seq_len, generated.device)

                # Transformer decoder forward pass
                output = self.transformer_decoder(tgt_emb, memory=torch.zeros_like(tgt_emb), tgt_mask=tgt_mask)
                output = torch.sigmoid(self.output_layer(output))

                return output

            return generated

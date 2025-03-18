import torch
import torch.nn as nn
import math

import Project.src.models.transformer as tf


class Transformer(nn.Module):

    def __init__(self,
                 encoder: tf.Encoder,
                 src_embed: tf.InputEmbeddings,
                 src_pos: tf.PositionalEmbeddings,
                 last_linear: tf.LastLinear):
        super().__init__()
        # Set all the components we will need
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.last_linear = last_linear

    # We define an encode() function for the encoder.
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    # We pass the outputs through the final linear layer
    def linear(self, x):
        return self.last_linear(x)

    def forward(self, x, src_mask):

        output = self.encode(x, src_mask)
        output = self.linear(output)

        return output


"""
We will finally create a function that allows us to ensemble everything
together and create the layers of encoders and decoders. This is what the
original paper calls Nx in the figure to the sides of the main figure.
"""


def build_transformer_model(src_vars: int, src_seq_len: int,
                            tgt_seq_len: int, d_model: int = 512, n_layers: int = 6,
                            n_heads: int = 8, dropout: float = 0.1, hidden_size: int = 2048):
    # Make the embedding layers for input and target
    src_embed = tf.InputEmbeddings(d_model, src_vars)

    # Make the positional embeddings for input and target
    # (in practice you can use the same for both src and tgt)
    src_pos = tf.PositionalEmbeddings(d_model, src_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(n_layers):
        encoder_self_attention = tf.MultiHeadAttentionBlock(d_model, n_heads, dropout)
        feed_forward = tf.FeedForwardBlock(d_model, hidden_size, dropout)
        encoder_block = tf.EncoderBlock(encoder_self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    """
    Now we pass those layers as the argument to the main objects.
    Remember that our main classes for encoder and decoder take
    nn.ModuleList as arguments (aka, the layers stacked)
    """
    encoder = tf.Encoder(nn.ModuleList(encoder_blocks))

    last_layer = tf.LastLinear(d_model, tgt_seq_len)

    # Now, build our model using the transformer class we built
    transformer = Transformer(
        encoder=encoder,
        src_embed=src_embed,
        src_pos=src_pos,
        last_linear=last_layer,
    )

    # Now we initialize the parameters with Xavier initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # FINALLY, return the transformer
    return transformer

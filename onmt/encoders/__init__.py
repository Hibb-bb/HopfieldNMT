"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.ggnn_encoder import GGNNEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.hopfield_transformer import HopfieldTransformerEncoder


str2enc = {"ggnn": GGNNEncoder, "rnn": RNNEncoder, "brnn": RNNEncoder,
           "cnn": CNNEncoder, "transformer": TransformerEncoder,
           "mean": MeanEncoder, "hopfield": HopfieldTransformerEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc", "HopfieldTransformerEncoder"]

"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder, TransformerLMDecoder
from onmt.decoders.cnn_decoder import CNNDecoder
from onmt.decoders.hopfield_transformer_ import HopfieldTransformerDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder,
           "transformer_lm": TransformerLMDecoder, "hopfield": HopfieldTransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec", "TransformerLMDecoder", "HopfieldTransformerDecoder"]

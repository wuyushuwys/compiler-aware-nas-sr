from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from .wdsr_b import NAS_MODEL, ModelOutput
from .basic_wdsr_b import BASIC_MODEL


def update_argparser(parser: argparse.ArgumentParser):
    parser.add_argument('--learning_rate', help='Learning rate.', default=0.001, type=float, )
    parser.add_argument('--pretrained', action='store_true', default=False, help='Learning rate.')
    parser.add_argument('--width_search', action='store_true', default=False, help='Width Search.')
    parser.add_argument('--length_search', action='store_true', default=False, help='Length Search.')


    # Decoder
    parser.add_argument('--num_blocks', help='Number of residual blocks in networks.', default=16, type=int)
    parser.add_argument('--num_residual_units', help='Number of residual units in networks.', default=24, type=int)

    #  clip weight
    parser.add_argument('--clip_range', help='weight clip range.', default=None, type=float)
    parser.add_argument('--trainable_clip', action='store_true', default=False, help='trainable clip.')
    parser.add_argument('--clip_quantile_lb', help='weight clip quantile lower bound.', default=None, type=float)
    parser.add_argument('--clip_quantile_ub', help='weight clip quantile upper bound.', default=None, type=float)
    parser.add_argument('--clip_range_tail', help='weight clip range for tail layer.', default=None, type=float)
    parser.add_argument('--clip_range_skip', help='weight clip range for skip layer.', default=None, type=float)
    parser.add_argument('--clip_scale', help='weight clip scale factor for quantile clipping.', default=1.0, type=float)

def get_model(params):
    return eval(params.model_type)(params)

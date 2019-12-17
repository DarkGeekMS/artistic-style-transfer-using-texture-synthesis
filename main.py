import argparse
from utils import utils
from src.style_transfer import style_transfer

def main():
    """
    The main function (argument parsing to be replaced by GUI input, if applicable)
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    # general arguments
    argparser.add_argument(
        '-c', '--content_path',
        metavar='C',
        type=str,
        default='data/content/eagles.jpg',
        help='path to content image (or video)'
    )
    argparser.add_argument(
        '-s', '--style_path',
        metavar='S',
        type=str,
        default='data/style/van_gogh.jpg',
        help='path to style image'
    )
    argparser.add_argument(
        '-is', '--img_size',
        metavar='IS',
        type=int,
        default=400,
        help='maximum image size'
    )
    # fundamental arguments
    argparser.add_argument(
        '-nr', '--num_res',
        metavar='NR',
        type=int,
        default=3,
        help='number of resolution layers'
    )
    argparser.add_argument(
        '-ps', '--patch_sizes',
        metavar='PS',
        type=int,
        nargs='+',
        default=[33,21,13,9],
        help='patch sizes to be used'
    )
    argparser.add_argument(
        '-sg', '--sub_gaps',
        metavar='SG',
        type=int,
        nargs='+',
        default=[28,18,8,5],
        help='subsampling gaps to be used'
    )
    # learning parameters
    argparser.add_argument(
        '-ii', '--irls_iter',
        metavar='II',
        type=int,
        default=3,
        help='number of IRLS iterations'
    )
    argparser.add_argument(
        '-ai', '--alg_iter',
        metavar='AI',
        type=int,
        default=3,
        help='number of update iterations per patch size'
    )
    argparser.add_argument(
        '-r', '--robust_stat',
        metavar='R',
        type=float,
        default=0.8,
        help='robust statistics value'
    )
    argparser.add_argument(
        '-cw', '--content_weight',
        metavar='CW',
        type=float,
        default=5.0,
        help='content weight during fusion'
    )
    # segmentation arguments
    argparser.add_argument(
        '-sm', '--segmentation_mode',
        metavar='SM',
        type=int,
        default=2,
        help='edge segmentation method to be used'
    )
    # color transfer arguments
    argparser.add_argument(
        '-ctm', '--color_transfer_mode',
        metavar='CTM',
        type=str,
        default=1,
        help='color transfer method to be used'
    )
    # denoise parameters
    argparser.add_argument(
        '-dss', '--denoise_sigma_s',
        metavar='DSS',
        type=int,
        default=5,
        help='sigma_s for denoise'
    )
    argparser.add_argument(
        '-dsr', '--denoise_sigma_r',
        metavar='DSR',
        type=float,
        default=0.2,
        help='sigma_r for denoise'
    )
    argparser.add_argument(
        '-di', '--denoise_iter',
        metavar='DI',
        type=int,
        default=1,
        help='number of iterations for denoise'
    )
    args = argparser.parse_args()

    content, style, seg_mask, X = style_transfer(args.content_path, args.style_path, args.img_size, args.num_res, args.patch_sizes, args.sub_gaps, args.irls_iter, \
    args.alg_iter, args.robust_stat, args.content_weight, args.segmentation_mode, args.color_transfer_mode, args.denoise_sigma_s, \
    args.denoise_sigma_r, args.denoise_iter)

    utils.show_images([content, style, seg_mask, X], ["Content", "Style", "Segmentation Mask", "Stylized Image"]) # display results


if __name__ == '__main__':
    main()

import argparse
from timeit import default_timer

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
        default='None',
        help='path to content image (or video)')
    argparser.add_argument(
        '-s', '--style_path',
        metavar='S',
        default='None',
        help='path to style image'
    )   
    argparser.add_argument(
        '-is', '--img_size',
        metavar='IS',
        default=400,
        help='maximum image size'
    ) 
    # fundamental arguments
    argparser.add_argument(
        '-nr', '--num_res',
        metavar='NR',
        default=3,
        help='number of resolution layers'
    ) 
    argparser.add_argument(
        '-ps', '--patch_sizes',
        metavar='PS',
        default=[33,21,13,9,5],
        help='patch sizes to be used'
    ) 
    argparser.add_argument(
        '-sg', '--sub_gaps',
        metavar='SG',
        default=[28,18,8,5,3],
        help='subsampling gaps to be used'
    )
    # learning parameters
    argparser.add_argument(
        '-ii', '--irls_iter',
        metavar='II',
        default=3,
        help='number of IRLS iterations'
    )
    argparser.add_argument(
        '-ai', '--alg_iter',
        metavar='AI',
        default=3,
        help='number of update iterations per patch size'
    )
    argparser.add_argument(
        '-r', '--robust_stat',
        metavar='R',
        default=0.8,
        help='robust statistics value'
    )
    argparser.add_argument(
        '-cw', '--content_weight',
        metavar='CW',
        default=5.0,
        help='content weight during fusion'
    )
    # segmentation arguments
    argparser.add_argument(
        '-sm', '--segmentation_mode',
        metavar='SM',
        default=2,
        help='edge segmentation method to be used'
    )
    # color transfer arguments
    argparser.add_argument(
        '-ctm', '--color_transfer_mode',
        metavar='CTM',
        default=0,
        help='color transfer method to be used'
    )
    # denoise parameters
    argparser.add_argument(
        '-dss', '--denoise_sigma_s',
        metavar='DSS',
        default=20,
        help='sigma_s for denoise'
    )
    argparser.add_argument(
        '-dsr', '--denoise_sigma_r',
        metavar='DSR',
        default=0.17,
        help='sigma_r for denoise'
    )
    argparser.add_argument(
        '-di', '--denoise_iter',
        metavar='DI',
        default=1,
        help='number of iterations for denoise'
    )
    args = argparser.parse_args()
    
    start_time = default_timer() # get start time of stylization
    
    style_transfer(args.content_path, args.style_path, args.img_size, args.num_res, args.patch_sizes, args.sub_gaps, args.irls_iter, \
    args.alg_iter, args.robust_stat, args.content_weight, args.segmentation_mode, args.color_transfer_mode, args.denoise_sigma_s,    \
    args.denoise_sigma_r, args.denoise_iter)

    print("Stylization time = ", default_timer()-start_time, " Seconds")

if __name__ == '__main__':
    main()

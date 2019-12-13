import argparse

from src.style_transfer import style_transfer

def main():
    """
    The main function (argument parsing to be replaced by GUI input, if applicable)
    """
    argparser = argparse.ArgumentParser(description=__doc__)
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
    argparser.add_argument(
        '-nr', '--num_res',
        metavar='NR',
        default=3,
        help='number of resolution layers'
    ) 
    argparser.add_argument(
        '-ps', '--patch_sizes',
        metavar='PS',
        default=[33,21,13,9],
        help='patch sizes to be used'
    ) 
    argparser.add_argument(
        '-sg', '--sub_gaps',
        metavar='SG',
        default=[28,18,8,5],
        help='subsampling gaps to be used'
    )
    argparser.add_argument(
        '-ii', '--irls_iter',
        metavar='II',
        default=3,
        help='number of IRLS iterations'
    )
    argparser.add_argument(
        '-ai', '--alg_iter',
        metavar='AI',
        default=10,
        help='number of update iterations per patch size'
    )
    argparser.add_argument(
        '-r', '--robust_stat',
        metavar='R',
        default=0.8,
        help='robust statistics value'
    )
    args = argparser.parse_args()
    style_transfer(args.content_path, args.style_path, args.img_size, args.num_res, args.patch_sizes, args.sub_gaps, args.irls_iter, \
    args.alg_iter, args.robust_stat)

if __name__ == '__main__':
    main()

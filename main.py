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
        default='None',
        help='maximum image size'
    ) 
    argparser.add_argument(
        '-nr', '--num_res',
        metavar='NR',
        default='None',
        help='number of resolution layers'
    ) 
    argparser.add_argument(
        '-ps', '--patch_sizes',
        metavar='PS',
        default='None',
        help='patch sizes to be used'
    ) 
    argparser.add_argument(
        '-ii', '--irls_iter',
        metavar='II',
        default='None',
        help='number of IRLS iterations'
    )
    args = argparser.parse_args()
    style_transfer(args.content_path, args.style_path, args.img_size, args.num_res, args.patch_sizes, args.irls_iter)

if __name__ == '__main__':
    main()

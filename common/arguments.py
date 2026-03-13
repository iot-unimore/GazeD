"""
Creative Commons Attribution-NonCommercial ShareAlike 4.0 International License  https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--config', default='config/egoexo_train.yaml', 
                        type=str, metavar='PATH')

    # General arguments
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('-cf','--checkpoint-frequency', default=1, type=int, metavar='N',
                        help='create a checkpoint every N epochs')
    parser.add_argument('-l', '--log', default='log/default', type=str, metavar='PATH',
                        help='log file directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--evaluate', default=None, required=True, type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--visualize', action='store_true')

    # Model arguments
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=0.00006, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.993, type=float, metavar='LR', help='learning rate decay per epoch')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N')
    parser.add_argument('-num_proposals', type=int, default=1, metavar='N')
    parser.add_argument('-timesteps', type=int, default=1, metavar='N')

    parser.add_argument('-s', '--save', type=str, default="model.bin", metavar='FILENAME')
    parser.add_argument('-d', '--dataset', type=str, default="EgoExo")
    parser.add_argument('-rt', '--resume_test', type=bool, default=False, metavar='path')
    parser.add_argument('-fr','--freeze',type=bool, default=False)
    parser.add_argument('-sd', '--seed', default=1234, type=int, help='seed')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--image_path', default='', type=str, metavar='FILENAME', help='image for inference')


    args = parser.parse_args()

    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args
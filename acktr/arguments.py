import argparse
import time
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument(
        '--mode', default='train', help='Test trained model or train new model, test | train'
    )
    parser.add_argument(
        '--real', action='store_true', default=False, help='use this flag when doing real evaluation'
    )
    # env
    parser.add_argument(
        '--env_name', default='Bpp-v0', type=str, help='bin packing environment name'
    )
    parser.add_argument(
        '--container_size', nargs='+', default=(10, 10, 10), type=int, help='container size along x, y and z axis'
    )
    parser.add_argument(
        '--item_size_range', nargs='+', default=(2,2,2,5,5,5), type=int, help='the item size range, (min_width, min_length, min_height, max_width, max_length, max_height)'
    )
    # device
    parser.add_argument(
        '--use_cuda', action='store_true', default=False, help='whether to use cuda'
    )
    parser.add_argument(
        '--device', default=0, type=int,  help='device id (default: 0)'
    )
    
    # training
    parser.add_argument(
        '--use_existing_data', action='store_true', default=False, help='whether to use existing data to train'
    )
    parser.add_argument(
        '--item_seq', default='cut1', help='item sequence generators (ignored when testing), cut1|cut2|rs'
    )
    parser.add_argument(
        '--data_name', default='cut_2.pt', help=' the name of dataset, check data_dir for details'
    )
    parser.add_argument(
        '--tensor_dtype', default='float32', type=str, help='the torch tensor datatype'
    )
    parser.add_argument(
        '--enable_rotation', action='store_true', default=False,  help='whether agent can rotate box'
    )
    parser.add_argument(
        '--seed', default=1, type=int,  help='random seed (default: 1)'
    )
    parser.add_argument(
        '--algorithm', default='acktr', type=str,  help='algorithm used, acktr|ppo|a2c'
    )

    # test
    parser.add_argument(
        '--preview', default=1, type=int, help='the item number agent knows (ignored when training)'
    )
    parser.add_argument(
        '--cases', default=100, type=int,  help='the number of sequences used for test (default 100)'
    )

    # saving and logging
    parser.add_argument(
        '--tensorboard', action='store_true', default=False, help='whether use tensorboard to tracing trainning process'
    )
    parser.add_argument(
        '--save_model', action='store_true', default=False,  help='whether to save training model'
    )
    parser.add_argument(
        '--save_dir', default='./saved_models/', help='directory to save agent logs (default: ./saved_models/)'
    )
    parser.add_argument(
        '--save_interval', default=10, type=int,  help='save interval, one save per n updates (default: 100)'
    )
    parser.add_argument(
        '--log_interval', default=10, type=int,  help='log interval, one log per n updates (default: 100)'
    )

    # loading pretrained model
    parser.add_argument(
        '--load_model', action='store_true', default=False,  help='Whether to use trained model'
    )
    parser.add_argument(
        '--load_name', default='default_cut_2.pt', help='default trained model for testing or continuing training'
    )
    parser.add_argument(
        '--load_dir', default='./pretrained_models/', help='directory to load agent logs (default: ./pretrained_models/)'
    )
    
    # rarely tuning
    parser.add_argument(
        '--gamma', default=1.0, type=float,  help='discount factor for rewards (default: 1.0)'
    )
    parser.add_argument(
        '--entropy_coef', default=0.01, type=float,  help='entropy term coefficient (default: 0.01)'
    )
    parser.add_argument(
        '--value_loss_coef', default=0.5, type=float,  help='value loss coefficient (default: 0.5)'
    )
    parser.add_argument(
        '--invalid_coef', default=2, type=float,  help='invalid action possibility term coefficient'
    )
    parser.add_argument(
        '--hidden_size', default=256, type=int,  help='hidden layer cell number (default: 256)'
    )
    parser.add_argument(
        '--learning_rate', default=1e-6, type=float,  help='learning rate for a2c (default: 1e-6)'
    )
    parser.add_argument(
        '--eps', default=1e-5, type=float,  help='RMSprop optimizer epsilon (default: 1e-5)'
    )
    parser.add_argument(
        '--alpha', default=0.99, type=float,  help='RMSprop optimizer apha (default: 0.99)'
    )
    parser.add_argument(
        '--num_processes', default=16, type=int,  help='how many training CPU processes to use (default: 16)'
    )
    parser.add_argument(
        '--num_steps', default=5, type=int,  help='number of forward steps in A2C (default: 5)'
    )

    args = parser.parse_args()

    args.device = "cuda:" + str(args.device) if args.use_cuda else "cpu"
    args.bin_size = args.container_size
    args.pallet_size = args.container_size[0]   # TODO: it assumes the pallet is square?
    args.channel = 4 # channels of CNN: 4 for hmap+next box, 5 for hmap nextbox+truemask
    args.data_type = args.item_seq
    args.test = (args.mode == 'test')

    box_range = args.item_size_range
    box_size_set = []
    for i in range(box_range[0], box_range[3] + 1):
        for j in range(box_range[1], box_range[4] + 1):
            for k in range(box_range[2], box_range[5] + 1):
                box_size_set.append((i, j, k))
    args.box_size_set = box_size_set

    assert args.mode in ['train', 'test']
    if args.mode == 'train' and args.load_model:
        print('Continue training model \"%s\"'%args.load_name)
    if args.mode == 'test' and args.load_model:
        print('Test trained model \"%s\"'%args.load_name)
    if args.mode == 'train' and not args.load_model:
        print('Train new model')
    if args.mode == 'test' and not args.load_model:
        raise Exception('No trained model chosed')
    if args.mode not in ['test', 'train']:
        raise Exception('Unknown option \'%s\''%(args.mode))
    if args.item_seq not in ['cut1', 'rs', 'cut2']:
        raise Exception('Unsupported generator \'%s\''%(args.item_seq))
    
    print("===== PARSED ARGS INFO =====")
    if not args.real:
        print('data_name: ', args.data_name)
        time.sleep(0.5)
    if args.mode == 'train':
        print('item_size_range: ', args.item_size_range)
        time.sleep(0.5)
        print('item_seq: ', args.item_seq)
        time.sleep(0.5)
    print('bin_size: ', args.bin_size)
    time.sleep(0.5)
    print('preview: ', args.preview)
    time.sleep(0.5)
    print('enable_rotation: ', args.enable_rotation)
    print('use_cuda: ', args.use_cuda)
    time.sleep(0.5)
    
    # print('box_size_set length: ', len(box_size_set))
    # print('ITEM SET SIZE: ', box_size_set)
 
    return args

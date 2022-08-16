import common
import importlib
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str, required=True, help='Dataset name.')
    parser.add_argument('--scale', default=None, type=int, required=True, help='Scale Factor')
    parser.add_argument('--mode', default='train', type=str, help='dataset mode')

    args = parser.parse_args()
    dataset_module = importlib.import_module(f'datasets.{args.dataset}' if args.dataset else 'datasets')
    if args.mode == 'train':
        train_dataset = dataset_module.get_dataset(common.modes.TRAIN, args)
        print(f"Training Dataset {args.dataset} Ready")
    elif args.mode == 'eval':
        eval_dataset = dataset_module.get_dataset(common.modes.EVAL, args)
        print(f"Eval Dataset {args.dataset} Ready")
    else:
        raise NotImplemented(f"Mode {args.mode} not implement")
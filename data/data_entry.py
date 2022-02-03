from torch.utils.data import DataLoader
from .call_func import get_call_func_by_name
from .sighan_dataset import SigDataset, SigPromptDataset, SigPromptDatasetFromPkl
from .error_judge_dataset import ErrJudgeDataset, ErrJudgePromptDataset

_type2dataset = {
    "SigDataset": SigDataset,
    "ErrJudgeDataset": ErrJudgeDataset,
    "ErrJudgePromptDataset": ErrJudgePromptDataset,
    "SigPromptDataset": SigPromptDataset,
    "SigPromptDatasetFromPkl":SigPromptDatasetFromPkl
}

def get_loader(dataset, args):
    loader = DataLoader(dataset, args['batch_size'], shuffle=args['shuffle'],
                        collate_fn=get_call_func_by_name(args['call_func']),
                        num_workers=args['num_workers'], pin_memory=True, drop_last=False, prefetch_factor = 2)
    return loader

def get_dataset(args):
    dataset = _type2dataset[args.type](args)
    print('{} samples found in {}'.format(len(dataset), args['text_path']))
    return dataset






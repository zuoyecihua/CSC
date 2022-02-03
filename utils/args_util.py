
from dotted_dict import DottedDict
import yaml
from .common_utils import current_timestamp

def load_args(yaml_path):
    args = DottedDict(yaml.load(open(yaml_path, "r"), Loader=None))
    try:
        args.train_paras.save_dir += "%s" % current_timestamp()
        args.train_paras.val_label_file = args.dataset.test.label_path
        args.train_paras.train_label_file = args.dataset.train.label_path

        args.optimizer.use_swa = args.train_paras.use_swa
    except:
        pass

    return args

if __name__ == '__main__':
    args = {"test":{"test1":2}}
    config = DottedDict(args)
    print(config.test)
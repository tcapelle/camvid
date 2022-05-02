import rich
from rich import print
import wandb
import fastai
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

SEED = 2022

path = untar_data(URLs.CAMVID)

codes = np.loadtxt(path/'codes.txt', dtype=str)
fnames = get_image_files(path/"images")

def label_func(fn): return path/"labels"/f"{fn.stem}_P{fn.suffix}"
class_labels = {i:c for i,c in enumerate(codes)}

def get_preds(learn):
    inp,preds,targs,out = learn.get_preds(with_input=True, with_decoded=True)
    b = tuplify(inp) + tuplify(targs)
    x,y,samples,outs = learn.dls.valid.show_results(b, out, show=False, max_n=36)
    class_labels = {i:code for i,code in enumerate(codes)}
    return samples, outs, preds

def create_table(samples, outs, preds):
    "Creates a wandb table with preds and targets side by side"
    res = []
    table = wandb.Table(columns=["image", "preds", "targets"])
    for (image, label), pred_label in zip(samples, outs):
        img = image.permute(1,2,0)
        table.add_data(wandb.Image(img),
                       wandb.Image(img, masks={"pred":  {'mask_data':  pred_label[0].numpy(), 'class_labels':class_labels}}),
                       wandb.Image(img, masks={"target": {'mask_data': label.numpy(), 'class_labels':class_labels}}))
    return table

#ready for sweeps
config = {"arch": "resnet34",
          "batch_size": 16,
          "loss_func": "CrossEntropyLossFlat",
          "lr": 0.002,
          "resize_factor":4}

def run_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        logger = WandbCallback(log_preds=False)
        
        # dataloaders
        # tfms = aug_transforms()
        set_seed(SEED)
        dls = SegmentationDataLoaders.from_label_func(
            path, 
            bs=config.batch_size, 
            fnames = fnames, 
            label_func = label_func, 
            codes = codes, 
            seed = SEED,
            item_tfms = Resize((720//config.resize_factor, 
                                960//config.resize_factor)),
            # batch_tfms = tfms,
        )

        # model
        cbs = [SaveModelCallback(fname=f"unet_{config.arch}"), logger, MixedPrecision()]
        
        loss_func = getattr(fastai.losses, config.loss_func)(axis=1)
        model = {"resnet34":resnet34, "xresnet34":xresnet34, "xresnext34":xresnext34}
        learn = unet_learner(dls, 
                             model[config.arch],
                             metrics = [DiceMulti(), foreground_acc],
                             loss_func = loss_func,
                             cbs = cbs)
        learn.fine_tune(10, base_lr=config.lr)
        
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'valid_loss',
        'goal': 'minimize'   
    }
}

    
parameters = {
    'lr': {# a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.01
      },
    'batch_size': {
        "values": [4, 8, 16]
    },
    'arch':{
        "values": ["resnet34", "xresnet34", "xresnext34"]
    },
    'loss_func': {
        "values": ["CrossEntropyLossFlat", "FocalLossFlat"]
    },
    'resize_factor':{
        "value": 4
    }
}

sweep_config["parameters"] = parameters
sweep_config["parameters"] = parameters

sweep_id = wandb.sweep(sweep_config, 
                       project="CamVid", 
                       entity="hydranet")

wandb.agent(sweep_id, run_train, count=30)
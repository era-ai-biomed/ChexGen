import logging
import os
import torch


def save_checkpoint(work_dir,
                    epoch,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    keep_last=False,
                    step=None,
                    ):
    os.makedirs(work_dir, exist_ok=True)
    state_dict = dict(state_dict=model.state_dict())
    if model_ema is not None:
        state_dict['state_dict_ema'] = model_ema.state_dict()
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        state_dict['scheduler'] = lr_scheduler.state_dict()
    if epoch is not None:
        state_dict['epoch'] = epoch
        file_path = os.path.join(work_dir, f"epoch_{epoch}.pth")
        if step is not None:
            state_dict['step'] = step
            file_path = file_path.split('.pth')[0] + f"_step_{step}.pth"
    logger = logging.getLogger(__name__)
    torch.save(state_dict, file_path)
    logger.info(f'Saved checkpoint of epoch {epoch} to {file_path}.')
    if keep_last and epoch is not None:
        for i in range(epoch):
            previous_ckpt = os.path.join(work_dir, f"epoch_{i}.pth")
            if os.path.exists(previous_ckpt):
                os.remove(previous_ckpt)
            # Also attempt to remove checkpoints with steps if they exist
            previous_step_ckpts = [f for f in os.listdir(work_dir) if f.startswith(f"epoch_{i}_step_") and f.endswith(".pth")]
            for ckpt in previous_step_ckpts:
                os.remove(os.path.join(work_dir, ckpt))


def load_checkpoint(checkpoint,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    load_ema=False,
                    resume_optimizer=True,
                    resume_lr_scheduler=True,
                    del_umap=False,
                    ):
    assert isinstance(checkpoint, str)
    ckpt_file = checkpoint
    checkpoint = torch.load(ckpt_file, map_location="cpu")

    for key in ["state_dict", "state_dict_ema"]:
        if key in checkpoint:
            checkpoint[key] = {k.replace('module.', ''): v for k, v in checkpoint[key].items()}

    state_dict_keys = ['pos_embed', 'base_model.pos_embed', 'model.pos_embed', 'y_embedder.y_embedding']
    for key in state_dict_keys:
        if key in checkpoint['state_dict']:
            del checkpoint['state_dict'][key]
            if 'state_dict_ema' in checkpoint and key in checkpoint['state_dict_ema']:
                del checkpoint['state_dict_ema'][key]
                
    if load_ema:
        state_dict = checkpoint['state_dict_ema']
    else:
        state_dict = checkpoint.get('state_dict', checkpoint)  # to be compatible with the official checkpoint

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing, unexpect = model.load_state_dict(state_dict, strict=False)

    # clear our the gradient cache of the model
    if model_ema is not None:
        model_ema.load_state_dict(checkpoint['state_dict_ema'], strict=False)
    if optimizer is not None and resume_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None and resume_lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    logger = logging.getLogger(__name__)
    if optimizer is not None:
        epoch = checkpoint.get('epoch')
        iter = checkpoint.get('step')
        logger.info(f'Resume checkpoint of iter {iter} epoch {epoch} from {ckpt_file}. Load ema: {load_ema}, '
                    f'resume optimizer： {resume_optimizer}, resume lr scheduler: {resume_lr_scheduler}.')
        return (iter, epoch), missing, unexpect
    logger.info(f'Load checkpoint from {ckpt_file}. Load ema: {load_ema}.')
    return missing, unexpect
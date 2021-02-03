import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from trainer2 import Trainer2
from trainer3 import Trainer3

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if args.data_test == 'video':
    from videotester import VideoTester
    model = model.Model(args, checkpoint)
    t = VideoTester(args, model, checkpoint)
    t.test()
elif args.trainer_v == 1:
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        if args.prune:
            while not t.terminate():
                t.finetune()
                t.test_finetune()
        else: 
            while not t.terminate():
                t.train()
                t.test()

        checkpoint.done()


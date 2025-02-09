import pickle
import random
import logging

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange


class Runner:
    def __init__(self, cfg, exp, device, resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.logger = logging.getLogger(__name__)

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def eval(self, epoch, save_predictions=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model = model.to(self.device)
        model.eval()

        dataloader = self.get_test_dataloader()
        
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        #self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            for idx, (images, _, _) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                output = model(images, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                predictions.extend(prediction)
                if self.view:
                    img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
                    if self.view == 'mistakes' and fp == 0 and fn == 0:
                        continue
                    #cv2.imshow('pred', img)
                    #cv2.waitKey(0)
                    

        if save_predictions:
            print(predictions)
            with open('predictions.pkl', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

        return predictions


    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader


    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)

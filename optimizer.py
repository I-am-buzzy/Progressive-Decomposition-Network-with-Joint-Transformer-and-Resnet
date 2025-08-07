#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import logging

logger = logging.getLogger()


class Optimizer(object):   #optimizer = Optimizer(model=PSF(9), lr0=0.001, momentum=0.9, wd=5e-4, warmup_steps=1000, warmup_start_lr=1e-5, max_iter=67750, power=0.9)
    def __init__(self, model, lr0, momentum, wd, warmup_steps, warmup_start_lr, max_iter, power, it=0, *args, **kwargs):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = it
        self.optim = torch.optim.SGD(model.parameters(), lr=lr0, momentum=momentum, weight_decay=wd)
        self.warmup_factor = (self.lr0/self.warmup_start_lr)**(1./self.warmup_steps)     #self.warmup_factor=(0.001/1e-5)**(1./1000)=1.0046157902783952

    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr*(self.warmup_factor**self.it)  #lr = 0.001*(1.0046157902783952**self.it)
        else:
            factor = (1-(self.it-self.warmup_steps)/(self.max_iter-self.warmup_steps))**self.power    #factor = (1-(self.it-1000)/(67750-1000))**0.9
            lr = self.lr0 * factor   #lr = 0.001 * factor
        return lr

    def step(self):
        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
            if pg.get('lr_mul', False):
                pg['lr'] = self.lr * 10
            else:
                pg['lr'] = self.lr
        if self.optim.defaults.get('lr_mul', False):
            self.optim.defaults['lr'] = self.lr * 10
        else:
            self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2:
            logger.info('==> warmup done, start to implement poly lr strategy')

    def zero_grad(self):
        self.optim.zero_grad()


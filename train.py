import cv2
from create_dataset import *
from utils import *
from PDFN import PDFN
from options import *
from saver import Saver, resume
from time import time
from tqdm import tqdm
from optimizer import Optimizer
import datetime
import torch.utils.data
import torch
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")


def main():
    # parse options    
    parser = TrainOptions()
    opts = parser.parse()
    # define model, optimiser and scheduler
    device = torch.device("cuda:{}".format(opts.gpu) if torch.cuda.is_available() else "cpu")
    MPF_model = PDFN(opts.class_nb).to(device)    #分割模型的类别数量9  semantic_out, binary_out, boundary_out, fused_img, vi_img, ir_img   #(4 9 256 256)  (4 2 256 256) (4 2 256 256) (4,1,256,256) (4,1,256,256) (4,1,256,256)
    # define dataset
    train_dataset = MSRSData(opts, is_train=True)     #训练
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.nThreads, shuffle=True)  #opts.nThreads=16
    test_dataset = MSRSData(opts, is_train=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=12, num_workers=opts.nThreads, shuffle=False)    #原来batch_size=12

    ## 先加载dataloader 计算每个epoch的的迭代步数 然后计算总的迭代步数
    ep_iter = len(train_loader)
    print('ep_iter:', ep_iter)            #1083/4=271
    max_iter = opts.n_ep * ep_iter        #250*271=67750
    print('Training iter: {}'.format(max_iter))
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-3       #原来1e-3
    # max_iter = 150000
    power = 0.9
    warmup_steps = 1000       #原来是1000
    warmup_start_lr = 1e-5     #原来是1e-5
    optimizer = Optimizer(model=MPF_model, lr0=lr_start, momentum=momentum, wd=weight_decay, warmup_steps=warmup_steps, warmup_start_lr=warmup_start_lr, max_iter=max_iter, power=power)
    #optimizer = Optimizer(model=PSF(9), lr0=0.001, momentum=0.9, wd=5e-4, warmup_steps=1000, warmup_start_lr=1e-5, max_iter=67750, power=0.9)
    if opts.resume:            #本文是None，没有用的下面
        MPF_model, optimizer.optim, ep, total_it = resume(MPF_model, optimizer.optim, opts.resume, device)
        optimizer = Optimizer(model=MPF_model, lr0=lr_start, momentum=momentum, wd=weight_decay, warmup_steps=warmup_steps, warmup_start_lr=warmup_start_lr, max_iter=max_iter, power=power, it=total_it)
        lr = optimizer.get_lr()
        print('lr:{}'.format(lr))
    else:                    #opts.resume=None
        ep = -1
        total_it = 0
    ep += 1
    # optimizer = optim.Adam(MPF_model.parameters(), lr=opts.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    log_dir = os.path.join(opts.display_dir, 'logger', opts.name)    #opts.name=PSFusion
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir + timestamp + '.pth')       #我修改的，希望看到日期,  log_path = os.path.join(log_dir, 'log.txt')   原来的

    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logger_config(log_path=log_path, logging_name='Timer')
    logger.info('Parameter: {:.6f}M'.format(count_parameters(MPF_model) / 1024 * 1024))

    # Train and evaluate multi-task network
    multi_task_trainer(train_loader, test_loader, MPF_model, device, optimizer, opts, logger, ep, total_it)


def multi_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, opt, logger, start_ep=0, total_it=0):
    global img_ir, img_vi, label, re_ir, fused_img, re_vi, vi_Cb, vi_Cr, seg_results, bi_results, bd_results
    total_epoch = opt.n_ep        #250
    saver = Saver(opt)    
    ## 计算分割损失相关的设计
    score_thres = 0.75
    ignore_idx = 255
    n_min = 16 * 256 * 256 // 16
    criteria = OhemCELoss(thresh=score_thres, n_min=n_min, device=device, ignore_lb=ignore_idx)    #图像分割损失函数,其实就是分类任务的交叉熵函数OhemCELoss(thresh=0.75,n_min=256*256,device=device,ignore_lb=256)
    # criteria_fusion = Fusionloss(device=device)
    binary_class_weight = np.array([1.4548, 19.8962])    
    binary_class_weight = torch.tensor(binary_class_weight).float().to(device)
    binary_class_weight = binary_class_weight.unsqueeze(0)      #torch.Size([1, 2])   [1.4548, 19.8962]
    binary_class_weight = binary_class_weight.unsqueeze(2)
    binary_class_weight = binary_class_weight.unsqueeze(2)      #torch.Size([1, 2, 1, 1])
    lb_ignore = [255]
    if opt.resume:
        best_mIou = multi_task_tester(test_loader, multi_task_model, device, opt) - 0.02
    else:          #本文是None
        best_mIou = 0.0
    print('best mIoU: {:.4f}'.format(best_mIou))
    start = glob_st = time()
    for ep in range(start_ep, total_epoch):           #每一个epoch 计算一次动态权重    （0，250）
        multi_task_model.train()        
        # Fusion_Criteria = Fusionloss(device=device)
        seg_metric = SegmentationMetric(opt.class_nb, device=device)          
        for it, (img_ir, img_vi, label, bi, bd, mask) in enumerate(train_loader):
            total_it += 1
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            label = label.to(device)           #torch.Size([4, 1, 256, 256])
            bi = bi.to(device).squeeze(1)      #torch.Size([4, 256, 256])
            bd = bd.to(device).squeeze(1)      #torch.Size([4, 256, 256])
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vi)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            mask = mask.to(device)
            seg_pred, bi_pred, bd_pred, fused_img, re_vi, re_ir = multi_task_model(img_vi, img_ir)   #MPF_model=PSF  #(4 9 256 256)  (4 2 256 256) (4 2 256 256) (4,1,256,256) (4,1,256,256) (4,1,256,256)
            # seg_pred = F.softmax(seg_pred, dim=1) 
            # seg_pred = multi_task_model(img_vi, img_ir)
            optimizer.zero_grad()
            seg_loss = Seg_loss(seg_pred, label, device, criteria)      #criteria = OhemCELoss  交叉熵损失
            bd = F.one_hot(bd, num_classes=2)      #torch.Size([4, 256, 256, 2])
            bd = bd.permute(0, 3, 1, 2).float()    #torch.Size([4, 2，256, 256])
            bi = F.one_hot(bi, num_classes=2)      #torch.Size([4, 256, 256, 2])
            bi = bi.permute(0, 3, 1, 2).float()    #torch.Size([4, 2，256, 256])
            bd_loss = F.binary_cross_entropy_with_logits(bd_pred, bd)           #交叉熵损失
            bi_loss = F.binary_cross_entropy_with_logits(bi_pred, bi, pos_weight=binary_class_weight)
            seg_results = torch.argmax(seg_pred, dim=1, keepdim=True)           #print(seg_result.shape())
            bi_results = torch.argmax(bi_pred, dim=1, keepdim=True)    #自己加的
            bd_results = torch.argmax(bd_pred, dim=1, keepdim=True)    #自己加的
            train_seg_loss = 10 * seg_loss + 10 * bi_loss + 10 * bd_loss

            #reconstruction-related loss
            fusion_loss, int_loss, grad_loss, corr_loss = Fusion_loss(vi_Y, img_ir, fused_img, mask=mask, device=device)     #return loss_total, loss_intensity, loss_grad, loss_ssim
            vi_re_loss, vi_int_loss, vi_grad_loss = Re_loss(re_vi, vi_Y, mask=mask, ir_flag=False)   #loss_total, loss_intensity, loss_grad
            ir_re_loss, ir_int_loss, ir_grad_loss = Re_loss(re_ir, img_ir, mask=mask, ir_flag=True)
        
            train_loss = 1 * train_seg_loss + 1 * fusion_loss + 1 * vi_re_loss + 1 * ir_re_loss
            train_loss.backward()
            #nn.utils.clip_grad_norm_(multi_task_model.parameters(), max_norm=0.01, norm_type=2)  # 梯度裁剪，防止梯度爆炸  自己加进去的效果不好
            optimizer.step()
            seg_metric.addBatch(seg_results, label, lb_ignore)
        lr = optimizer.get_lr()
        mIoU = np.array(seg_metric.meanIntersectionOverUnion().item())
        Acc = np.array(seg_metric.pixelAccuracy().item())
        end = time()
        training_time, glob_t_intv = end - start, end - glob_st
        now_it = total_it+1
        eta = int((total_epoch * len(train_loader) - now_it) * (glob_t_intv / (now_it)))
        eta = str(datetime.timedelta(seconds=eta))
        logger.info('ep: [{}/{}], learning rate: {:.6f}, time consuming: {:.2f}s, segmentation loss: {:.4f}, fusion loss: {:.4f}, vi rec loss: {:.4f}, ir rec loss: {:.4f}'.format(ep+1, total_epoch, lr, training_time, seg_loss.item(), fusion_loss.item(), vi_re_loss.item(), ir_re_loss.item()))
        logger.info('grad loss: [{:.4f}], int loss: [{:.4f}], corr loss: [{:.4f}], bi loss: [{:.4f}], bd loss: [{:.4f}], segmentation loss: {:.4f}, mIou: {:.4f}, Acc: {:.4f}, Eta: {}\n'.format(grad_loss.item(), int_loss.item(), corr_loss.item(), bi_loss.item(), bd_loss.item(), seg_loss.item(), mIoU, Acc, eta))
        start = time()

        # save Visualization results
        if (ep + 1) % opt.img_save_freq == 0:
            input = [img_ir, img_vi, fused_img, label]
            fused_rgb = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            vi_rgb = YCbCr2RGB(re_vi, vi_Cb, vi_Cr)
            output = [re_ir, vi_rgb, fused_rgb, seg_results]
            saver.write_img(ep, input, output)

            inimage = [img_ir, img_vi, fused_img, label]
            outimage = [bi_results, bd_results, fused_rgb, seg_results]
            saver.write_img1(ep, inimage, outimage)

        # save model
        if (ep + 1) % opt.model_save_freq == 0:
            if (ep + 1) > 60:    #1500
                if (ep + 1) > 70:    #2400
                    saver.write_model(ep, total_it, multi_task_model, optimizer.optim, best_mIou, device, is_best=False)
                test_mIoU = multi_task_tester(test_loader, multi_task_model, device, opt)            
                logger.info('test mIoU: {:.4f}, best mIoU:{:.4f}'.format(test_mIoU, best_mIou))
                if test_mIoU > best_mIou:
                    best_mIou = test_mIoU
                    saver.write_model(ep, total_it, multi_task_model, optimizer.optim, best_mIou, device)


def multi_task_tester(test_loader, multi_task_model, device, opts):
    multi_task_model.eval()
    test_bar = tqdm(test_loader)    #打印进度条
    seg_metric = SegmentationMetric(opts.class_nb, device=device)
    lb_ignore = [255]

    # define save dir
    with torch.no_grad():        # operations inside don't track history
        for it, (img_ir, img_vi, label, img_names) in enumerate(test_bar):
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            label = label.to(device)           
            Seg_pred, _, _, fused_img, re_vi, re_ir = multi_task_model(img_vi, img_ir)            
            seg_result = torch.argmax(Seg_pred, dim=1, keepdim=True)             #print(seg_result.shape())  (12 1 256 256)
            seg_metric.addBatch(seg_result, label, lb_ignore)        
    mIoU = np.array(seg_metric.meanIntersectionOverUnion().item())
    return mIoU


if __name__ == '__main__':
    main()

import os
import logging
import time
import warnings
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils import LRScheduler
 
from gluoncv.utils import download, viz
from matplotlib import pyplot as plt

from labelme.utils.map import Map

   
def read_classes(args):
    with open(args.classes_list) as f:
        return f.read().split('\n')
 
def get_dataset(args):
    train_dataset = gcv.data.RecordFileDetection(args.train_dataset)
    val_dataset = gcv.data.RecordFileDetection(args.validate_dataset)
    classes = read_classes(args.classes_list)
    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
   
    if args.num_samples < 0:
        args.num_samples = len(train_dataset)
    if args.mixup:
        from gluoncv.data import MixupDetection
        train_dataset = MixupDetection(train_dataset)
    return train_dataset, val_dataset, val_metric
 
def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, args):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))  # stack image, all targets generated
    if args.no_random_shape:
        print("no random shape")
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(YOLO3DefaultTrainTransform(width, height, net, mixup=args.mixup)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    else:
        print("with random shape")
        transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, net, mixup=args.mixup) for x in range(10, 20)]
        train_loader = RandomTransformDataLoader(
            transform_fns, train_dataset, batch_size=batch_size, interval=10, last_batch='rollover',
            shuffle=True, batchify_fn=batchify_fn, num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))    
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, True, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader
 
def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))
 
def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
 
        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()
 
def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    if args.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0
 
    if args.label_smooth:
        net._target_generator._label_smooth = True
 
    if args.lr_decay_period > 0:
        lr_decay_epoch = list(range(args.lr_decay_period, args.epochs, args.lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
    lr_scheduler = LRScheduler(mode=args.lr_mode,
                               baselr=args.lr,
                               niters=args.num_samples // args.batch_size,
                               nepochs=args.epochs,
                               step=lr_decay_epoch,
                               step_factor=args.lr_decay, power=2,
                               warmup_epochs=args.warmup_epochs)
 
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler},
        kvstore='local')
 
    # targets
    sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    l1_loss = gluon.loss.L1Loss()
 
    # metrics
    obj_metrics = mx.metric.Loss('ObjLoss')
    center_metrics = mx.metric.Loss('BoxCenterLoss')
    scale_metrics = mx.metric.Loss('BoxScaleLoss')
    cls_metrics = mx.metric.Loss('ClassLoss')
 
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        if args.mixup:
            # TODO(zhreshold): more elegant way to control mixup during runtime
            try:
                train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
            except AttributeError:
                train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
            if epoch >= args.epochs - args.no_mixup_epochs:
                try:
                    train_data._dataset.set_mixup(None)
                except AttributeError:
                    train_data._dataset._data.set_mixup(None)
 
        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        net.hybridize()
        for i, batch in enumerate(train_data):
            #print("training batch\n", batch)
            #return 0
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            # objectness, center_targets, scale_targets, weights, class_targets
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
            gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
            sum_losses = []
            obj_losses = []
            center_losses = []
            scale_losses = []
            cls_losses = []
            with autograd.record():
                for ix, x in enumerate(data):
                    obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                    sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                    obj_losses.append(obj_loss)
                    center_losses.append(center_loss)
                    scale_losses.append(scale_loss)
                    cls_losses.append(cls_loss)
                autograd.backward(sum_losses)
            lr_scheduler.update(i, epoch)
            trainer.step(batch_size)
            obj_metrics.update(0, obj_losses)
            center_metrics.update(0, center_losses)
            scale_metrics.update(0, scale_losses)
            cls_metrics.update(0, cls_losses)
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                name4, loss4 = cls_metrics.get()
                logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, trainer.learning_rate, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
            btic = time.time()
 
        name1, loss1 = obj_metrics.get()
        name2, loss2 = center_metrics.get()
        name3, loss3 = scale_metrics.get()
        name4, loss4 = cls_metrics.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
        if not (epoch + 1) % args.val_interval:
            print("validate:", epoch + 1)
            # consider reduce the frequency of validation to save time
            #map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            #val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            #logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            #current_map = float(mean_ap[-1])
            current_map = 0.
        else:
            current_map = 0.
        #save_params(net, best_map, current_map, epoch, args.save_interval, args.save_prefix)
        net.save_parameters('yolo3_pikachu')
    net.save_parameters('yolo3_pikachu')
 

def train_yolov3(output_dir, progress, data_shape=416, batch_size=8, num_workers=1, gpus='0', epochs=10, resume='',
    start_epoch=0, lr=0.001, lr_mode='step', lr_decay=0.1, lr_decay_period=0,
    lr_decay_epoch='160,180', warmup_lr=0.0, warmup_epochs=0, momentum=0.9, wd=0.0005,
    log_interval=100, save_prefix='', save_interval=1, val_interval=1, seed=42,
    num_samples=-1, syncbn=False, no_random_shape=False, no_wd=False, mixup=False, 
    no_mixup_epochs=20, train_dataset='', validate_dataset='', classes_list='', 
    pretrained=0, label_smooth=False, only_inference=False):

    args = Map({
        'output_dir': output_dir,
        #'network': 'yolo',
        'data_shape': data_shape,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'gpus': gpus,
        'epochs': epochs,
        'resume': resume,
        'start_epoch': start_epoch,
        'lr': lr,
        'lr_mode': lr_mode,
        'lr_decay': lr_decay,
        'lr_decay_period': lr_decay_period,
        'lr_decay_epoch': lr_decay_epoch,
        'warmup_lr': warmup_lr,
        'warmup_epochs': warmup_epochs,
        'momentum': momentum,
        'wd': wd,
        'log_interval': log_interval,
        'save_prefix': save_prefix,
        'save_interval': save_interval,
        'val_interval': val_interval,
        'seed': seed,
        'num_samples': num_samples,
        'syncbn': syncbn,
        'no_random_shape': no_random_shape,
        'no_wd': no_wd,
        'mixup': mixup,
        'no_mixup_epochs': no_mixup_epochs,
        'train_dataset': train_dataset,
        'validate_dataset': validate_dataset,
        'classes_list': classes_list,
        'pretrained': pretrained,
        'label_smooth': label_smooth,
        'only_inference': only_inference,
    })

    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)
 
    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
 
    # network
    net_name = 'yolo3_darknet53_coco'
    args.save_prefix += net_name
    # use sync bn if specified
    num_sync_bn_devices = len(ctx) if args.syncbn else -1
    
    classes = read_classes(args)
    
    if not args.only_inference:
        
        net = None
        if num_sync_bn_devices > 1:
            print("num_sync_bn_devices > 1")
            if args.pretrained == 0:
                net = get_model(net_name, pretrained=True, num_sync_bn_devices=num_sync_bn_devices)
            else:        
                net = get_model(net_name, pretrained_base=True, num_sync_bn_devices=num_sync_bn_devices)
               
            net.reset_class(classes)            
            async_net = get_model(net_name, pretrained_base=False)  # used by cpu worker
        else:
            print("num_sync_bn_devices <= 1")        
            if args.pretrained == 0:
                net = get_model(net_name, pretrained=True)            
            else:
                net = get_model(net_name, pretrained_base=True)
            net.reset_class(classes)            
            async_net = net
               
        if args.resume.strip():
            net.load_parameters(args.resume.strip())
            async_net.load_parameters(args.resume.strip())
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                net.initialize()
                async_net.initialize()
     
        # training data
        train_dataset, val_dataset, eval_metric = get_dataset(args)
        train_data, val_data = get_dataloader(
            async_net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers, args)
     
        # training
        train(net, train_data, val_data, eval_metric, ctx, args)
        
        # export
        net.export('export_yolo3_pikachu')
    
    # test_url = 'https://raw.githubusercontent.com/zackchase/mxnet-the-straight-dope/master/img/pikachu.jpg'
    # download(test_url, 'pikachu_test.jpg')
    # net = gcv.model_zoo.get_model('yolo3_darknet53_custom', classes=classes, pretrained_base=False)
    # net.load_parameters('yolo3_pikachu')
    # net.collect_params().reset_ctx(mx.cpu())
    # #validate(net, val_data, ctx, eval_metric)
    # x, image = gcv.data.transforms.presets.yolo.load_test('pikachu_test.jpg', 416)
    # cid, score, bbox = net(x)
    # ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
    # plt.show()
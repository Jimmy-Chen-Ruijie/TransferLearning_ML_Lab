from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import warnings

import copy

def read_weights(files, folder):
    weight_list = []
    for file in files:
        weight_path = folder + file
        with open(weight_path, 'r', encoding='utf-8') as f:
            for line in f:
                weight = float(line)
                weight_list.append(weight)
    weights = torch.Tensor([weight_list]).squeeze()
    return weights

def peak_at_one(w):
        tmp = 1.0 / (2 * torch.abs((w - 1)) + 1e-16)
        return 2 * F.sigmoid(tmp) - 1

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="config/custom_daytime_night_fakenight.data", help="path to data config file")
    parser.add_argument("--modularization",choices=[1,2,3], type=int, default=1)
    parser.add_argument("--models_def", nargs='+', help="paths to encoder and detector definition files", default=['config/yolov3-Mod1-Encoder.cfg', 'config/yolov3-Mod1-Detector.cfg'])  # first encoder, then detectors in ascending order
    parser.add_argument("--pretrained_weights", nargs='*', help="paths to encoder and detector pretrained weights")  # can be left empty
    parser.add_argument("--weighting", choices=["sigmoid", "peak_at_one", None], default="peak_at_one")
    parser.add_argument("--weights_folder", default="./data/custom/weights/train/")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accumulations before step")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument('--exp_id', type=int, default=1, help='experiment id')
    opt = parser.parse_args()
    print(opt)

    check_modularization(opt.modularization, opt.models_def, opt.pretrained_weights)
    config_name = 'config' + str(opt.exp_id)

    os.makedirs("logs-YOLO/{}/logs_night".format(config_name), exist_ok=True) 
    os.makedirs("logs-YOLO/{}/logs_fakenight".format(config_name), exist_ok=True)  
    os.makedirs("logs-YOLO/{}/logs_daytime".format(config_name), exist_ok=True)  
    os.makedirs("checkpoints-YOLO/{}".format(config_name), exist_ok=True)
    logger_fakenight = Logger("logs-YOLO/{}/logs_fakenight".format(config_name))
    logger_night = Logger("logs-YOLO/{}/logs_night".format(config_name))
    logger_daytime = Logger("logs-YOLO/{}/logs_daytime".format(config_name))

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train_fakenight"]
    valid_path_daytime = data_config["valid_daytime"]
    valid_path_night = data_config["valid_night"]
    valid_path_fakenight = data_config["valid_fakenight"]
    class_names = load_classes(data_config["names"])

    # Initialize model
    if opt.modularization == 1:
        encoder = Encoder_Mod1(opt.models_def[0]).to(device)
        detector = Detector_Mod1(opt.models_def[1]).to(device)
        if opt.pretrained_weights:
            encoder.load_state_dict(opt.pretrained_weights[0])
            detector.load_state_dict(opt.pretrained_weights[1])
        else:
            encoder.apply(weights_init_normal)
            detector.apply(weights_init_normal)
        model_list = [encoder, detector]
    elif opt.modularization == 2:
        encoder = Encoder_Mod2(opt.models_def[0]).to(device)
        if opt.pretrained_weights:
            encoder.load_state_dict(opt.pretrained_weights[0])
        else:
            encoder.apply(weights_init_normal)
        model_list = [encoder]
    else:
        encoder = Encoder_Mod3(opt.models_def[0]).to(device)
        detector_1 = Detector_Mod3(opt.models_def[1]).to(device)
        detector_2 = Detector_Mod3(opt.models_def[2]).to(device)
        detector_3 = Detector_Mod3(opt.models_def[3]).to(device)
        if opt.pretrained_weights:
            encoder.load_state_dict(opt.pretrained_weights[0])
            detector_1.load_state_dict(opt.pretrained_weights[1])
            detector_2.load_state_dict(opt.pretrained_weights[2])
            detector_3.load_state_dict(opt.pretrained_weights[3])
        else:
            encoder.apply(weights_init_normal)
            detector_1.apply(weights_init_normal)
            detector_2.apply(weights_init_normal)
            detector_3.apply(weights_init_normal)
        model_list = [encoder, detector_1, detector_2, detector_3]


    if opt.weighting == "sigmoid":
        transform_weights = F.sigmoid
    elif opt.weighting == "peak_at_one":
        transform_weights = peak_at_one    

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    lr = 0.0001 if opt.pretrained_weights else 0.001
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    if opt.modularization == 1:
        detector_opt = torch.optim.Adam(detector.parameters(), lr=lr)
    elif opt.modularization == 3:
        detector_1_opt = torch.optim.Adam(detector_1.parameters(), lr=lr)
        detector_2_opt = torch.optim.Adam(detector_2.parameters(), lr=lr)
        detector_3_opt = torch.optim.Adam(detector_3.parameters(), lr=lr)

    max_epoch = opt.epochs

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    best_mAP = 0
    for epoch in range(0, max_epoch): 
        for model in model_list:
            model.train()

        start_time = time.time() 
        for batch_i, (paths, imgs, targets) in enumerate(dataloader): 
            if imgs.shape[0] == opt.batch_size:
                files = [path[-21:-4] + ".txt" for path in paths]
                weights = None
                if opt.weighting is not None:
                    weights = read_weights(files, opt.weights_folder).to(device)
                    weights = transform_weights(weights)
                
                batches_done = len(dataloader) * epoch + batch_i 
                imgs = Variable(imgs.to(device))
                targets = Variable(targets.to(device), requires_grad=False)
                img_dim = imgs.shape[2]

                if opt.modularization == 1:
                    h = encoder(imgs)
                    loss, outputs = detector(h, img_dim, targets, weights)
                    loss.backward()

                    yolo_layers = detector.yolo_layers

                    if batches_done % opt.gradient_accumulations: 
                        encoder_opt.step(), detector_opt.step()
                        encoder_opt.zero_grad(), detector_opt.zero_grad()

                elif opt.modularization == 2:
                    layer_loss_small, layer_loss_middle, layer_loss_large, feature_small, feature_middle, feature_large, outputs = encoder(imgs, img_dim, targets, weights)
                    loss = layer_loss_small + layer_loss_middle + layer_loss_large
                    loss.backward()

                    yolo_layers = encoder.yolo_layers

                    if batches_done % opt.gradient_accumulations:
                        encoder_opt.step()
                        encoder_opt.zero_grad()

                else:  # modularization 3
                    h_detector_1, h_detector_2, h_detector_3 = encoder(imgs)
                    img_dim = imgs.shape[2]
                    loss_detector_1, outputs_detector_1 = detector_1(h_detector_1, img_dim, targets, weights)
                    loss_detector_2, outputs_detector_2 = detector_2(h_detector_2, img_dim, targets, weights)
                    loss_detector_3, outputs_detector_3 = detector_3(h_detector_3, img_dim, targets, weights)
                    loss = loss_detector_1 + loss_detector_2 + loss_detector_3
                    outputs = torch.cat([outputs_detector_1, outputs_detector_2, outputs_detector_3], 1)

                    yolo_layers = [detector_1.yolo_layers[0], detector_2.yolo_layers[0], detector_3.yolo_layers[0]]

                    loss.backward()

                    if batches_done % opt.gradient_accumulations: 
                        encoder_opt.step(), detector_1_opt.step(), detector_2_opt.step(), detector_3_opt.step()
                        encoder_opt.zero_grad(), detector_1_opt.zero_grad(), detector_2_opt.zero_grad(), detector_3_opt.zero_grad()
                    
                if batches_done % 100 == 0:
                    # -----------------------------------
                    #   Log progress every 100th batch
                    # -----------------------------------

                    log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader)) 

                    metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(yolo_layers))]]] 

                    # Log metrics at each YOLO layer
                    for i, metric in enumerate(metrics):
                        formats = {m: "%.6f" for m in metrics}
                        formats["grid_size"] = "%2d"
                        formats["cls_acc"] = "%.2f%%"
                        row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in yolo_layers]
                        metric_table += [[metric, *row_metrics]] 

                    log_str += AsciiTable(metric_table).table
                    log_str += f"\nTotal loss {loss.item()}"

                    # Determine approximate time left for epoch
                    epoch_batches_left = len(dataloader) - (batch_i + 1)
                    time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                    log_str += f"\n---- ETA {time_left}"
                    print(log_str) 

        # probably don't need this, just in case
        if opt.modularization == 1:
            model_list = [encoder, detector]
        elif opt.modularization == 2:
            model_list = [encoder]
        else:
            model_list = [encoder, detector_1, detector_2, detector_3]

        if epoch % opt.evaluation_interval == 0: 
            
            # Tensorboard logging
            tensorboard_log = []
            for j, yolo in enumerate(yolo_layers): 
                for name, metric in yolo.metrics.items():
                    if name != "grid_size":
                        tensorboard_log += [(f"{name}_{j + 1}", metric)]  
            tensorboard_log += [("loss", loss.item())]
            logger_fakenight.list_of_scalars_summary(tensorboard_log, epoch)     
            
            print("\n---- Evaluating Model on Daytime ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model_list,
                path=valid_path_daytime,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger_daytime.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            
            print("\n---- Evaluating Model on Night ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model_list,
                path=valid_path_night,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger_night.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            
            print("\n---- Evaluating Model on Fake Night ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model_list,
                path=valid_path_fakenight,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger_fakenight.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(encoder.state_dict(), "checkpoints-YOLO/{}/encoder_{}.pth".format(config_name, epoch))
            if opt.modularization == 1:
                torch.save(detector.state_dict(), "checkpoints-YOLO/{}/detector_{}.pth".format(config_name, epoch))
            elif opt.modularization == 3:
                torch.save(detector_1.state_dict(), "checkpoints-YOLO/{}/detector1_{}.pth".format(config_name, epoch))
                torch.save(detector_2.state_dict(), "checkpoints-YOLO/{}/detector2_{}.pth".format(config_name, epoch))
                torch.save(detector_3.state_dict(), "checkpoints-YOLO/{}/detector3_{}.pth".format(config_name, epoch))
            
        if AP.mean() > best_mAP:
            best_epoch = epoch
            best_mAP = AP.mean()
            best_model_weights_encoder = copy.deepcopy(encoder.state_dict())
            if opt.modularization == 1:
                best_model_weights_detector = copy.deepcopy(detector.state_dict())
            elif opt.modularization == 3:
                best_model_weights_detector_1 = copy.deepcopy(detector_1.state_dict())
                best_model_weights_detector_2 = copy.deepcopy(detector_2.state_dict())
                best_model_weights_detector_3 = copy.deepcopy(detector_3.state_dict())

    torch.save(best_model_weights_encoder, "checkpoints-YOLO/{}/best_encoder_{}.pth".format(config_name, best_epoch))
    if opt.modularization == 1:
        torch.save(best_model_weights_detector, "checkpoints-YOLO/{}/best_detector_{}.pth".format(config_name, best_epoch))
    elif opt.modularization == 3:
        torch.save(best_model_weights_detector_1, "checkpoints-YOLO/{}/best_detector1_{}.pth".format(config_name, best_epoch))
        torch.save(best_model_weights_detector_2, "checkpoints-YOLO/{}/best_detector2_{}.pth".format(config_name, best_epoch))
        torch.save(best_model_weights_detector_3, "checkpoints-YOLO/{}/best_detector3_{}.pth".format(config_name, best_epoch))

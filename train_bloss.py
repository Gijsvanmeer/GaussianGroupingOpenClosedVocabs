# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, loss_cls_3d
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils_added.secondary_losses import BoundaryLoss
from os import makedirs
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from utils_added.dataloader import dist_map_transform


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb, lmd, dl3_classes, save_param, save_file, dl3_path):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, use_dl3=True, use_edges=False, dl3_path=dl3_path)
    gaussians.training_setup(opt)
    num_classes = dataset.num_classes
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    classifier.cuda()
    disttransform = dist_map_transform([1, 1], 256)
    dl3_disttransform = dist_map_transform([1, 1], 41)

    if save_param:
        L1_losses = []
        loss_vals = []
        bloss_DEVA = []
        bloss_DL3 = []
        save_file = save_file + "bloss_method"
        save_file_eval = save_file + "_test_vals.txt"
        save_file_loss = save_file + ".txt"
    else:
        save_file_eval = None

    lmd2 = 1 - lmd
    classifier_dl3 = torch.nn.Conv2d(gaussians.num_objects, dl3_classes, kernel_size=1)
    cls_criterion_dl3 = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer_dl3 = torch.optim.Adam(classifier_dl3.parameters(), lr=5e-4)
    classifier_dl3.cuda()
    makedirs(os.path.join(dataset.model_path,"point_cloud","dl3_iteration_1000"),exist_ok=True)
    makedirs(os.path.join(dataset.model_path,"point_cloud","dl3_iteration_7000"),exist_ok=True)
    makedirs(os.path.join(dataset.model_path,"point_cloud","dl3_iteration_30000"),exist_ok=True)


    boundary_loss_DEVA = BoundaryLoss(idc=np.arange(num_classes))

    boundary_loss_DL3 = BoundaryLoss(idc=np.arange(41))

    alpha=0.01

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    start_time = time.time()

    for iteration in range(first_iter, opt.iterations + 1):  

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]


        # Object Loss

        # Obtain distance mapping for Boundary Loss
        distance_map_deva = disttransform(viewpoint_cam.objects.cpu())
        distance_map_deva = distance_map_deva.cuda().long()
        gt_obj = viewpoint_cam.objects.cuda().long()

        if torch.isnan(objects).any():
            print("NaN in used objects")
        
        # Obtain classifications
        logits = classifier(objects)
        scaled_logits = F.softmax(logits, dim=0)

        loss_deva = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0))
        loss_deva = loss_deva.squeeze().mean()
        loss_deva = loss_deva / torch.log(torch.tensor(num_classes))  # normalize to (0,1)

        # Calculate Boundary Loss
        bl_loss_deva = boundary_loss_DEVA(scaled_logits.unsqueeze(0), distance_map_deva.unsqueeze(0))
        loss_obj = loss_deva + alpha*bl_loss_deva

        # Closed vocabulary model loss
        
        # Obtain distance mapping for Boundary Loss
        distance_map_dl3 = dl3_disttransform(viewpoint_cam.dl3_objects.cpu()).cuda().long()
        gt_obj_dl3 = viewpoint_cam.dl3_objects.cuda().long()

        # Obtain classifications
        logits_dl3 = classifier_dl3(objects)
        scaled_dl3_logits = F.softmax(logits_dl3, dim=0)

        loss_dl3 = cls_criterion_dl3(logits_dl3.unsqueeze(0), gt_obj_dl3.unsqueeze(0))
        loss_dl3 = loss_dl3.squeeze().mean()
        loss_dl3 = loss_dl3 / torch.log(torch.tensor(dl3_classes))  # normalize to (0,1)

        # Calculate Boundary Loss
        bl_loss_dl3 = boundary_loss_DL3(scaled_dl3_logits.unsqueeze(0), distance_map_dl3.unsqueeze(0))
        loss_obj_dl3 = loss_dl3 + alpha*bl_loss_dl3

        if save_param:
            bloss_DEVA.append(bl_loss_deva.item())
            bloss_DL3.append(bl_loss_dl3.item())


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if save_param:
            L1_losses.append(Ll1.item())

        loss_obj_3d = None
        if iteration % opt.reg3d_interval == 0:
            # regularize at certain intervals
            logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
            loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)

            # regularize as well for the second model
            logits3d_dl3 = classifier_dl3(gaussians._objects_dc.permute(2,0,1))
            prob_obj3d_dl3 = torch.softmax(logits3d_dl3,dim=0).squeeze().permute(1,0)
            loss_obj_3d_dl3 = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d_dl3, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image.to("cuda"), gt_image.to("cuda"))) + lmd*(loss_obj + loss_obj_3d) + lmd2*(loss_obj_dl3 + loss_obj_3d_dl3)

        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image.to("cuda"), gt_image.to("cuda"))) + lmd*loss_obj + lmd2*loss_obj_dl3


        if save_param:
            loss_vals.append(loss.item())
        loss.backward()

        # Prevent possible exploding gradients
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_obj_3d, use_wandb, save_param, save_file_eval)
            if (iteration in saving_iterations):
                # print(loss_vals)
                # print(L1_losses)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'classifier.pth'))

                torch.save(classifier_dl3.state_dict(), os.path.join(scene.model_path, "point_cloud/dl3_iteration_{}".format(iteration),'classifier.pth'))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                cls_optimizer.step()
                cls_optimizer.zero_grad()

                cls_optimizer_dl3.step()
                cls_optimizer_dl3.zero_grad()


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        torch.cuda.empty_cache()
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Save parameters
    if save_param:
        # Time cost
        f = open(save_file_eval, "a")
        f.write("\nTime cost: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        f.close()

        # Loss progression
        g = open(save_file_loss, "a")
        g.write("\nL1Losses: " + str(L1_losses))
        g.write("\nBaseLosses: " + str(loss_vals))
        g.close()

        # Create loss graph DEVA Boundary Loss
        if dl3_path == None:
            graph_name = "BaseDL3_"
        else:
            graph_name = dl3_path + "_"
        version = 1
        while os.path.isfile("boundary_loss_DEVA_" + str(version) + "_" + str(graph_name) + ".png"):
            version += 1
        filename = "boundary_loss_DEVA_" + str(version) + "_" + str(graph_name) + ".png"
        plt.plot(np.arange(0, len(bloss_DEVA), 1, dtype=int), bloss_DEVA)
        plt.title("Loss progress of DEVA Boundary Loss.")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(filename)
        plt.clf()
        # Create loss graph DL Boundary Loss
        version = 1
        graph_name = "boundary_loss_secondary_loss_" + graph_name + "_"
        while os.path.isfile(graph_name + str(version) + ".png"):
            version += 1
        filename = graph_name + str(version) + ".png"
        plt.plot(np.arange(0, len(bloss_DL3), 1, dtype=int), bloss_DL3)
        plt.title("Loss progress of Closed Vocab Boundary Loss.")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(filename)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_obj_3d, use_wandb, save_param, save_file):

    if use_wandb:
        if loss_obj_3d:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(), "train_loss_patches/loss_obj_3d": loss_obj_3d.item(), "iter_time": elapsed, "iter": iteration})
        else:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(), "iter_time": elapsed, "iter": iteration})
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if use_wandb:
                        if idx < 5:
                            wandb.log({config['name'] + "_view_{}/render".format(viewpoint.image_name): [wandb.Image(image)]})
                            if iteration == testing_iterations[0]:
                                wandb.log({config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): [wandb.Image(gt_image)]})
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if use_wandb:
                    wandb.log({config['name'] + "/loss_viewpoint - l1_loss": l1_test, config['name'] + "/loss_viewpoint - psnr": psnr_test})
        if save_param:
            f = open(save_file, "a")
            values = "\nEvaluating " + str(iteration) + ": L1: " + str(l1_test) + ", PSNR: " + str(psnr_test)
            f.write(values)
            f.close()
        if use_wandb:
            wandb.log({"scene/opacity_histogram": scene.gaussians.get_opacity, "total_points": scene.gaussians.get_xyz.shape[0], "iter": iteration})
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--closed_vocab_path', type=str, default=None)
    parser.add_argument('--save_param', type=bool, default=False)
    parser.add_argument('--save_file', type=str, default="ExperimentData/IntNetBase")
    parser.add_argument('--BLoss', type=bool, default=False)
    parser.add_argument('--lmd', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--dl3_classes', type=int, default=41)
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Add an argument for the configuration file
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.densify_until_iter = config.get("densify_until_iter", 15000)
    args.num_classes = config.get("num_classes", 200)
    args.reg3d_interval = config.get("reg3d_interval", 2)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)
    
    print("Optimizing " + args.model_path)

    if args.use_wandb:
        wandb.init(project="gaussian-splatting")
        wandb.config.args = args
        wandb.run.name = args.model_path

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.use_wandb, args.lmd, args.dl3_classes, args.save_param, args.save_file, args.closed_vocab_path)

    # All done
    print("\nTraining complete.")

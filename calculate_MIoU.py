import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from utils.loss_utils import MIoU


def miou_calc(views, gaussians, pipeline, background, classifier):

    MIoUs =[]
    for view in tqdm(views, desc="MIoU progress"):
        results = render(view, gaussians, pipeline, background)
        rendering_obj = results["render_object"]
        
        logits = classifier(rendering_obj)
        pred_obj = torch.argmax(logits,dim=0)
        gt_objects = view.objects

        MIoU_val = MIoU(gt_objects, pred_obj, num_classes=256)
        MIoUs.append(MIoU_val)

    return MIoUs


def obtain_MIoU(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, use_dl3: bool, file_name: str, dl3_path):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, use_dl3=use_dl3, dl3_path=dl3_path)
        
        num_classes = dataset.num_classes

        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             MIoU_vals= miou_calc(scene.getTrainCameras(), gaussians, pipeline, background, classifier)

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
             MIoU_vals= miou_calc(scene.getTestCameras(), gaussians, pipeline, background, classifier)
        f = open(file_name, "a")
        f.write(str(MIoU_vals))
        f.write("\nMEANIOU: "+str(np.mean(MIoU_vals))+"\n\n")
        f.close()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--closed_vocab_path', type=str, default=None)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--file_name', type=str, default="ExperimentData/IntNet.txt")
    parser.add_argument('--use_dl3', type=bool, default=False)
    args = get_combined_args(parser)
    print("Obtaining MIoU " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    obtain_MIoU(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.use_dl3, args.file_name, args.closed_vocab_path)
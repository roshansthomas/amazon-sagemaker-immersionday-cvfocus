import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch

import argparse
import json
import logging
import sys

#import sagemaker_containers
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
import utils
from dataloader import DataLoaderSegmentation

from torchvision.models.segmentation.deeplabv3 import DeepLabHead

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]

            confmat.update(target.flatten(), output.argmax(1).flatten())
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            num_processed_samples += image.shape[0]

        confmat.reduce_from_all_processes()

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    return confmat

def train(args):
    # Define number of classes
    num_classes=1
    
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), args.num_gpus)
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    
    data_transforms = transforms.Compose([transforms.Resize(256),
#                                           transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                               std = [0.229, 0.224, 0.225])])
    mask_transforms = transforms.Compose([transforms.Resize(256),
#                                           transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    
    train_data_dir = os.path.join(args.data_dir, 'train/')
    test_data_dir = os.path.join(args.data_dir, 'test/')
    
    train_data = DataLoaderSegmentation(train_data_dir, transforms=data_transforms, mask_transforms=mask_transforms)
    test_data = DataLoaderSegmentation(test_data_dir, transforms=data_transforms, mask_transforms=mask_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=args.batch_size,
                                          shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=args.test_batch_size,
                                          shuffle=True)


    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )
    
#     CriterionL = torch.nn.MSELoss(reduction='mean')

    if num_classes == 1:
        # Use binary cross entropy for
        CriterionL = torch.nn.BCEWithLogitsLoss()
    else:
        # Use cross entropy for multi class
        CriterionL = torch.nn.CrossEntropyLoss()

#     model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=num_classes)
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
#     model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    
    #Change to a single class classification
    model.classifier = DeepLabHead(2048, num_classes)
#     model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(num_classes=1)
    
    model.to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
#             loss = F.nll_loss(output, target)
#             loss = criterion(output, target)
#             logger.info("output shape: {}, target shape: {}".format(output['out'].shape, target.shape))
            loss = CriterionL(output['out'], target)
            loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        
#         confmat = evaluate(model, test_loader, device=device, num_classes=num_classes)
#         print(confmat)
#         test(model, test_loader, device)
    # Save the model
    save_model(model, args.model_dir)


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def input_fn(request_body, request_content_type):
    import io

    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    print("input function")
    f = io.BytesIO(request_body)
    
    print("opening image")
    input_image = Image.open(f).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
#             transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    print("transform input")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    print("return input")
    return input_batch

def output_fn(prediction, content_type):
    from sagemaker_inference import encoder
    
    print("content type {}".format(content_type))
    
    print("output prediction")
    print(type(prediction))
#     print("prediction shape: {}".format(prediction.shape))
    
#     print("resize")
#     prediction = transforms.Resize((650,650))(prediction['out'])
    
    
    
#     print("resize prediction shape: {}".format(prediction.shape))

    print(type(prediction))
    
#     np_prediction = torch.argmax(prediction[0], 1).cpu().detach().numpy()
#     np_prediction = prediction[0].data.cpu().numpy()

#     np_prediction = prediction.cpu().detach().numpy()
#     np_prediction = torch.argmax(prediction.squeeze(), dim=1).detach().cpu().numpy()
#     np_prediction = prediction['out'].argmax(0).cpu().detach().numpy()

    np_prediction = prediction['out'].data.cpu().numpy()
    print("np_prediction shape: {}".format(np_prediction.shape))
    print("np_prediction type: {}".format(np_prediction.dtype))
    
    return encoder.encode(np_prediction, content_type)

    
#     if type(prediction) == torch.Tensor:
#         print("got prediction")
# #         prediction = transforms.Resize((650,650))(Prd[0])
#         prediction = prediction.detach().cpu().numpy()
#         print("shape")
#         print(prediction.shape)
#         prediction = prediction.detach().cpu().numpy().tolist()
        
#     return "success"

#     for content_type in utils.parse_accept(accept):
#         if content_type in encoder.SUPPORTED_CONTENT_TYPES:
#             encoded_prediction = encoder.encode(prediction, content_type)
#             if content_type == content_types.CSV:
#                 encoded_prediction = encoded_prediction.encode("utf-8")
#             return encoded_prediction

#     raise errors.UnsupportedFormatError(accept)

    
def model_fn(model_dir):
    num_classes = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
#     Net = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)  # Load net
    
    Net.classifier = DeepLabHead(2048, num_classes)
#     Net.load_state_dict(torch.load(modelPath)) # Load trained model

    model = torch.nn.DataParallel(Net)
      
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
        
    model.eval()
    return model.to(device)

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e4, metavar="LR", help="learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())

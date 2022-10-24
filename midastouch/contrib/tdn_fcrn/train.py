# Original source: https://github.com/XPFly1989/FCRN
# A Pytorch implementation of Laina, Iro, et al. "Deeper depth prediction with fully convolutional residual networks."
# 3D Vision (3DV), 2016 Fourth International Conference on. IEEE, 2016.

# Copyright (c) 2016, Iro Laina
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) Meta Platforms, Inc. All Rights Reserved

"""
Loads TACTO training data and trains the TDN
"""

import torch
import torch.utils.data
from midastouch.contrib.tdn_fcrn.data_loader import data_loader, real_data_loader
import numpy as np
import os
from os import path as osp
from torch.autograd import Variable
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plot
from torch.utils.tensorboard import SummaryWriter
from weights import load_weights
from tqdm import tqdm
from midastouch.render.digit_renderer import pixmm
from midastouch.modules.misc import DIRS
from midastouch.contrib.tdn_fcrn.fcrn import FCRN_net
import hydra
from omegaconf import DictConfig
import time

dtype = torch.cuda.FloatTensor


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    batch_size, learning_rate, num_epochs = cfg.batch_size, cfg.lr, cfg.max_epochs
    resume_from_file = cfg.resume_from_file
    checkpoint_path = osp.join(DIRS["weights"], cfg.checkpoint_weights)
    checkpoint_save_path = osp.join(
        DIRS["weights"], time.strftime("%Y%m%d_%H") + "_" + cfg.checkpoint_weights
    )
    print(f"Saving checkpoint to {checkpoint_save_path}")
    # momentum, weight_decay = 0.9, 0.0005

    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    results_path = osp.join(DIRS["debug"], "tdn_train")

    data_file_path = cfg.data_file_path
    train_data_file = osp.join(data_file_path, "train_data.txt")
    dev_data_file = osp.join(data_file_path, "dev_data.txt")
    train_label_file = osp.join(data_file_path, "train_label.txt")
    dev_label_file = osp.join(data_file_path, "dev_label.txt")
    test_data_file = osp.join(data_file_path, "test_data.txt")
    test_label_file = osp.join(data_file_path, "test_label.txt")

    print(f"Loading data, Resume training: {resume_from_file}")
    train_loader = torch.utils.data.DataLoader(
        data_loader(train_data_file, train_label_file),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        data_loader(dev_data_file, dev_label_file),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        data_loader(test_data_file, test_label_file),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    ## test with real data
    # test_real_file = osp.join(data_file_path,'test_data_real.txt')
    # test_loader = torch.utils.data.DataLoader(real_data_loader(test_real_file), batch_size=batch_size, shuffle=False, drop_last=True)

    print("Loading model...")
    model = FCRN_net(batch_size=batch_size)
    model = model.cuda()

    loss_fn = torch.nn.MSELoss().cuda()

    input_path = osp.join(results_path, "input")
    gt_path = osp.join(results_path, "gt")
    pred_path = osp.join(results_path, "pred")

    if not osp.exists(input_path):
        os.makedirs(input_path)
    if not osp.exists(gt_path):
        os.makedirs(gt_path)
    if not osp.exists(pred_path):
        os.makedirs(pred_path)

    writer = SummaryWriter("train_log")

    start_epoch = 0
    if resume_from_file:
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    checkpoint_path, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
    else:
        # curl -O http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy
        weights_file = osp.join(DIRS["weights"], "NYU_ResNet-UpProj.npy")
        print("=> loading pre-trained NYU weights'{}'".format(weights_file))
        model.load_state_dict(load_weights(model, weights_file, dtype))

    # validate
    print("Validating on sim data")
    model.eval()
    num_samples, loss_local = 0, 0
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, depth in val_loader:
            input_var = Variable(input.type(dtype))
            gt_var = Variable(depth.type(dtype))
            output = model(input_var)
            loss_local += loss_fn(output, gt_var)
            num_samples += 1
            pbar.update(1)
        pbar.close()

    best_val_err = np.sqrt(float(loss_local) / num_samples)
    print("Before train error: {:.3f} pixel RMSE".format(best_val_err))

    for epoch in range(num_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        print("Starting train epoch %d / %d" % (start_epoch + epoch + 1, num_epochs))
        model.train()
        running_loss, count, epoch_loss = 0, 0, 0

        pbar = tqdm(total=len(train_loader))
        # for i, (input, depth) in enumerate(train_loader):
        for input, depth in train_loader:
            input_var = Variable(input.type(dtype))
            gt_var = Variable(depth.type(dtype))

            output = model(input_var)
            loss = loss_fn(output, gt_var)

            # print('loss:', loss.item())
            # input_img = input_var.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            # output_img = output.squeeze().detach().cpu().numpy()
            running_loss += loss.data.cpu().numpy()
            count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_description(
                "RMSE pixel loss: {:.2f}".format(np.sqrt(running_loss / count))
            )
        pbar.close()

        # TODO: tensorboard
        epoch_loss = np.sqrt(running_loss / count)
        print("Epoch error: {:.3f} pixel RMSE".format(epoch_loss))

        writer.add_scalar("train_loss", epoch_loss, start_epoch + epoch + 1)

        # validate
        print("Validating on sim data")
        model.eval()
        num_samples, loss_local = 0, 0
        with torch.no_grad():
            pbar = tqdm(total=len(val_loader))
            for input, depth in val_loader:
                input_var = Variable(input.type(dtype))
                gt_var = Variable(depth.type(dtype))

                output = model(input_var)
                loss_local += loss_fn(output, gt_var)
                num_samples += 1
                pbar.update(1)
            pbar.close()

        err = np.sqrt(float(loss_local) / num_samples)
        print(
            "Validation error: {:.3f} pixel RMSE, Best validation error: {:.3f} pixel RMSE".format(
                err, best_val_err
            )
        )
        writer.add_scalar("val_loss", err, start_epoch + epoch + 1)

        if err < best_val_err:
            print("Saving new checkpoint: {}".format(checkpoint_save_path))
            best_val_err = err
            torch.save(
                {
                    "epoch": start_epoch + epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_save_path,
            )
        else:
            learning_rate = learning_rate * 0.6
            print(
                "No reduction of validation error, dropping learning rate to {}".format(
                    learning_rate
                )
            )

        if (epoch > 0) and (epoch % 10 == 0):
            learning_rate = learning_rate * 0.6
            print("10 epochs, dropping learning rate to {}".format(learning_rate))

        print("Testing on sim data")
        model.eval()
        num_samples, loss_local = 0, 0
        # make local IoU
        with torch.no_grad():
            pbar = tqdm(total=len(test_loader))
            for input, depth in test_loader:
                input_var = Variable(input.type(dtype))
                gt_var = Variable(depth.type(dtype))

                output = model(input_var)

                if num_samples == 0:
                    input_rgb_image = (
                        input_var[0]
                        .data.permute(1, 2, 0)
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )
                    gt_image = gt_var[0].data.squeeze().cpu().numpy().astype(np.float32)
                    pred_image = (
                        output[0].data.squeeze().cpu().numpy().astype(np.float32)
                    )
                    gt_image /= np.max(gt_image)
                    pred_image /= np.max(pred_image)

                    plot.imsave(
                        osp.join(
                            input_path,
                            "input_epoch_{}.png".format(start_epoch + epoch + 1),
                        ),
                        input_rgb_image,
                    )
                    plot.imsave(
                        osp.join(
                            gt_path, "gt_epoch_{}.png".format(start_epoch + epoch + 1)
                        ),
                        gt_image,
                        cmap="viridis",
                    )
                    plot.imsave(
                        osp.join(
                            pred_path,
                            "pred_epoch_{}.png".format(start_epoch + epoch + 1),
                        ),
                        pred_image,
                        cmap="viridis",
                    )
                loss_local += loss_fn(output, gt_var)
                num_samples += 1
                pbar.update(1)
            pbar.close()
        err = np.sqrt(float(loss_local) / num_samples) * pixmm
        print(f"Test error: {err:.3f} mm RMSE")
        writer.add_scalar("test_loss", err, start_epoch + epoch + 1)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()

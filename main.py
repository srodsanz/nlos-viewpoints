import argparse
import os
import logging

from tqdm import tqdm

from model import DEVICE_DEFAULT, SPEED_OF_LIGHT, EPOCHS_DEFAULT, \
    LOSS_LITERALS, TIME_BINS_DEFAULT, DELTA_METERS_DEFAULT, VOLUME_SAMPLES_DEFAULT


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="store", type=str, required=True)
    parser.add_argument("--epochs", action="store", type=int, default=EPOCHS_DEFAULT,
                        help="Number of epochs for trainning")
    parser.add_argument("--delta", action="store", type=float, default=DELTA_METERS_DEFAULT,
                        help="Delta bin discriminant for time discretization")
    parser.add_argument("--device", action="store", type=int, default=DEVICE_DEFAULT,
                        help="Device URI for instantiation")
    parser.add_argument("--tbins", action="store", type=int, default=TIME_BINS_DEFAULT,
                        help="Number of time bins for input transients")
    parser.add_argument("--loss", action="store", type=str, choices=LOSS_LITERALS,
                        help="String identifier for loss function")
    parser.add_argument("--vsamples", action="store", type=int, default=VOLUME_SAMPLES_DEFAULT,
                        help="Number of samples for 2D sampling in Lightfield coordinates")

    args = parser.parse_args()
    path = args.path
    n_epochs = args.epochs
    delta_bins = args.delta 
    device = args.device 
    nt_bins = args.tbins 
    loss = args.loss
    volume_samples = args.vsamples


    




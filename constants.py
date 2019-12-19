#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Directory to save models
SAVE_DIR = "saved_models"
# Directory to save plots
PLOT_DIR = "plots"
# Directory to save logs
LOG_DIR = "logs"
# Directory to save options
OPTIONS_DIR = "options"
# Directory to save images
IMAGES_SAVE_DIR = "reconstructions"

# Default values for command arguments
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_ANNEAL_TEMPERATURE = 8 # Anneal Alpha
DEFAULT_ALPHA = 0.005# Scaling factor for reconstruction loss
DEFAULT_DATASET = "yqdataset"
DEFAULT_DECODER = "Conv" # 'FC' or 'Conv'
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 1000 # DEFAULT_EPOCHS = 300
DEFAULT_USE_GPU = True
DEFAULT_ROUTING_ITERATIONS = 3
DEFAULT_VALIDATION_SIZE = 1000

# Random seed for validation split
VALIDATION_SEED = 889256487
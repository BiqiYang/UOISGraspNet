#!/usr/bin/env python
# coding: utf-8

# # Train a Depth Seeding Network

# In[ ]:


import os
from time import time

import numpy as np
import matplotlib.pyplot as plt

# My libraries
import src.data_loader_graspnet as data_loader
import src.segmentation as segmentation
import src.train as train
import src.util.utilities as util_
import src.util.flowlib as flowlib

os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU


# # Example Dataset: TableTop Object Dataset (TOD)

# In[ ]:


TOD_filepath = '/research/d6/gds/bqyang/object_localization_network/data/graspnet/' # TODO: change this to the dataset you want to train on
data_loading_params = {
    
    # Camera/Frustum parameters
    'img_width' : 1280, 
    'img_height' : 720,
    'near' : 0.01,
    'far' : 100,
    'fov' : 45, # vertical field of view in degrees
    
    'use_data_augmentation' : True,

    # Multiplicative noise
    'gamma_shape' : 1000.,
    'gamma_scale' : 0.001,
    
    # Additive noise
    'gaussian_scale_range' : [0., 0.003], # up to 2.5mm standard dev
    'gp_rescale_factor_range' : [12, 20], # [low, high (exclusive)]
    
    # Random ellipse dropout
    'ellipse_dropout_mean' : 10, 
    'ellipse_gamma_shape' : 5.0, 
    'ellipse_gamma_scale' : 1.0,

    # Random high gradient dropout
    'gradient_dropout_left_mean' : 15, 
    'gradient_dropout_alpha' : 2., 
    'gradient_dropout_beta' : 5.,

    # Random pixel dropout
    'pixel_dropout_alpha' : 0.2, 
    'pixel_dropout_beta' : 10.,
}

dl = data_loader.get_TOD_train_dataloader(TOD_filepath, data_loading_params, batch_size=1, num_workers=2, shuffle=True)


# ## Train Depth Seeding Network

# In[ ]:


dsn_config = {
    
    # Sizes
    'feature_dim' : 64, # 32 would be normal

    # Mean Shift parameters (for 3D voting)
    'max_GMS_iters' : 10, 
    'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
    'epsilon' : 0.05, # Connected Components parameter
    'sigma' : 0.02, # Gaussian bandwidth parameter
    'subsample_factor' : 5,
    'min_pixels_thresh' : 500,
    
    # Differentiable backtracing params
    'tau' : 15.,
    'M_threshold' : 0.3,
    
    # Robustness stuff
    'angle_discretization' : 100,
    'discretization_threshold' : 0.,

}

tb_dir = './' # TODO: change this to desired tensorboard directory
dsn_training_config = {
    
    # Training parameters
    'lr' : 1e-4, # learning rate
    'iter_collect' : 20, # Collect results every _ iterations
    'max_iters' : 150000,
    
    # Loss function stuff
    'lambda_fg' : 3.,
    'lambda_co' : 5., 
    'lambda_sep' : 1., 
    'lambda_cl' : 1.,
    'num_seeds_training' : 50, 
    'delta' : 0.1, # for clustering loss. 2*eps
    'max_GMS_iters' : 10, 

    # Tensorboard stuff
    'tb_directory' : os.path.join(tb_dir, 'train_DSN/'),
    'flush_secs' : 10, # Write tensorboard results every _ seconds
}

iter_num = 0
dsn_training_config.update({
    # Starting optimization from previous checkpoint
    'load' : False,
    'opt_filename' : os.path.join(dsn_training_config['tb_directory'],
                                  f'DSNTrainer_DSNWrapper_{iter_num}_checkpoint.pth'),
    'model_filename' : os.path.join(dsn_training_config['tb_directory'],
                                    f'DSNTrainer_DSNWrapper_{iter_num}_checkpoint.pth'),
})

dsn = segmentation.DSNWrapper(dsn_config)
dsn_trainer = train.DSNTrainer(dsn, dsn_training_config)

# Train the network for 1 epoch
num_epochs = 12
dsn_trainer.train(num_epochs, dl)
dsn_trainer.save()

# ## Visualize some stuff
dl = data_loader.get_TOD_test_dataloader(TOD_filepath, data_loading_params, batch_size=8, num_workers=8, shuffle=True)
dl_iter = dl.__iter__()

batch = next(dl_iter)
rgb_imgs = util_.torch_to_numpy(batch['rgb'], is_standardized_image=True) # Shape: [N x H x W x 3]
xyz_imgs = util_.torch_to_numpy(batch['xyz']) # Shape: [N x H x W x 3]
foreground_labels = util_.torch_to_numpy(batch['foreground_labels']) # Shape: [N x H x W]
center_offset_labels = util_.torch_to_numpy(batch['center_offset_labels']) # Shape: [N x 2 x H x W]
N, H, W = foreground_labels.shape[:3]

print("Number of images: {0}".format(N))

dsn.eval_mode()

### Compute segmentation masks ###
st_time = time()
fg_masks, center_offsets, object_centers, initial_masks = dsn.run_on_batch(batch)
total_time = time() - st_time
print('Total time taken for Segmentation: {0} seconds'.format(round(total_time, 3)))
print('FPS: {0}'.format(round(N / total_time,3)))

# Get segmentation masks in numpy
fg_masks = fg_masks.cpu().numpy()
center_offsets = center_offsets.cpu().numpy().transpose(0,2,3,1)
initial_masks = initial_masks.cpu().numpy()
for i in range(len(object_centers)):
    object_centers[i] = object_centers[i].cpu().numpy()

fig_index = 1
for i in range(N):
    
    fig = plt.figure(fig_index); fig_index += 1
    fig.set_size_inches(20,5)

    # Plot image
    plt.subplot(1,5,1)
    plt.imshow(rgb_imgs[i,...].astype(np.uint8))
    plt.title('Image {0}'.format(i+1))

    # Plot Depth
    plt.subplot(1,5,2)
    plt.imshow(xyz_imgs[i,...,2])
    plt.title('Depth')
    
    # Plot prediction
    plt.subplot(1,5,3)
    plt.imshow(util_.get_color_mask(fg_masks[i,...]))
    plt.title("Predicted Masks")
    
    # Plot Center Direction Predictions
    plt.subplot(1,5,4)
    fg_mask = np.expand_dims(fg_masks[i,...] == 2, axis=-1)
    plt.imshow(flowlib.flow_to_image(direction_predictions[i,...] * fg_mask))
    plt.title("Center Direction Predictions")
    
    # Plot Initial Masks
    plt.subplot(1,5,5)
    plt.imshow(util_.get_color_mask(initial_masks[i,...]))
    plt.title(f"Initial Masks. #objects: {np.unique(initial_masks[i,...]).shape[0]-1}")


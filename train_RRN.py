#!/usr/bin/env python
# coding: utf-8

# # Train a Region Refinement Network

# In[ ]:


import os
from time import time

import numpy as np
import matplotlib.pyplot as plt

# My libraries
import src.data_loader as data_loader
import src.segmentation as segmentation
import src.train as train
import src.util.utilities as util_
import src.util.flowlib as flowlib

os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU


# # Example Dataset: TableTop Object Dataset (TOD)

# In[ ]:


TOD_filepath = '...' # TODO: change this to the dataset you want to train on
data_loading_params = {
    
    'max_augmentation_tries' : 10,
    
    # Padding
    'padding_alpha' : 1.0,
    'padding_beta' : 4.0, 
    'min_padding_percentage' : 0.05, 
    
    # Erosion/Dilation
    'rate_of_morphological_transform' : 0.9,
    'label_dilation_alpha' : 1.0,
    'label_dilation_beta' : 19.0,
    'morphology_max_iters' : 3,
    
    # Ellipses
    'rate_of_ellipses' : 0.8,
    'num_ellipses_mean' : 50,
    'ellipse_gamma_base_shape' : 1.0, 
    'ellipse_gamma_base_scale' : 1.0,
    'ellipse_size_percentage' : 0.025,
    
    # Translation
    'rate_of_translation' : 0.7,
    'translation_alpha' : 1.0,
    'translation_beta' : 19.0,
    'translation_percentage_min' : 0.05,
    
    # Rotation
    'rate_of_rotation' : 0.7,
    'rotation_angle_max' : 10, # in degrees
    
    # Label Cutting
    'rate_of_label_cutting' : 0.3,
    'cut_percentage_min' : 0.25,
    'cut_percentage_max' : 0.5,
    
    # Label Adding
    'rate_of_label_adding' : 0.5,
    'add_percentage_min' : 0.1,
    'add_percentage_max' : 0.4,
    
}
dl = data_loader.get_Synth_RGBO_train_dataloader(TOD_filepath, data_loading_params, batch_size=16, num_workers=8, shuffle=True)


# ## Train Region Refinement Network

# In[ ]:


rrn_config = {
    
    # Sizes
    'feature_dim' : 64, # 32 would be normal
    'img_H' : 224,
    'img_W' : 224,
    
    # architecture parameters
    'use_coordconv' : False,
    
}

tb_dir = ... # TODO: change this to desired tensorboard directory
rrn_training_config = {
    
    # Training parameters
    'lr' : 1e-4, # learning rate
    'iter_collect' : 20, # Collect results every _ iterations
    'max_iters' : 100000,
    
    # Tensorboard stuff
    'tb_directory' : tb_dir + 'Trainer_test' + '/',
    'flush_secs' : 10, # Write tensorboard results every _ seconds
}

iter_num = 0
rrn_training_config.update({
    # Starting optimization from previous checkpoint
    'load' : False,
    'opt_filename' : os.path.join(rrn_training_config['tb_directory'],
                                  f'RRNTrainer_RRNWrapper_{iter_num}_checkpoint.pth'),
    'model_filename' : os.path.join(rrn_training_config['tb_directory'],
                                    f'RRNTrainer_RRNWrapper_{iter_num}_checkpoint.pth'),
})


# In[ ]:


rrn = segmentation.RRNWrapper(rrn_config)
rrn_trainer = train.RRNTrainer(rrn, rrn_training_config)


# In[ ]:


# Train the network for 1 epoch
num_epochs = 1
rrn_trainer.train(num_epochs, dl)
rrn_trainer.save()


# ## Plot some losses

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(1, figsize=(15,3))
total_subplots = 1
starting_epoch = 0
info_items = {k:v for (k,v) in rrn.infos.items() if k > starting_epoch}

plt.subplot(1,total_subplots,1)
losses = [x['loss'] for (k,x) in info_items.items()]
plt.plot(info_items.keys(), losses)
plt.xlabel('Iteration')
plt.title('Losses. {0}'.format(losses[-1]))

print("Number of iterations: {0}".format(rrn.iter_num))


# ## Visualize some stuff
# 
# Run the network on a single batch, and plot the results

# In[ ]:


dl = data_loader.get_Synth_RGBO_train_dataloader(TOD_filepath, data_loading_params, batch_size=16, num_workers=8, shuffle=True)
dl_iter = dl.__iter__()

batch = next(dl_iter)
rgb_imgs = util_.torch_to_numpy(batch['rgb'], is_standardized_image=True) # Shape: [N x H x W x 3]
initial_masks = util_.torch_to_numpy(batch['initial_masks']) # Shape: [N x H x W]
labels = util_.torch_to_numpy(batch['labels']) # Shape: [N x H x W]
N, H, W = labels.shape[:3]


# In[ ]:


print("Number of images: {0}".format(N))

rrn.eval_mode()

### Compute segmentation masks ###
st_time = time()
seg_masks = rrn.run_on_batch(batch)
total_time = time() - st_time
print('Total time taken for Segmentation: {0} seconds'.format(round(total_time, 3)))
print('FPS: {0}'.format(round(N / total_time,3)))

# Get segmentation masks in numpy
seg_masks = seg_masks.cpu().numpy()


# In[ ]:


num_colors = 2
fig_index = 1
for i in range(N):
    
    fig = plt.figure(fig_index); fig_index += 1
    fig.set_size_inches(20,5)

    # Plot image
    plt.subplot(1,4,1)
    plt.imshow(rgb_imgs[i].astype(np.uint8))
    plt.title('Image {0}'.format(i+1))
    
    # Plot initial mask
    plt.subplot(1,4,2)
    plt.imshow(util_.get_color_mask(initial_masks[i]))
    plt.title("Initial Mask")
    
    # Plot labels
    plt.subplot(1,4,3)
    plt.imshow(util_.get_color_mask(labels[i], nc=num_colors))
    plt.title(f"GT Masks")
    
    # Plot prediction
    plt.subplot(1,4,4)
    plt.imshow(util_.get_color_mask(seg_masks[i,...], nc=num_colors))
    plt.title(f"Predicted Masks")


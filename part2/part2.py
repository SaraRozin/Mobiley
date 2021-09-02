#!/usr/bin/env python
# coding: utf-8

# ## Hands on lecture: train TFL net
# 
#  In this hands-on lecture, we will train a model for predicting Traffic-light (TFL) in image patches, you will use the data generated in previous course to this end.
#  You will :
#  1. Evaluate the data previously generated and ensure it's validity.
#  2. Train a CNN 
#  3. Evaluate results of the trained model.
#  4. seek ways to improve results



#
# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('autosave', '120')
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np, matplotlib.pyplot as plt
from os.path import join


# In[ ]:



# ## Step 1. Validate your data
#     use the example in the cell below, to verify the TFL patch you've generated is sane.
#     Things to watch for:
#     1. You are able to load and vizualize your train and val data, using the functions below.
#     2. using the vizualization verify  image <--> label correspondence is correct.
#     3. % Negative vs. Positive examples is aprroximately 50%
#  

# In[ ]:


def load_tfl_data(data_dir, crop_shape=(81,81)):
    images = np.memmap(join(data_dir,'data.bin'),mode='r',dtype=np.uint8).reshape([-1]+list(crop_shape) +[3])
    labels = np.memmap(join(data_dir,'labels.bin'),mode='r',dtype=np.uint8)
    return {'images':images,'labels':labels}

def viz_my_data(images,labels, predictions=None, num=(5,5), labels2name= {0:'No TFL',1:'Yes TFL'}):
    assert images.shape[0] == labels.shape[0]
    assert predictions is None or predictions.shape[0] == images.shape[0]
    h = 5
    n = num[0]*num[1]
    ax = plt.subplots(num[0],num[1],figsize=(h*num[0],h*num[1]),gridspec_kw={'wspace':0.05},squeeze=False,sharex=True,sharey=True)[1]#.flatten()
    idxs = np.random.randint(0,images.shape[0],n)
    for i,idx in enumerate(idxs):
        ax.flatten()[i].imshow(images[idx])
        title = labels2name[labels[idx]]
        if predictions is not None : title += ' Prediction: {:.2f}'.format(predictions[idx])
        ax.flatten()[i].set_title(title)
   

# root = './'  #this is the root for your val and train datasets
data_dir = "./Data_dir/"
datasets = {
    'val':load_tfl_data(join(data_dir,'val')),
    'train': load_tfl_data(join(data_dir,'train')),
    }
for k,v in datasets.items():
    
    print ('{} :  {} 0/1 split {:.1f} %'.format(k,v['images'].shape, np.mean(v['labels']==1)*100))

viz_my_data(num=(6,6),**datasets['val'])   


# In[ ]:


def tfl_model():
    from keras.applications.vgg16 import VGG16
    model = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(81, 81, 3)))
    return model
m = tfl_model()
m.summary()


# ## define the model used for training
# 

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Activation,MaxPooling2D,BatchNormalization,Activation, Conv2D

def tfl_model():
    input_shape =(81,81,3)
    
    model = Sequential()
    def conv_bn_relu(filters,**conv_kw):
        model.add(Conv2D(filters,  use_bias=False, kernel_initializer='he_normal',**conv_kw))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    def dense_bn_relu(units):
        model.add(Dense(units, use_bias=False,kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    
    def spatial_layer(count, filters): 
        for i in range(count):
            conv_bn_relu(filters,kernel_size=(3,3))
        conv_bn_relu(filters, kernel_size=(3,3),strides=(2,2))
    
    conv_bn_relu(32,kernel_size=(3,3),input_shape=input_shape)
    spatial_layer(1,32) 
    spatial_layer(2,64)
    spatial_layer(2,96) 
    
    
    model.add(Flatten())
    dense_bn_relu(96)
    model.add(Dense(2, activation='softmax'))
    return model
m = tfl_model()
m.summary()


# In[ ]:


from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(81,81,3))
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)
restnet = Model(restnet.input, outputs=output)
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()


# In[ ]:


import tensorflow as tf

def dropblock(x, keep_prob, block_size):
    _,w,h,c = x.shape.as_list()
    gamma = (1. - keep_prob) * (w * h) / (block_size ** 2) / ((w - block_size + 1) * (h - block_size + 1))
    sampling_mask_shape = tf.stack([1, h - block_size + 1, w - block_size + 1, c])
    noise_dist = tf.compat.v1.distributions.Bernoulli(probs=gamma)
    mask = noise_dist.sample(sampling_mask_shape)

    br = (block_size - 1) // 2
    tl = (block_size - 1) - br
    pad_shape = [[0, 0], [tl, br], [tl, br], [0, 0]]
    mask = tf.pad(mask, pad_shape)
    mask = tf.nn.max_pool(mask, [1, block_size, block_size, 1], [1, 1, 1, 1], 'SAME')
    mask = tf.cast(1 - mask,tf.float32)
    return tf.multiply(x,mask)


# In[ ]:


[n,w,h,c] = [5,5,5,3]
tf.compat.v1.disable_eager_execution()
ph = tf.compat.v1.placeholder(shape=[None,w,h,c],dtype=tf.float32)
model = dropblock(ph,0.1,3)
out = sess.run(model,feed_dict={ph:input})
print(out[0,:,:,0])


# In[ ]:


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import regularizers
model = Sequential()
# model.add(d_model)
model.add(restnet)
model.add(Dense(512/4, activation='relu', input_dim=(81,81,3)))
model.add(Dropout(0.3))
model.add(Dense(512/4,activation=('relu'),input_dim=(81,81,3)))
model.add(Dropout(0.3))
# model.add(Dense(256,activation=('relu'))) 
# model.add(Dropout(0.3))
# model.add(Dense(512,activation=('relu'))) 
# model.add(Dropout(0.3))
# model.add(Dense(128,activation=('relu')))
# model.add(Dropout(0.3))
# model.add(Dense(10,activation=('softmax'))) 
# model.add(Flatten())
# model.add(Dense(512/4, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(512/4, activation='relu'))
# model.add(Dropout(0.3))

# model.add(Dense(2, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#     bias_regularizer=regularizers.l2(1e-4),
#     activity_regularizer=regularizers.l2(1e-5)))
model.add(Dense(2, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',
#               optimizer=Adam(),
#               metrics=['accuracy'])
model.summary()


# 

# ## train

# In[ ]:


from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
data_dir = './'
datasets = {
    'val':load_tfl_data(join(data_dir,'val')),
    'train': load_tfl_data(join(data_dir,'train')),
    }
#prepare our model
# m = tfl_model()
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
lr_schedule = InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)
model.compile(optimizer=Adam(),loss =sparse_categorical_crossentropy,metrics=['accuracy'])
train,val = datasets['train'],datasets['val']
#train it, the model uses the 'train' dataset for learning. We evaluate the "goodness" of the model, by predicting the label of the images in the val dataset.
history=model.fit(train['images'],train['labels'],validation_data=(val['images'],val['labels']),epochs =4)


# In[ ]:


#compare train vs val acccuracy, 
# why is val_accuracy not as good as train accuracy? are we overfitting?
epochs = history.history
epochs['train_accuracy'] = epochs['accuracy']
plt.figure(figsize=(10,10))
for k in ['train_accuracy','val_accuracy']:
    plt.plot(range(len(epochs[k])), epochs[k],label=k)

plt.legend();


# In[ ]:


epochs = history.history
epochs['train_loss'] = epochs['loss']
plt.figure(figsize=(10,10))
for k in ['train_loss','val_loss']:
    plt.plot(range(len(epochs[k])), epochs[k],label=k)

plt.legend();


# ## evaluate and predict
# Now thet we have a model we can use to predict results on the validation dataset.
# 1. What can say about example that fail prediction? can we find patterns that are common for failure cases?

# In[ ]:


import seaborn as sbn
predictions = model.predict(val['images'])
sbn.distplot(predictions[:,0]);

predicted_label = np.argmax(predictions, axis=-1)
print ('accuracy:', np.mean(predicted_label==val['labels']))


# In[ ]:


viz_my_data(num=(6,6),predictions=predictions[:,1],**val);


# ### Seek ways to improve resutls 
# 1. Try to play with diffferent models , increase / decrease the number of conv layers or number of  filters. you'll need to find a balanced model that is sufficiently large but minimzes overfit
#     - try to enable tensorboard vizualization (see keras/ tensorboard doc) to help you carry the analysis
# 2. Data augmentation: feed your network with more examples by using data augmentation techniques: such as horizontal image flip, noise, etc

# ### Saving the model
# After we trained our model and made predictions with it, we will now want to save the **architecture** together with its learned **weights** in order for us to be able to use it in the TFL manager.
# The architecture will be saved as a json, the weights in the h5 format: 

# In[ ]:


model.save("/content/drive/MyDrive/Colab Notebooks/model.h5")


# In[ ]:


# If you want to make sure that this model can be used on different operating systems and different
# versions of keras or tensorflow, this is the better way to save. For this project the simpler
# method above should work fine.

# json_filename = 'model.json'
# h5_filename   = 'weights.h5'
# # create a json with the model architecture
# model_json = m.to_json()
# # save the json to disk
# with open(json_filename, 'w') as f:
#     f.write(model_json)
# # save the model's weights:
# m.save_weights(h5_filename)
# print(" ".join(["Model saved to", json_filename, h5_filename]))


# ### Loading the model

# In[ ]:


from tensorflow.keras.models import load_model
loaded_model = load_model("/content/drive/MyDrive/Colab Notebooks/model.h5")


# In[ ]:


# If you use the more robust method of saving above, this is how you load the model.

# with open(json_filename, 'r') as j:
#     loaded_json = j.read()

# # load the model architecture: 
# loaded_model = keras.models.model_from_json(loaded_json)
# #load the weights:
# loaded_model.load_weights(h5_filename)
# print(" ".join(["Model loaded from", json_filename, h5_filename]))


# In[ ]:


# code copied from the training evaluation: 
train,val = datasets['train'],datasets['val']
l_predictions = loaded_model.predict(val['images'])
sbn.distplot(l_predictions[:,0]);

l_predicted_label = np.argmax(l_predictions, axis=-1)
print ('accuracy:', np.mean(l_predicted_label==[val['labels']]))


# In[ ]:


val['images'].shape


# In[ ]:


loaded_model.predict(val['images'])


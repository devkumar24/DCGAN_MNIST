import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import math
import json

from keras.layers import *
from keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from keras.datasets import mnist



import model
from preprocess_data import generate_data



TOTAL_EPOCHS = 20
BATCH_SIZE = 256
HALF_BATCH = int(BATCH_SIZE/2)
NOISE_DIM = 100

adam = Adam(lr = 2e-4,beta_1 = 0.5)



gen = model.generator(NOISE_DIM,adam)
dis = model.discrimator(adam)



def display_images(fp:str="",samples = 100):
    folder = 'fake_data'
    noise = np.random.normal(0,1, size = (samples, NOISE_DIM))
    
    generated_img = gen.predict(noise)
    
    
    plt.figure(figsize=(10,10))
    
    for i in range(samples):
        plt.subplot(10,10,i+1)
        plt.imshow(generated_img[i].reshape(28,28),interpolation='nearest',cmap='gray')
        plt.axis("off")
        
    plt.show()
    plt.savefig(folder+fp)



def save_loss(gen_loss,dis_loss,gen_name : str = "", dis_name : str = ""):

	if os.path.isdir(gen_name):

		with open(gen_name,'w') as g:
			json.dumps(gen_loss,g)

	else:
		os.mknod(gen_name)


	if os.path.isdir(dis_name):
		with open(dis_name,'w') as d:
			json.dumps(dis_loss,d)
	

	else:
		os.mknod(dis_name)



def plot_loss(gen_loss,dis_loss):

	gen_key = list(gen_loss.keys())
	gen_value = list(gen_loss.values())

	dis_key = list(dis_loss.keys())
	dis_value = list(dis_loss.values())

	plt.plot(gen_key,gen_value,label = 'Generator Loss')
	plt.plot(dis_key,dis_value, label = 'Discriminator Loss')
	plt.title('LOSS vs EPOCH')
	plt.xlabel('EPOCHS')
	plt.ylabel('LOSS')
	plt.legend()
	plt.savefig('LOSS GRAPH')




def show_images(fp : str = 'fake_data'):

    if os.listdir(fp) is None:
        raise TypeError("File Not Found")

    else:
        plt.figure(figsize=(100,100))
        images = os.listdir(fp)

        for i in range(len(images)):
            plt.subplot(10,10,i+1)
            plt.imshow(images[i],interpolation='nearest',cmap='gray')
            plt.axis("off")





if __name__ == '__main__':
	data = generate_data()
	dis.trainable = False
	## Input and Output
	DCGAN_INPUT = Input((NOISE_DIM,))
	GENERATED_IMAGE = gen(DCGAN_INPUT)
	DCGAN_OUTPUT = dis(GENERATED_IMAGE)

	# Functional API
	model = Model(DCGAN_INPUT,DCGAN_OUTPUT)
	model.compile(loss = 'binary_crossentropy',optimizer = adam)


	NO_BATCHES = math.ceil(data.shape[0]/float(BATCH_SIZE))

	## Training loop


	generator_loss = {}
	discriminator_loss = {}

	for epoch in range(TOTAL_EPOCHS):
	    
	    generator_epoch_loss = 0.0
	    discrimator_epoch_loss = 0.0
	    
	    for batch in range(NO_BATCHES):
	        # STEP_1 Train discriminator
	        
	#         Real Images Data
	        image_idx = np.random.randint(0,data.shape[0],HALF_BATCH)
	        
	        real_image_Data = data[image_idx]
	        
	#         FAke Image Data

	        noise = np.random.normal(loc = 0,scale = 1,size = (HALF_BATCH,NOISE_DIM))
	        fake_image_data = gen.predict(noise)
	        
	#         Train Discriminator
	        
	        #Labels
	        real_Y = np.ones((HALF_BATCH,1))*0.9
	        fake_Y = np.zeros((HALF_BATCH,1))
	        
	        discrimator_loss_real = dis.train_on_batch(real_image_Data,real_Y)
	        discrimator_loss_fake = dis.train_on_batch(fake_image_data,fake_Y)
	        
	        # Total Loss
	        
	        discrimator_epoch_loss += 0.5*(discrimator_loss_fake + discrimator_loss_real)
	        
	        #  STEP_2 Train Generator
	        noise_ = np.random.normal(0,1,(BATCH_SIZE,NOISE_DIM))
	        truth_Y = np.ones((BATCH_SIZE,1))
	        generator_loss_ = model.train_on_batch(noise_,truth_Y)
	        generator_epoch_loss += generator_loss_
	        
	    if discriminator_loss.get(epoch+1) is None:
	        discriminator_loss[epoch+1] = {}
	        discriminator_loss[epoch+1] = discrimator_epoch_loss/NO_BATCHES
	    else:
	        discriminator_loss[epoch+1] = discrimator_epoch_loss/NO_BATCHES
	        
	    if generator_loss.get(epoch+1) is None:
	        generator_loss[epoch+1] = {}
	        generator_loss[epoch+1] = generator_epoch_loss/NO_BATCHES
	    else:
	        generator_loss[epoch+1] = generator_epoch_loss/NO_BATCHES
	    
	    print("EPOCH {} :::: DISCRIMINATOR LOSS {} :::: GENERATOR LOSS {}".format(epoch+1,discrimator_epoch_loss/NO_BATCHES,generator_epoch_loss/NO_BATCHES))
	    
	    if (epoch+1)%10 == 0:
	        gen.save('MNIST_{}.h5'.format(epoch+1))
	        display_images(fp = '{}.png'.format(epoch+1))    
	
	show_images(fp = 'fake_data')
	    


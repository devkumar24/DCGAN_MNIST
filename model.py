from keras.layers import *
from keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam





def generator(NOISE_DIM,adam,return_summary = False):
    gen = Sequential()
    gen.add( Dense(7*7*128, input_shape = (NOISE_DIM,)) )
    gen.add( Reshape((7,7,128)) )
    gen.add( LeakyReLU(0.2) )
    gen.add( BatchNormalization() )
    
    # Convert 7,7,128 => 14,14,64
    gen.add( UpSampling2D() )
    gen.add( Conv2D(64,kernel_size = (3,3),padding='same') )
    gen.add( LeakyReLU(0.2) )
    gen.add( BatchNormalization() )
    
    
    # Convert 14,14,64 => 28,28,1
    gen.add( UpSampling2D() )
    gen.add( Conv2D(1, kernel_size = (3,3), padding='same', activation = 'tanh') )
    
    gen.compile(loss = 'binary_crossentropy', optimizer = adam)
    
    if return_summary:
        print(gen.summary())
        
    return gen




def discrimator(adam,return_summary = False):
    # 28,28,1 => 14,14,64
    dis = Sequential()
    dis.add( Conv2D(64, kernel_size = (3,3), strides = (2,2), padding = 'same', input_shape = (28,28,1)) )
    dis.add( LeakyReLU(0.2) )
    
    # 14,14,64 => 7,7,128
    dis.add( Conv2D(128, kernel_size = (3,3), padding = 'same', strides = (2,2)))
    dis.add( LeakyReLU(0.2) )

    dis.add( Flatten() )
    dis.add( Dense(1, activation='sigmoid') )
    
    dis.compile(loss= 'binary_crossentropy', optimizer=adam)
    
    if return_summary:
        print(dis.summary())
    
    return dis



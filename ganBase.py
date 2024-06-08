from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np


# mnist image shape is 28x28 pixels
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)

# generator neurel network
def build_generator():

    # generator input is randome noise
    # vector of 1x100
    noise_shape = (100,)    

    #sequential inherits class model
    model = Sequential()

    # building generator layers
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    # model.summary()

    noise = Input(shape=noise_shape)
    # image generated 
    img = model(noise)    

    # return model
    return Model(noise, img)


# discriminator network
def build_discriminator():
    # sequential model
    model = Sequential()

    # make input generated image into 1 dimensional
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


def save_imgs(epoch):
    row, col = 5, 5
    noise = np.random.normal(0, 1, (row * col, 100))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(row, col)
    cnt = 0
    for i in range(row):
        for j in range(col):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("baseImages/mnist_%d.png" % epoch)
    plt.close()


def train(epochs, batch_size=128, save_interval=50):

    #load data to train
    (x_train,_), (_,_) = mnist.load_data()


    x_train = (x_train.astype(np.float32) - 127.5)  / 127.5

    x_train = np.expand_dims(x_train, axis=3)


    half_batch = int(batch_size/2)

    # lists to hold losses to plot them
    d_losses = []
    g_losses = []

    for epoch in range(epochs):

        # train the discriminator
        index = np.random.randint(0,x_train.shape[0], half_batch)
        imgs = x_train[index]

        noise = np.random.normal(0,1,(half_batch,100))

        # now generate half batch of images
        generated_imgs = generator.predict(noise)

        #train discriminator on real and fake images seperately
        d_loss_real = discriminator.train_on_batch(imgs,np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_imgs, np.zeros((half_batch, 1)))

        # average loss from real and fake images
        d_loss =  0.5 * np.add(d_loss_real, d_loss_fake)


        # now training my generator taking in random noise to generate images
        noise = np.random.normal(0,1, (batch_size, 100))


        valid_y = np.array([1] *batch_size)

        g_loss = combined.train_on_batch(noise,valid_y)

        d_losses.append(d_loss[0])
        g_losses.append(g_loss)


        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))


        if epoch % save_interval == 0:
            save_imgs(epoch)

    return d_losses, g_losses

def plot_losses(d_losses,g_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label="Discriminator Loss")
        plt.plot(g_losses, label="Generator Loss")
        plt.title("Loss During Training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.savefig('LossPlots/baseplot.png')
        plt.close()
    

##########compiling and training gan

optimizer = Adam(0.0002, 0.5)  

# build discriminator and compile
# using binary cross entropy 
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])


generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

# z is the random noise
z = Input(shape=(100,))   
img = generator(z)
     
discriminator.trainable = False  

valid = discriminator(img)  #Validity check on the generated image

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)


d_losses, g_losses = train(epochs=100, batch_size=32, save_interval = 10)

plot_losses(d_losses,g_losses)
generator.save('models/base_generator_model.h5')
# Author: Ali Kaddoura
# GAN relu model

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import sqrtm
import numpy as np
from scipy.stats import entropy



# mnist image shape is 28x28 pixels
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)

# generator neurel network
def build_generator_relu():
    noise_shape = (100,)
    model = Sequential()

    model.add(Dense(256, input_shape=noise_shape))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(ReLU())
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    noise = Input(shape=noise_shape)
    img = model(noise)

    return Model(noise, img)

# discriminator network
def build_discriminator_relu():
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(ReLU())
    model.add(Dense(256))
    model.add(ReLU())
    model.add(Dense(1, activation='sigmoid'))

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

def calculate_fid(model, real_images, fake_images):
    # calculate activations
    act1 = model.predict(real_images)
    act2 = model.predict(fake_images)

    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def train(epochs, batch_size=128, save_interval=50):

    #load data to train
    (x_train,_), (_,_) = mnist.load_data()


    x_train = (x_train.astype(np.float32) - 127.5)  / 127.5

    x_train = np.expand_dims(x_train, axis=3)


    half_batch = int(batch_size/2)

    # lists to hold losses to plot them
    d_losses = []
    g_losses = []
    fids = []

    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    for epoch in range(epochs):


        # train the discriminator
        index = np.random.randint(0,x_train.shape[0], half_batch)
        imgs = x_train[index]

        noise = np.random.normal(0,1,(half_batch,100))

        # now generate half batch of images
        generated_imgs = generator.predict(noise)

        real_imgs_resized = resize_images((imgs + 1) * 127.5)
        fake_imgs_resized = resize_images((generated_imgs + 1) * 127.5)

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
        
        if epoch == epochs -1:
            noise = np.random.normal(0, 1, (500, 100))  # Adjust number based on your GPU capacity
            images = generator.predict(noise)
            images = resize_images(images)
            mean_is, std_is = inception_score(images)
            print(f"Inception Score: {mean_is} ± {std_is}")
    plot_losses(d_losses, g_losses)
    generator.save('models/base_generator_model.h5')

    
def plot_losses(d_losses,g_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label="Discriminator Loss")
        plt.plot(g_losses, label="Generator Loss")
        plt.title("Loss During Training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('LossPlots/baseplot.png')
        plt.show()
        plt.close()


def plot_fid(fids):
        plt.figure(figsize=(10, 5))
        plt.plot(fids, label="FID score")
        plt.title("FID Score during training")
        plt.xlabel("Epoch")
        plt.ylabel("FID")
        plt.legend()
        plt.savefig('FIdPlots/baseplot.png')
        plt.show()
        plt.close()

def resize_images(imgs):
    resized_imgs = []
    for img in imgs:
        # Rescale to [0, 255] and convert to RGB by repeating the grayscale image across three channels
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        img_rgb = np.stack([img.squeeze()] * 3, axis=-1)
        # Resize image to 299x299 using PIL
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((299, 299), Image.BILINEAR)
        resized_imgs.append(np.array(img_pil))
    return np.array(resized_imgs)


def inception_score(images, n_split=10, eps=1E-16):
    # Load Inception model
    model = InceptionV3(include_top=True)

    # Manually preprocess the images
    images = preprocess_input(images)  # This normalizes the image data to [-1, 1]

    # Get model predictions
    preds = model.predict(images)

    # Calculate Inception Score
    scores = []
    n_part = len(preds) // n_split
    for i in range(n_split):
        part = preds[i * n_part:(i + 1) * n_part]
        kl_div = part * (np.log(part + eps) - np.log(np.expand_dims(np.mean(part, axis=0), axis=0) + eps))
        kl_div = np.mean(np.sum(kl_div, axis=1))
        scores.append(np.exp(kl_div))

    return np.mean(scores), np.std(scores)
    

##########  compiling and training gan  ####################

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

train(epochs=100, batch_size=32, save_interval = 100)

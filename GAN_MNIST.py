
import numpy as np
import tensorflow as tf

#creating the discriminator part of GAN
X=tf.placeholder(tf.float32,shape=[None, 784], name='X')

D_W1 = tf.Variable(xavier_init([784,128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

D_W2 = tf.Variable(xavier_init([128,1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

Disc = [D_W1, D_W2, D_b1, D_b2]

#creating the generator part of GAN

Z=tf.placeholder(tf.float32,shape=[None, 100], name='Z')

G_W1 = tf.Variable(xavier_init([100,128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

G_W2 = tf.Variable(xavier_init([128,784]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

Gene = [G_W1, G_W2, G_b1, G_b2]

#function to create genrator NN

def generator(z):
	G_h1=tf.nn.relu(tf.matmul(z,G_W1)+G_b1)
	G_log_prob=tf.matmul(G_h1,G_W2)+G_b2
	G_prob = tf.nn.sigmoid(G_log_prob)

	return G_prob

def discriminator(x):
	D_h1=tf.nn.relu(tf.matmul(x,D_W1)+D_b1)
	D_logit=tf.matmul(D_h1,D_W2)+D_b2
	D_prob = tf.nn.sigmoid(D_logit)

	return D_prob,D_logit


G_sample=generator(Z)
D_real,D_logit_real=discriminator(X)
D_fake, D_logit_fake=discriminator(G_sample)

D_loss=-tf.reduce_mean(tf.log(D_real)+tf.log(1 - D_fake))
G_loss=-tf.reduce_mean(tf.log(D_fake))

D_solve = tf.train.AdamOptimizer().minimize(D_loss, var_list=Disc)

G_solve = tf.train.AdamOptimizer().minimize(G_loss,var_list=Gene)

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])

for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})








os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'




data_shape = 360*480

class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

train_data = np.load('/data/train_data.npz')
train_label = np.load('/data/train_label.npz')



# load the data

# load the model:

autoencoder.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

# current_dir = os.path.dirname(os.path.realpath(__file__))
# model_path = os.path.join(current_dir, "autoencoder.png")
# plot(model_path, to_file=model_path, show_shapes=True)

nb_epoch = 4
batch_size = 8

history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, class_weight=class_weighting )#, validation_data=(X_test, X_test))

autoencoder.save_weights('model_weight_{}.hdf5'.format(nb_epoch))


# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:43:31 2019

@author: LXI-294-VINU
"""
import problem_unittests as tests
import helper
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile
from django.core.validators import URLValidator
from django.core.exceptions import ValidationError
import time
import numpy as np
import pickle
import tensorflow as tf
import random




def neural_net_image_input(image_shape):
    """
    Return a Tensor for a bach of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
    
    tf.reset_default_graph()

    return tf.placeholder(tf.float32, shape=(None, *image_shape), name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
   
    return tf.placeholder(tf.float32, shape=(None, n_classes), name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
   

    return tf.placeholder(tf.float32, name='keep_prob')


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    
    
    #Define weight
    weight_shape = [*conv_ksize, int(x_tensor.shape[3]), conv_num_outputs]
    w = tf.Variable(tf.random_normal(weight_shape, stddev=0.1))
    
    #Define bias
    b = tf.Variable(tf.zeros(conv_num_outputs))
    
    #Apply convolution
    x = tf.nn.conv2d(x_tensor, w, strides=[1, *conv_strides, 1], padding='SAME')
    
    #Apply bias
    x = tf.nn.bias_add(x, b)
    
    #Apply RELU
    x = tf.nn.relu(x)
    
    #Apply Max pool
    x = tf.nn.max_pool(x, [1, *pool_ksize, 1], [1, *pool_strides, 1], padding='SAME')
    return x

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    batch_size, *fltn_img_size = x_tensor.get_shape().as_list()
    img_size = fltn_img_size[0] * fltn_img_size[1] * fltn_img_size[2]
    tensor = tf.reshape(x_tensor, [-1, img_size])
    return tensor


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    #weights
    w_shape = (int(x_tensor.get_shape().as_list()[1]), num_outputs)
    weights = tf.Variable(tf.random_normal(w_shape, stddev=0.1))
    
    #bias
    bias = tf.Variable(tf.zeros(num_outputs))
    x = tf.add(tf.matmul(x_tensor, weights), bias)
    output = tf.nn.relu(x)
    return output




def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    x = conv2d_maxpool(x, 32, (3, 3), (1, 1), (2, 2), (2, 2))
    x = conv2d_maxpool(x, 32, (3, 3), (2, 2), (2, 2), (2, 2))
    x = conv2d_maxpool(x, 64, (3, 3), (1, 1), (2, 2), (2, 2))
    

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    x = flatten(x)
    

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    x = fully_conn(x, 128)
    x = tf.nn.dropout(x, keep_prob)
    
    
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    result = output(x, 10)
    
    
    # TODO: return output
    return result


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    #weights
    w_shape = (int(x_tensor.get_shape().as_list()[1]), num_outputs)
    weights = tf.Variable(tf.random_normal(w_shape, stddev=0.1))
    
    #bias
    bias = tf.Variable(tf.zeros(num_outputs))
    x = tf.add(tf.matmul(x_tensor, weights), bias)
    return x



tf.reset_default_graph()

    # Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

    # Model
logits = conv_net(x, keep_prob)

    # Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
epochs = 52
batch_size = 256
keep_probability = 0.6
save_model_path = './image_classification'



def test_model(plotCanvas):
    """
    Test the saved model against the test dataset
    """
    try:
        if batch_size:
            pass
    except NameError:
            batch_size = 64
                

    save_model_path = './image_classification'
    n_samples = 4
    top_n_predictions = 3
    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions, plotCanvas)


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
     # TODO: Implement Function
     

 
    valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

    loss = session.run(cost, feed_dict={
        x: feature_batch,
        y: label_batch,
        keep_prob: 1.
    })
    
    valid_accuracy = session.run(accuracy, feed_dict={
        x: valid_features,
        y: valid_labels,
        keep_prob: 1.
    })

    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_accuracy))


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
   

    session.run(optimizer, feed_dict={  x: feature_batch,  y: label_batch,  keep_prob: keep_probability
    })

    








def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    #Normalization equation
    #zi=xi−min(x)/max(x)−min(x)
    normalized = (x-np.min(x))/(np.max(x)-np.min(x))
    return normalized

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    return np.eye(10)[x]







class DLProgress(tqdm):
    last_block = 0
    textBrowse = None
    mainWindow = None
          
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.textBrowse.setText( "{} of {} Downloaded".format( self   , total_size  ))
        self.mainWindow.centralwidget.repaint()
        time.sleep(0.6)
        self.last_block = block_num
    
    


class Model:
    
    
    def __init__( self ):
        '''
        Initializes the two members the class holds:
        the file name and its contents.
        '''
        self.fileName = None
        self.fileContent = ""
        self.cifar10_dataset_folder_path = 'cifar-10-batches-py'
        self.floyd_cifar10_location = 'cifar-10-python.tar.gz'        
        self.tar_gz_path=""
        self.url= 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.batch_id =1
        self.sample_id = 5
        self.testOutput = ""
        self.epochs = 1
        self.batch_size = 256
        self.keep_probability = 0.6
        
        
        
    def get_cifar10_dataset_folder_path(self ):
        return self.cifar10_dataset_folder_path
    
    def get_floyd_cifar10_location(self):
        return self.floyd_cifar10_location
    
    def get_url(self):
        return self.url
    
    def extractTar(self, debugTextBrowse):
        if isfile(self.floyd_cifar10_location):
            self.tar_gz_path = self.floyd_cifar10_location
        else:
            debugTextBrowse.setText("Invalid Tar File")
            return False

        if not isdir(self.cifar10_dataset_folder_path):
            with tarfile.open(self.tar_gz_path) as tar:
                tar.extractall()
                tar.close()
                debugTextBrowse.setText("Successfully Extracted")
                return True
        else:
            debugTextBrowse.setText("Invalid CIFAR Folder, please identify the folder above")
            return False
                
                
    def downloadTar(self, debugTextBrowse, mainWindow):
        if not isfile(self.tar_gz_path):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
                pbar.textBrowse = debugTextBrowse
                pbar.mainWindow = mainWindow
                urlretrieve( self.url,
                        self.tar_gz_path,
                        pbar.hook)
        self.extractTar(debugTextBrowse)
                
    
        
        
    def runCifarTest(self)    :
        
        

        if isfile(self.floyd_cifar10_location):
            self.tar_gz_path = self.floyd_cifar10_location
        else:
            self.tar_gz_path = 'cifar-10-python.tar.gz'

        if not isfile(self.tar_gz_path):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
                urlretrieve( self.url,
                        self.tar_gz_path,
                        pbar.hook)

        if not isdir(self.cifar10_dataset_folder_path):
            with tarfile.open(self.tar_gz_path) as tar:
                tar.extractall()
                tar.close()
       
        tests.test_folder_path(self.cifar10_dataset_folder_path, self)
        
    def displayStats(self, debugTextBrowser, batchId, sampleId,m ):
        
        helper.display_stats( self.cifar10_dataset_folder_path,self, debugTextBrowser, batchId, sampleId, m )
        
        
        
        
    
    def runDataTests(self):
        self.runCifarTest();
        print(" Model DataTests Ran Successfully")
         
    def runImageInputTests(self):
        tests.test_nn_image_inputs(neural_net_image_input)
        print("Model ImageInputTests Ran Successfully")
    def runKeepProbTests(self):
        tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)
        print("Model KeepProbTests Ran Successfully")
    def runLabelInputTests(self):
        tests.test_nn_label_inputs(neural_net_label_input)
        print("Model LabelInputTests Ran Successfully")
    def runNormalisationTests(self):
        tests.test_normalize(normalize)
        print("Model NormalisationTests Ran Successfully")
    def runOneHotEncodeTests(self):
        tests.test_one_hot_encode(one_hot_encode)
        print("Model OneHotEncodeTests Ran Successfully")
    def runFullyConvLayerTests(self):
        tests.test_fully_conn(fully_conn)
        print("Model FullyConvLayerTests Ran Successfully")
    def runConvMaxLayerTest(self):
        tests.test_con_pool(conv2d_maxpool)
        print("Model ConvMaxLayerTest Ran Successfully")

    
#    get_ipython().run_line_magic('matplotlib', 'inline')
#    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")



    def runOutputLayerTests(self):
        tests.test_output(output)
        ##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
     



        tests.test_conv_net(conv_net)
        print("Model OutputLayerTests Ran Successfully")
        

        
    def runFlattenLayerTests(self):
        tests.test_flatten(flatten)
        print("Model FlattenLayerTests Ran Successfully")
        
        
    def runCNNNetworkTests(self):
        print("Model CNNNetworkTests Ran Successfully")
    def runTrainingTests(self):
        




      


        tests.test_train_nn(train_neural_network)
        print("Model TrainingTests Ran Successfully")
    
    def runPreProcessAndSave(self):
        helper.preprocess_and_save_data(self.cifar10_dataset_folder_path, normalize, one_hot_encode)
        valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
        
        print("Model PreProcessAndSave Ran Successfully")
    def runShowStats(self):
        print("Model ShowStats Ran Successfully")
    def runTrainOnSingleBatch(self):
        
        print('Checking the Training on a Single Batch...')
        with tf.Session() as sess:
    # Initializing the variables
            sess.run(tf.global_variables_initializer())
        
    # Training cycle
            for epoch in range(epochs):
                batch_i = 1
                for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                    train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                print_stats(sess, batch_features, batch_labels, cost, accuracy)
        print("Model TrainOnSingleBatch Ran Successfully")
        
    def runFullyTrainModel(self, epochs, batch_size, keep_probability):
        
        save_model_path = './image_classification'

        print('Training...')
        with tf.Session() as sess:
    # Initializing the variables
            sess.run(tf.global_variables_initializer())
    
    # Training cycle
            for epoch in range(epochs):
        # Loop over all batches
                n_batches = 5
                for batch_i in range(1, n_batches + 1):
                    for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                        train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                    print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                    print_stats(sess, batch_features, batch_labels, cost, accuracy)
            saver = tf.train.Saver()
            save_path = saver.save(sess, save_model_path)
        print("Model FullyTrainModel Ran Successfully")
       
        
        
        
        
        
    def runClassificationOnTestData(self, plotCanvas):
        test_model(plotCanvas)
        print("Model Classification on Test Data Completed Successfully")
    

    def isValid( self, fileName ):
        '''
        returns True if the file exists and can be
        opened.  Returns False otherwise.
        '''
        try: 
            file = open( fileName, 'r' )
            file.close()
            return True
        except:
            return False
        
    def isValidFolder( self, folderName ):
        '''
        returns True if the file exists and can be
        opened.  Returns False otherwise.
        '''
        if isdir(folderName):
            return True
        else:
            return False
        
        
        
    def isValidURL( self, URLName ):
        '''
        returns True if the URL exists and can be
        opened.  Returns False otherwise.
        '''
        
        val = URLValidator()
        try: 
            val( URLName )
            return True
        except ValidationError:
            return False
        
        
    def setFolderName( self, folderName ):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''
        if self.isValid( folderName ):
            self.cifar10_dataset_folder_path = folderName
        else:
            self.fileContents = ""
            self.fileName = ""
            
    def setTarName( self, tarName ):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''
        if self.isValid( tarName ):
            self.floyd_cifar10_location = tarName
        else:
            self.fileContents = ""
            self.fileName = ""

    def setFileName( self, fileName ):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''
        if self.isValid( fileName ):
            self.fileName = fileName
            self.fileContents = open( fileName, 'r' ).read()
        else:
            self.fileContents = ""
            self.fileName = ""
            
    def getFileName( self ):
        '''
        Returns the name of the file name member.
        '''
        return self.fileName

    def getFileContents( self ):
        '''
        Returns the contents of the file if it exists, otherwise
        returns an empty string.
        '''
        return self.fileContents
    
    def writeDoc( self, text ):
        '''
        Writes the string that is passed as argument to a
        a text file with name equal to the name of the file
        that was read, plus the suffix ".bak"
        '''
        if self.isValid( self.fileName ):
            fileName = self.fileName + ".bak"
            file = open( fileName, 'w' )
            file.write( text )
            file.close()


    def downloadCIFARData(self ):
        '''
        download the [CIFAR-10 dataset for python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).
        '''
        
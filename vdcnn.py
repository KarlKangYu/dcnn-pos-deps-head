import tensorflow as tf
import numpy as np
import math

# weights initializers
#he_normal = tf.contrib.keras.initializers.he_normal()
he_normal = tf.contrib.layers.variance_scaling_initializer()
regularizer = tf.contrib.layers.l2_regularizer(1e-4)

def Convolutional_Block(inputs, shortcut, num_filters, name, is_training):
    print("-"*20)
    print("Convolutional Block", str(num_filters), name)
    print("-"*20)
    with tf.variable_scope("conv_block_" + str(num_filters) + "_" + name):
        for i in range(2):
            with tf.variable_scope("conv1d_%s" % str(i)):
                filter_shape = [3, inputs.get_shape()[2], num_filters]
                W = tf.get_variable(name='W', shape=filter_shape, 
                    initializer=he_normal,
                    regularizer=regularizer)
                inputs = tf.nn.conv1d(inputs, W, stride=1, padding="SAME")
                inputs = tf.layers.batch_normalization(inputs=inputs, momentum=0.997, epsilon=1e-5, 
                                                center=True, scale=True, training=is_training)
                inputs = tf.nn.relu(inputs)
                print("Conv1D:", inputs.get_shape())
    print("-"*20)
    if shortcut is not None:
        print("-"*5)
        print("Optional Shortcut:", shortcut.get_shape())
        print("-"*5)
        return inputs + shortcut
    return inputs

# Three types of downsampling methods described by paper
def downsampling(inputs, downsampling_type, name, optional_shortcut=False, shortcut=None):
    # k-maxpooling
    if downsampling_type=='k-maxpool':
        k = math.ceil(int(inputs.get_shape()[1]) / 2)
        pool = tf.nn.top_k(tf.transpose(inputs, [0,2,1]), k=k, name=name, sorted=False)[0]
        pool = tf.transpose(pool, [0,2,1])
    # Linear
    elif downsampling_type=='linear':
        pool = tf.layers.conv1d(inputs=inputs, filters=inputs.get_shape()[2], kernel_size=3,
                            strides=2, padding='same', use_bias=False)
    # Maxpooling
    else:
        pool = tf.layers.max_pooling1d(inputs=inputs, pool_size=3, strides=2, padding='same', name=name)
    if optional_shortcut:
        shortcut = tf.layers.conv1d(inputs=shortcut, filters=shortcut.get_shape()[2], kernel_size=1,
                            strides=2, padding='same', use_bias=False)
        print("-"*5)
        print("Optional Downsampling Shortcut:", shortcut.get_shape())
        print("-"*5)
        pool += shortcut
    pool = fixed_padding(inputs=pool)
    return tf.layers.conv1d(inputs=pool, filters=pool.get_shape()[2]*2, kernel_size=1,
                            strides=1, padding='valid', use_bias=False)

def fixed_padding(inputs, kernel_size=3):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [0, 0]])
    return padded_inputs

class VDCNN():
    def __init__(self, num_classes, sequence_max_length=20, num_quantized_chars=50000, tags_vocab_size=44,
                 deps_vocab_size=47, embedding_size=300,
                 depth=9, downsampling_type='maxpool', use_he_uniform=True, optional_shortcut=False):

        # Depth to No. Layers
        if depth == 9:
            num_layers = [2,2,2,2]
        elif depth == 17:
            num_layers = [4,4,4,4]
        elif depth == 29:
            num_layers = [10,10,4,4]
        elif depth == 49:
            num_layers = [16,16,10,6]
        else:
            raise ValueError('depth=%g is a not a valid setting!' % depth)

        # input tensors
        self.input_x = tf.placeholder(tf.int32, [None, sequence_max_length], name="input_x")
        self.input_tags = tf.placeholder(tf.int32, [None, sequence_max_length], name="input_tags")
        self.input_deps = tf.placeholder(tf.int32, [None, sequence_max_length], name="input_dependency")
        self.input_head = tf.placeholder(tf.int32, [None, sequence_max_length], name="input_head")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.is_training =  tf.placeholder(tf.bool)

        initializer = tf.contrib.layers.variance_scaling_initializer()

        # Embedding Lookup 16
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if use_he_uniform:
                self.embedding_W = tf.get_variable(name='lookup_W', shape=[num_quantized_chars, embedding_size],
                                                   initializer=tf.contrib.layers.variance_scaling_initializer())
            else:
                self.embedding_W = tf.Variable(tf.random_uniform([num_quantized_chars, embedding_size], -1.0, 1.0),name="embedding_W")
            self.embedded_characters = tf.nn.embedding_lookup(self.embedding_W, self.input_x)
            embedded_text_expand = tf.expand_dims(self.embedded_characters, -1)

        with tf.device('/cpu:0'), tf.name_scope("embedding_tags"):
            W_tags = tf.get_variable("embed_W_tags", [tags_vocab_size, embedding_size], initializer=initializer)
            embedded_tags = tf.nn.embedding_lookup(W_tags, self.input_tags)
            embedded_tags_expanded = tf.expand_dims(embedded_tags, -1)

        with tf.device('/cpu:0'), tf.name_scope("embedding_deps"):
            W_deps = tf.get_variable("embed_W_deps", [deps_vocab_size, embedding_size], initializer=initializer)
            embedded_deps = tf.nn.embedding_lookup(W_deps, self.input_deps)
            embedded_deps_expanded = tf.expand_dims(embedded_deps, -1)

        with tf.device('/cpu:0'), tf.name_scope("embedding_head"):
            W_head = tf.get_variable("embed_W_head", [num_quantized_chars, embedding_size], initializer=initializer)
            embedded_head = tf.nn.embedding_lookup(W_head, self.input_head)
            embedded_head_expanded = tf.expand_dims(embedded_head, -1)

        cnn_inputs = tf.concat(
            [embedded_text_expand, embedded_tags_expanded, embedded_deps_expanded, embedded_head_expanded], -1)

        print("-" * 20)
        print("Embedded Lookup:", cnn_inputs.get_shape())
        print("-" * 20)

        self.layers = []

        # Temp(First) Conv Layer
        with tf.variable_scope("temp_conv") as scope: 
            filter_shape = [3, embedding_size, 4, 64]
            W = tf.get_variable(name='W_1', shape=filter_shape, 
                initializer=he_normal,
                regularizer=regularizer)
            paddings = [[0, 0], [1, 1], [0, 0], [0, 0]]
            cnn_inputs = tf.pad(cnn_inputs, paddings, "CONSTANT")
            #print("cnn_inputs shape:", cnn_inputs.shape)
            inputs = tf.nn.conv2d(cnn_inputs, W, strides=[1, 1, 1, 1], padding="VALID", name="first_conv")
            #print("temp cnn output shape:", inputs.shape)
            inputs = tf.squeeze(inputs, axis=2)
            #print("squeeze shape", inputs.shape)
            #inputs = tf.nn.relu(inputs)
        print("Temp Conv", inputs.get_shape())
        self.layers.append(inputs)

        # Conv Block 64
        for i in range(num_layers[0]):
            if i < num_layers[0] - 1 and optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=64, is_training=self.is_training, name=str(i+1))
            self.layers.append(conv_block)
        pool1 = downsampling(self.layers[-1], downsampling_type=downsampling_type, name='pool1', optional_shortcut=optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool1)
        print("Pooling:", pool1.get_shape())

        # Conv Block 128
        for i in range(num_layers[1]):
            if i < num_layers[1] - 1 and optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=128, is_training=self.is_training, name=str(i+1))
            self.layers.append(conv_block)
        pool2 = downsampling(self.layers[-1], downsampling_type=downsampling_type, name='pool2', optional_shortcut=optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool2)
        print("Pooling:", pool2.get_shape())

        # Conv Block 256
        for i in range(num_layers[2]):
            if i < num_layers[2] - 1 and optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=256, is_training=self.is_training, name=str(i+1))
            self.layers.append(conv_block)
        pool3 = downsampling(self.layers[-1], downsampling_type=downsampling_type, name='pool3', optional_shortcut=optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool3)
        print("Pooling:", pool3.get_shape())

        # Conv Block 512
        for i in range(num_layers[3]):
            if i < num_layers[3] - 1 and optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=512, is_training=self.is_training, name=str(i+1))
            self.layers.append(conv_block)

        # Extract 8 most features as mentioned in paper
        self.k_pooled = tf.nn.top_k(tf.transpose(self.layers[-1], [0,2,1]), k=8, name='k_pool', sorted=False)[0]
        print("8-maxpooling:", self.k_pooled.get_shape())
        self.flatten = tf.reshape(self.k_pooled, (-1, 512*8))

        # fc1
        with tf.variable_scope('fc1'):
            w = tf.get_variable('w', [self.flatten.get_shape()[1], 2048], initializer=he_normal,
                regularizer=regularizer)
            b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(1.0))
            out = tf.matmul(self.flatten, w) + b
            self.fc1 = tf.nn.relu(out)

        # fc2
        with tf.variable_scope('fc2'):
            w = tf.get_variable('w', [self.fc1.get_shape()[1], 2048], initializer=he_normal,
                regularizer=regularizer)
            b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(1.0))
            out = tf.matmul(self.fc1, w) + b
            self.fc2 = tf.nn.relu(out)

        # fc3
        with tf.variable_scope('fc3'):
            w = tf.get_variable('w', [self.fc2.get_shape()[1], num_classes], initializer=he_normal,
                regularizer=regularizer)
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(1.0))
            self.fc3 = tf.matmul(self.fc2, w) + b

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            self.predictions = tf.argmax(self.fc3, 1, name="predictions")
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc3, labels=self.input_y)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(losses) + sum(regularization_losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

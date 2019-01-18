import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import driving_data
import model
import time

def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.float32) / 255
    annotation = tf.decode_raw(features['mask_raw'], tf.float32)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = tf.stack([tf.shape(image)[0], 66, 200, 3])
    annotation_shape = tf.stack([tf.shape(image)[0], 1])
    
    print([tf.size(image), height, width, 3])
    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
    
    return image, annotation

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    with tf.name_scope('gradient_averaging'):
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
              # Add 0 dimension to the gradients to represent the tower.
              expanded_g = tf.expand_dims(g, 0)
           
              # Append on a 'tower' dimension which we will average over below.
              grads.append(expanded_g)
           
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
           
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


epochs = 30
batch_size = 100
global_step = driving_data.num_images
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")

start = time.time()
LOGDIR = './save'

sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
#sess.run(tf.initialize_all_variables())

L2NormConst = 0.001

opt = tf.train.AdamOptimizer(1e-4)

filename_queue = tf.train.string_input_producer(
    ["driving_dataset/train_data.tfrecords"])
Xs, Ys = read_and_decode(filename_queue)
print(Xs.shape, Ys.shape)
dataset = tf.data.Dataset.from_tensor_slices((Xs, Ys))
'''
Xs_batch, Ys_batch = tf.train.batch(
    [Xs, Ys],
    batch_size=batch_size,
    num_threads=16)
    #capacity=0.4*driving_data.num_images+3*batch_size)
Ys_batch_reshaped = tf.reshape(Ys_batch, [batch_size])
'''


tower_grads = []
for i in range(FLAGS.num_gpus):
    print("{:d}th device".format(i))
    with tf.variable_scope("cnn"):
        with tf.name_scope("train_{:d}".format(i)) as scope:
            with tf.device("/gpu:{:d}".format(i)):
                shared_dataset = dataset.shard(FLAGS.num_gpus, i)
                shared_dataset_batch = shared_dataset.batch(batch_size)
                shared_dataset_batch.repeat(epochs)
                iterator = shared_dataset_batch.make_initializable_iterator()
                x_batch, y_batch = iterator.get_next()
                '''
                train_vars = tf.trainable_variables()
                variables_names = [v.name for v in tf.trainable_variables()]
                print("there are {:d} variables".format(len(variables_names)))
                values = sess.run(variables_names)
                for k, v in zip(variables_names, values):
                    print( "Variable: ", k)
                    print( "Shape: ", v.shape)
                    print(v)
                '''
                loss = tf.reduce_mean(tf.square(tf.subtract(y_batch, model.get_model(x_batch)))) \
                        + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * L2NormConst
                tf.get_variable_scope().reuse_variables()
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                grads = opt.compute_gradients(loss)
                tower_grads.append(grads)

grads = average_gradients(tower_grads)
apply_gradient_op = opt.apply_gradients(grads)

'''
variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
'''

train_op = tf.group(apply_gradient_op, variables_averages_op)

for grad, var in grads:
    if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name, var))

for var in tf.trainable_variables():
    summaries.append(tf.summary.histogram(var.op.name, var))

summary_op = tf.summary.merge(summaries)

print("before run")
sess.run(tf.initialize_all_variables())
sess.run(iterator.initializer)
print("after run")

# create a summary to monitor cost tensor
#tf.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op =  tf.summary.merge_all()

saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


# train over the dataset about 30 times
for epoch in range(epochs):
  for i in range(int(driving_data.num_images/batch_size)):
    #xs, ys = driving_data.LoadTrainBatch(batch_size)
    #train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
    _, loss_value = sess.run([train_op, loss])

    if i % 10 == 0:
      # xs, ys = driving_data.LoadValBatch(batch_size)
      # loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)

    if i % batch_size == 0:
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename)

print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
print("It takes {:f} seconds to train".format(time.time() - start))

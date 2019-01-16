import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import driving_data
import model
import time

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

sess = tf.InteractiveSession()

L2NormConst = 0.001

opt = tf.train.AdamOptimizer(1e-4)

Xs, Ys = driving_data.LoadTrainBatch(int(driving_data.num_images))
Xs = tf.convert_to_tensor(Xs)
Ys = tf.convert_to_tensor(Ys)
Xs_batch, Ys_batch = tf.train.batch(
    [Xs, Ys],
    batch_size=batch_size,
    num_threads=16,
    capacity=0.4*driving_data.num_images+3*batch_size)
Ys_batch_reshaped = tf.reshape(Ys_batch, [batch_size])

batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        [Xs_batch, Ys_batch_reshaped], capacity=2*FLAGS.num_gpus)

tower_grads = []
with tf.variable_scope(tf.get_variable_scope()):
    for i in xrange(FLAGS.num_gpus):
        with tf.device("/gpu:{:d}".format(i)):
            with tf.name_scope("train_{:d}".format(i)) as scope:
                x_batch, y_batch = batch_queue.dequeue()
                train_vars = tf.trainable_variables()
                loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
                tf.get_variable_scope().reuse_variables()
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                grads = opt.compute_cradients(loss)
                tower_grads.append(grads)

grads = average_gradients(tower_grads)
apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables)

train_op = tf.group(apply_gradient_op, variables_averages_op)

for grad, var in grads:
    if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name, var))

for var in tf.trainable_variables():
    summaries.append(tf.summary.histogram(var.op.name, var))

summary_op = tf.summary.merge(summaries)

sess.run(tf.initialize_all_variables())

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
    xs, ys = driving_data.LoadTrainBatch(batch_size)
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

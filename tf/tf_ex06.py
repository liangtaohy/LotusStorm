import tensorflow as tf

## Save to file

W = tf.Variable([[1,2,3],[4,5,6]], dtype=tf.float32, name="weights")
b = tf.Variable([[1,2,3]], dtype=tf.float32, name="bias")

init = tf.global_variables_initializer()

saver = tf.train.Saver()

sess = tf.Session()
sess.run(init)
save_path = saver.save(sess, 'logs/save_net.pt')
print("Save to path: " + save_path)



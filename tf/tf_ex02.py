#
# 常量、变量、placeholder
#
import tensorflow as tf

# 常量矩阵定义
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)

# method 1
sess = tf.Session()
result1 = sess.run(product)
print(result1)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

# 变量定义

state = tf.Variable(0, name='counter')
print(state.name)

one = tf.constant(1)

# state += one
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
    sess.close()

# placeholder

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7., 2.], input2: [2., 3.]}))
    sess.close()

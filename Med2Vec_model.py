from tqdm import trange
import tensorflow as tf

class Med2VecModel():
    def __init__(self, flag):
        self.flag = flag
        self.Building_model()
        
    def Building_model(self):
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.flag.code_size], name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=[None, self.flag.code_size], name='labels')
        self.visit_times = tf.placeholder(tf.float32, shape=[], name='visit_times')
        self.cooccur_idx_i = tf.placeholder(tf.int32, shape=[None], name='cooccur_idx_i')
        self.cooccur_idx_j = tf.placeholder(tf.int32, shape=[None], name='cooccur_idx_j')
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.lr = tf.train.exponential_decay(self.flag.lr_init, self.global_step, 
                                             self.flag.decay_step, self.flag.decay_rate, staircase=True)
        

        ##Learning from the code-level info
        W_c = tf.Variable(tf.random_uniform(shape=[self.flag.code_size, self.flag.u_size], minval=-0.01, maxval=0.01), name='W_c')
        b_c = tf.Variable(0.001, name='b_c')
        u_t = tf.nn.relu(tf.matmul(self.inputs, W_c) + b_c)
        print('u_t: ', u_t)
        #W'_c
        emb_W_c = tf.nn.relu(W_c)
        W_i = tf.nn.embedding_lookup(emb_W_c, self.cooccur_idx_i)
        W_j = tf.nn.embedding_lookup(emb_W_c, self.cooccur_idx_j)
        print('emb_W_c: ', emb_W_c)
        print('W_i: ', W_i)
        print('W_j: ', W_j)
        
        ##Learning from the visit-level info
        W_v = tf.Variable(tf.random_normal(shape=[self.flag.u_size, self.flag.v_size]), name='W_v')
        b_v = tf.Variable(tf.random_normal(shape=[self.flag.v_size]), name='b_v')
        v_t = tf.nn.relu(tf.matmul(u_t, W_v) + b_v)
        print('v_t: ', v_t)

        ##softmax
        W_s = tf.Variable(tf.random_normal(shape=[self.flag.v_size, self.flag.code_size]), name='W_s')
        b_s = tf.Variable(tf.random_normal(shape=[self.flag.code_size]), name='b_s')
        logits = tf.nn.softmax(tf.matmul(v_t, W_s) + b_s)
        print('logits: ', logits)
        
        ##cross_entropy
        #visit-level
        visit_cross_entropy_vec = tf.reduce_sum(-self.labels*tf.log(logits)-(1-self.labels)*tf.log(1-logits), axis=1)
        print('visit_cross_entropy_vec', visit_cross_entropy_vec)
        print('self.visit_times', self.visit_times)
        self.visit_cross_entropy = 1/self.visit_times*tf.reduce_sum(visit_cross_entropy_vec, axis=0)
        #code-level
        code_cross_entropy_vec = tf.log(tf.exp(tf.reduce_sum(W_j*W_i, axis=1))/tf.exp(tf.reduce_sum(tf.matmul(emb_W_c, tf.transpose(W_i)), axis=0)))
        code_cross_entropy = -1/self.visit_times*tf.reduce_sum(code_cross_entropy_vec)
        
        self.cross_entropy = self.visit_cross_entropy + code_cross_entropy 
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cross_entropy, global_step=self.global_step)
        self.optimizer_visit = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.visit_cross_entropy, global_step=self.global_step)
        
def Train_model(datasets, model):
    from tqdm import trange
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess)
        tf.global_variables_initializer().run(session=sess)

        for step in trange(model.flag.training_step):
            batch_inputs, batch_labels, batch_visit_times, batch_cooccur_idx_i, batch_cooccur_idx_j = datasets.next_batch()
            feed_dict = {model.inputs: batch_inputs, 
                         model.labels: batch_labels, 
                         model.visit_times: batch_visit_times, 
                         model.cooccur_idx_i: batch_cooccur_idx_i, 
                         model.cooccur_idx_j: batch_cooccur_idx_j}
            if len(batch_cooccur_idx_i)*len(batch_cooccur_idx_j):
                _, g_step, lr_, loss = sess.run([model.optimizer, model.global_step, model.lr, model.cross_entropy], 
                                                feed_dict=feed_dict)
            else:
                visit_, visit_g_step, visit_lr_, visit_loss = sess.run([model.optimizer_visit, model.global_step, model.lr, 
                                                                        model.visit_cross_entropy], feed_dict=feed_dict)
                print('step: {} \t\t\t\t\t\t\t||visit_loss: {:.8f}'.format(step, visit_loss)) 
                
            if step % model.flag.printby == 0:
                print('step: {} \t||loss: {:.8f} \t||lr: {:.8f}'.format(step, loss,  lr_)) 
            
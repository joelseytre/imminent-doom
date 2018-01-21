from sklearn.manifold import TSNE
import numpy as np
from vizdoom import *
import tensorflow as tf
import cv2
import pickle
import itertools as it
from sklearn.cluster import KMeans
from tqdm import trange
import matplotlib.pyplot as plt
import time
from tensorflow.core.protobuf import saver_pb2

# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 20
learning_steps_per_epoch = 20000
replay_memory_size = 2000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 200

model_savefile = "curiosity_models/doomSparse_ICM"
config_file_path =  "/home/arnaud/ViZDoom/scenarios/my_way_home.cfg"

frame_repeat = 3
resolution = (50, 50)
initial_resolution = (256, 192)

make_memory = True

clustering_folder = "clustering/way_home/tmp/"


class Coherent_KMeans:
    def __init__(self, cluster_centers, w):
        self.cluster_centers_ = cluster_centers
        self.n_clusters_ = len(self.cluster_centers_)
        self.w = w

    def predict(self, traj):
        traj = np.array(traj)
        clustering = np.zeros(len(traj))
        for t in range(len(traj)):
            extended_state = traj[int(max(t-self.w,0)):int(min(t+self.w+1,len(traj)))]
            clustering[t] = np.argmin([np.linalg.norm(extended_state-center) for center in self.cluster_centers_])
        return clustering

    def transform(self, traj):
        traj = np.array(traj)
        distances = np.zeros((len(traj), self.n_clusters_))
        for t in range(len(traj)):
            extended_state = traj[int(max(t-self.w,0)):int(min(t+self.w+1, len(traj)))]
            distances[t] = [np.linalg.norm(extended_state-center) for center in self.cluster_centers_]
        return distances

def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_256X192)
    game.set_episode_timeout(1000)
    game.init()
    print("Doom initialized.")
    return game

def preprocess(img):
    img = cv2.resize(img, resolution)
    img = img.astype(np.float32)
    return img.T

class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        blob = (capacity, initial_resolution[1], initial_resolution[0], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.s1_hd = np.zeros(blob, dtype=np.float32)
        self.s2_hd = np.zeros(blob, dtype=np.float32)

        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)
        self.s1_sne = np.zeros((capacity, 3), dtype=np.int32)
        self.s2_sne = np.zeros((capacity, 3), dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.sne_map=None

    def add_transition(self, s1, action, s2, isterminal, reward, s1_hd):
        self.s1[self.pos, :, :, 0] = s1
        self.s1_hd[self.pos, :, :, 0] = s1_hd
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

    def translate_to_tsne(self, fn_hidden_layer, fn_q):
        print("Translating memory to (tsne, Q) coordinates")
        sne_map = TSNE(n_components=2)
        # X = [fn_hidden_layer(state) for state in self.s1]
        X = fn_hidden_layer(self.s1)
        X_sne = sne_map.fit_transform(X)
        self.sne_map = sne_map

        self.s1_sne[:, :2] = X_sne

        self.s1_sne[:, 2] = np.max(fn_q(self.s1), axis=1)
        # self.s1_sne[:, 2] = [np.max(fn_q(state)) for state in self.s1]

        # Careful, TSNE is not generalizable
        self.s2_sne[:, :2] = sne_map.fit_transform(fn_hidden_layer(self.s2))
        self.s2_sne[:, 2] = np.max(fn_q(self.s2), axis=1)
        # self.s2_sne[:, :2] = sne_map.transform([fn_hidden_layer(state) for state in self.s2])
        # self.s2_sne[:, 2] = [np.max(fn_q(state)) for state in self.s1]

        return self.s1_sne, self.s2_sne


def create_network(session, available_actions_count):
    # Create the input variables
    s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [1], name="State")
    a_ = tf.placeholder(tf.int32, [None], name="Action")
    target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

    # Add 2 convolutional layers with ReLu activation
    conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2_flat = tf.contrib.layers.flatten(conv2)
    fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))

    q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))
    best_a = tf.argmax(q, 1)

    loss = tf.losses.mean_squared_error(q, target_q_)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # Update the parameters according to the computed gradient using RMSProp.
    train_step = optimizer.minimize(loss)

    def function_learn(s1, target_q):
        feed_dict = {s1_: s1, target_q_: target_q}
        l, _ = session.run([loss, train_step], feed_dict=feed_dict)
        return l

    def function_get_q_values(state):
        return session.run(q, feed_dict={s1_: state})

    def function_get_last_hidden_layer(state):
        return session.run(fc1, feed_dict={s1_: state})

    def function_get_best_action(state):
        return session.run(best_a, feed_dict={s1_: state})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))[0]

    return function_learn, function_get_q_values, function_simple_get_best_action, function_get_last_hidden_layer


def build_memory(replay_memory_size=20000):
    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # session = tf.Session()
    # learn, get_q_values, get_best_action, get_last_hidden_layer = create_network(session, len(actions))
    # saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

    session = tf.Session()
    print("Loading model from: ", model_savefile)
    saver = tf.train.import_meta_graph(model_savefile + '.meta')
    saver.restore(session, model_savefile)

    def get_last_hidden_layer(state):
        return session2.run('l2', feed_dict={'x': state})

    get_last_hidden_layer(game.get_state().screen_buffer)

    writer = tf.summary.FileWriter("tmp/log/...", session2.graph)

    session2.run(tf.global_variables_initializer())
    writer.flush()

    for op in tf.global_variables():
        print(str(op.name))

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)
    game.init()

    # Fill the memory
    memory_full = False
    for _ in trange(replay_memory_size):
        game.new_episode()
        if memory_full:
            break
        while not game.is_episode_finished():
            s1_hd = game.get_state().screen_buffer
            s1 = preprocess(s1_hd)
            # plt.imshow(s1)
            # plt.show()
            a = get_best_action(s1)
            reward = game.make_action(actions[a], frame_repeat)

            isterminal = game.is_episode_finished()
            s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

            if memory.pos == memory.capacity:
                memory_full = True
                break

            memory.add_transition(s1, a, s2, isterminal, reward, s1_hd)

            game.set_action(actions[a])
            for _ in range(frame_repeat):
                game.advance_action()

    _, _ = memory.translate_to_tsne(get_q_values, get_last_hidden_layer)
    pickle.dump(memory, open(clustering_folder + 'memory.pkl', 'wb'))

def aggregate_states(sne_trajs, terminations, K=3, w=2, n_it=20):
        # K : number of clusters
        # w : size of the sliding window


        ##### Initialize clustering with KMeans ######
        N = sne_trajs.shape[0]
        initializer = KMeans(n_clusters=K)
        clustering = initializer.fit_predict(sne_trajs)
        cluster_centers = initializer.cluster_centers_

        n_trajs = np.sum(terminations)
        plop = np.concatenate(([0], np.where(terminations == 1)[0], [N]))
        print(plop)
        traj_limits = [(plop[i], plop[i+1] + 1) for i in range(len(plop)-1)]
        print(traj_limits)

        for iteration in trange(n_it, leave=False):
            # update cluster_centers
            for k in range(K):
                states_k = sne_trajs[clustering==k]
                if len(states_k)==0: # no state is assigned to cluster k -> don't change it
                    continue
                cluster_centers[k] = states_k.sum(axis=0) / float(len(states_k))

            # update clustering

            for start, end in traj_limits:
                traj = sne_trajs[start:end]

                for t in range(len(traj)):
                    extended_state = traj[int(max(t-w,0)):int(min(t+w+1,len(traj)))]
                    clustering[start + t] = np.argmin([np.linalg.norm(extended_state-cluster_centers[k]) for k in range(K)])

        return Coherent_KMeans(cluster_centers, w), clustering


def infer_transitions(clusterer, sne_trajs, terminations):
    n_clusters = clusterer.n_clusters_
    agent_transitions = np.zeros((n_clusters, n_clusters))
    N = sne_trajs.shape[0]

    n_trajs = np.sum(terminations)
    plop = np.concatenate(([0], np.where(terminations == 1), [N]))
    print(plop)
    traj_limits = [(plop[i], plop[i+1] + 1) for i in range(len(plop)-1)]
    print(traj_limits)

    for start, end in traj_limits:
        traj = sne_trajs[start:end-1]
        next_traj = sne_trajs[start+1:end]
        clusters = clusterer.predict(traj)
        next_clusters = clusterer.predict(next_traj)

        for idx, cluster, next_cluster in enumerate(zip(clusters, next_clusters)):
            if cluster != next_cluster or idx == len(clusters):
                agent_transitions[cluster,next_cluster] += 1

    # Normalization
    for cluster in range(n_clusters):
        agent_sample_nb = agent_transitions[cluster,:].sum()
        if agent_sample_nb:
            agent_transitions[cluster,:] /= float(agent_sample_nb)

    return agent_transitions


def get_best_representatives(cluster_centers, memory, k=5):
    n_clusters = cluster_centers.shape[0]
    representatives = np.zeros((n_clusters, initial_resolution[1], initial_resolution[0]))
    for c in range(n_clusters):
        ind = np.argpartition([np.linalg.norm(state-cluster_centers[c]) for state
                                in memory.s1_sne], k)[:k]
        # best_idx = np.argmin()

        tmp = [memory.s1_hd[i][:, :, 0] for i in ind]
        representatives[c, :, :] = np.mean(tmp, axis=0)
        plt.imshow(representatives[c, :, :])
        plt.show()
    return representatives

if __name__ == '__main__':
    if make_memory:
        print('Building memory')
        build_memory(replay_memory_size)

    memory = pickle.load(open(clustering_folder + 'memory.pkl', 'rb'))

    sne_trajs = memory.s1_sne

    terminations = memory.isterminal
    Clusterer, clustering = aggregate_states(sne_trajs, terminations,
                                K=3, w=0, n_it=100)
    pickle.dump(Clusterer, open(clustering_folder + 'clusterer.pkl', 'wb'))
    best_frames = get_best_representatives(Clusterer.cluster_centers_, memory)
    np.save(clustering_folder + 'representatives', best_frames)

    transitions = infer_transitions(Clusterer, sne_trajs, terminations)
    np.save(transitions, clustering_folder + 'transitions')

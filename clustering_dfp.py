clustering_folder = "clustering/tmp/"


memory = MultiExperiment.test_policy_memory

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
        next_traj = sne_trajs[start+1:end-]
        clusters = clusterer.predict(traj)
        next_clusters = clusterer.predict(next_traj)

        for idx, cluster, next_cluster in enumerate(zip(clusters, next_clusters)):
            if cluster != next_cluster or idx = len(clusters):
                agent_transitions[cluster,next_cluster] += 1

    # Normalization
    for cluster in range(n_clusters):
        agent_sample_nb = agent_transitions[cluster,:].sum()
        if agent_sample_nb:
            agent_transitions[cluster,:] /= float(agent_sample_nb)

    return agent_transitions


def get_best_representatives(cluster_centers, memory):
    n_clusters = cluster_centers.shape[0]
    representatives = np.zeros(n_clusters, resolution[0], resolution[1])
    for c in range(n_clusters):
        best_idx = np.argmin([np.norm(state-cluster_centers[c]) for state
                                in memory.s1_sne])
        representatives[c, :, :] = memory.s1[best_index]
    return representatives

if __name__ == '__main__':

    if build_memory:
        build_memory(20000)

    memory = pickle.load(open(clustering_folder + 'memory.pkl', 'rb'))
    sne_trajs = memory.s1_sne
    terminations = memory.isterminal
    Clusterer, clustering = aggregate_states(sne_trajs, terminations,
                                K=9, w=5, n_it=20)
    pickle.dump(Clusterer, open(clustering_folder + 'clusterer.pkl', 'wb'))
    best_frames = get_best_representatives(Clusterer.cluster_centers_, memory)
    np.savetxt(best_frames, clustering_folder + 'representatives.txt')

    transitions = infer_transitions(Clusterer, sne_trajs, terminations)
    np.savetxt(transitions, clustering_folder + 'transitions.txt')

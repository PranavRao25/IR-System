import matplotlib.pyplot as plt

precision_tfidf = [0.5378, 0.4578, 0.4074, 0.3644, 0.3298, 0.3089, 0.2902, 0.2789, 0.2617, 0.2444]
recall_tfidf = [0.0922, 0.1502, 0.1974, 0.2281, 0.2537, 0.2812, 0.3060, 0.3305, 0.3464, 0.3563]
fscore_tfidf = [0.1509, 0.2127, 0.2477, 0.2603, 0.2653, 0.2724, 0.2760, 0.2805, 0.2767, 0.2692]
map_tfidf = [0.0, 0.0922, 0.1418, 0.1743, 0.1925, 0.2074, 0.2212, 0.2323, 0.2416, 0.2476]
ndcg_tfidf = [0.0, 0.1990, 0.2421, 0.2455, 0.2374, 0.2235, 0.2146, 0.2176, 0.2150, 0.2108]
mrr_tfidf = [0.5378, 0.5978, 0.6274, 0.6363, 0.6399, 0.6436, 0.6455, 0.6499, 0.6519, 0.6528]


precision_esa = [0.28, 0.2578, 0.2415, 0.2211, 0.2044, 0.1919, 0.179, 0.1717, 0.164, 0.1556]
recall_esa = [0.0419, 0.0745, 0.1025, 0.1264, 0.1455, 0.1629, 0.1761, 0.1905, 0.2053, 0.2161]
fscore_esa = [0.0695, 0.1091, 0.1347, 0.1494, 0.1577, 0.1633, 0.1647, 0.1676, 0.1693, 0.1682]
map_esa = [0.0, 0.0419, 0.0656, 0.0826, 0.0945, 0.1049, 0.1129, 0.1167, 0.1218, 0.1266]
ndcg_esa = [0.0, 0.1616, 0.1956, 0.2053, 0.1877, 0.1843, 0.1895, 0.19, 0.1893, 0.1887]
mrr_esa = [0.28, 0.3444, 0.3681, 0.3804, 0.3813, 0.3842, 0.3893, 0.3932, 0.3952, 0.3965]


precision_lsa = [0.5956, 0.4978, 0.4296, 0.3811, 0.352, 0.3237, 0.3035, 0.28, 0.2632, 0.248]
recall_lsa = [0.1011, 0.1647, 0.2082, 0.2364, 0.2671, 0.2899, 0.3123, 0.3287, 0.3436, 0.36]
fscore_lsa = [0.1662, 0.232, 0.2603, 0.2697, 0.28, 0.282, 0.2842, 0.2793, 0.2756, 0.2716]
map_lsa = [0.0, 0.1011, 0.155, 0.1884, 0.2072, 0.2235, 0.2348, 0.2446, 0.2509, 0.2564]
ndcg_lsa = [0.0, 0.2058, 0.2216, 0.2234, 0.2272, 0.2208, 0.217, 0.2114, 0.2033, 0.198]
mrr_lsa = [0.5956, 0.6556, 0.6704, 0.6759, 0.6839, 0.6869, 0.6894, 0.6911, 0.6926, 0.6935]


fig, ax = plt.subplots(2, 3, figsize=(18, 12))

ax[0, 0].plot(range(1, 11), precision_tfidf, label='TF-IDF')
ax[0, 0].plot(range(1, 11), precision_esa, label='ESA')
ax[0, 0].plot(range(1, 11), precision_lsa, label='LSA')
ax[0, 0].set_title('Precision of different ranking algorithms')
ax[0, 0].set_xlabel('k')
ax[0, 0].set_ylabel('Precision')
ax[0, 0].legend()

ax[0, 1].plot(range(1, 11), recall_tfidf, label='TF-IDF')
ax[0, 1].plot(range(1, 11), recall_esa, label='ESA')
ax[0, 1].plot(range(1, 11), recall_lsa, label='LSA')
ax[0, 1].set_title('Recall of different ranking algorithms')
ax[0, 1].set_xlabel('k')
ax[0, 1].set_ylabel('Recall')
ax[0, 1].legend()


ax[0, 2].plot(range(1, 11), fscore_tfidf, label='TF-IDF')
ax[0, 2].plot(range(1, 11), fscore_esa, label='ESA')
ax[0, 2].plot(range(1, 11), fscore_lsa, label='LSA')
ax[0, 2].set_title('F-score of different ranking algorithms')
ax[0, 2].set_xlabel('k')
ax[0, 2].set_ylabel('F-score')
ax[0, 2].legend()


ax[1, 0].plot(range(1, 11), map_tfidf, label='TF-IDF')
ax[1, 0].plot(range(1, 11), map_esa, label='ESA')
ax[1, 0].plot(range(1, 11), map_lsa, label='LSA')
ax[1, 0].set_title('MAP of different ranking algorithms')
ax[1, 0].set_xlabel('k')
ax[1, 0].set_ylabel('MAP')
ax[1, 0].legend()


ax[1, 1].plot(range(1, 11), ndcg_tfidf, label='TF-IDF')
ax[1, 1].plot(range(1, 11), ndcg_esa, label='ESA')
ax[1, 1].plot(range(1, 11), ndcg_lsa, label='LSA')
ax[1, 1].set_title('nDCG of different ranking algorithms')
ax[1, 1].set_xlabel('k')
ax[1, 1].set_ylabel('nDCG')
ax[1, 1].legend()


ax[1, 2].plot(range(1, 11), mrr_tfidf, label='TF-IDF')
ax[1, 2].plot(range(1, 11), mrr_esa, label='ESA')
ax[1, 2].plot(range(1, 11), mrr_lsa, label='LSA')
ax[1, 2].set_title('MRR of different ranking algorithms')
ax[1, 2].set_xlabel('k')
ax[1, 2].set_ylabel('MRR')
ax[1, 2].legend()

plt.tight_layout()
plt.savefig("performance_comparison.png")
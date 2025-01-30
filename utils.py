
class FilteredImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except Exception as e:
            print(f"skip file: {path}，error: {e}")

def get_image_embeddings(images, feature_extractor, model, batch_size=32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    embeddings = []
    print('Start embedding')
    for i in tqdm(range(0, len(images), batch_size)):
        batch_images = images[i: i + batch_size]
        inputs = feature_extractor(batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    
    return np.concatenate(embeddings, axis=0)


def cluster_images(embeddings, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def cluster_images_markov(embeddings, knn_k=10, inflation=1.4):

    adjacency_matrix = kneighbors_graph(embeddings, knn_k, mode='connectivity', include_self=True).toarray()


    result = mc.run_mcl(adjacency_matrix, inflation=inflation)  
    clusters = mc.get_clusters(result)
    

    cluster_labels = np.zeros(len(embeddings))
    for label, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_labels[idx] = label
    return cluster_labels


def imbalance_score(label_counts):
    total = sum(label_counts.values())
    if total == 0:
        return 0
    max_count = max(label_counts.values())
    imbalance = max_count / total
    return imbalance




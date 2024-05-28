import math
from collections import Counter
import heapq


class SimpleKNeighborsClassifier:
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

    def predict(self, X_test):
        predicted_labels = []
        for i, x_test_entry in enumerate(X_test):
            distances = []
            for j, x_train_entry in enumerate(self.X_train):
                dist = self.euclidean_distance(x_train_entry, x_test_entry)
                #append a tuple(distance, index_training_sample)
                distances.append((dist, j))
            
            # use a heap to more efficiently find the k-nearest neighbors in the distances list   
            # compared to e.g. sorted(distances, key=lambda x:x[0])[:self.k] 
            k_nearest_neighbors = heapq.nsmallest(self.k, distances)
            
            # extract the labels of the corresponing k_nearest_neighbors and assign them to the list of votes 
            # use _, as distance is not needed from the tuple
            votes = [self.y_train[j] for _, j in k_nearest_neighbors]
            
            # majority vote: use the Counter to select the most common label 
            predicted_label = Counter(votes).most_common(1)[0][0]
            predicted_labels.append(predicted_label)
        return predicted_labels

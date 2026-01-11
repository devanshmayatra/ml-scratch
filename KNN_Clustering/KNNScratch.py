import math

class KNNScratch:
  def __init__(self,k=3):
    self.k = k
    self.X_train = []
    self.y_train = []
    
  def fit(self, X, y):
    
    """
    KNN is a 'Lazy Learner'. 
    Fit just means 'memorize the data'.
    """
    
    self.X_train = X
    self.y_train = y
    
  def _euclidean_distance(self, x1, x2):
    distance = 0.0
    for i in range(len(x1)):
      distance += (x1[i] - x2[i]) ** 2
    return math.sqrt(distance)
    
  def predict(self, X):
    predictions = []
    for x_new in X:
      # Step 1: Calculate distances to ALL training points
      distances = []
      for i in range(len(self.X_train)):
        dist = self._euclidean_distance(x_new, self.X_train[i])
        # We store a tuple: (distance, actual_label)
        distances.append((dist, self.y_train[i]))
        
      # Step 2: Sort by distance (first item in tuple)
      distances.sort(key=lambda x: x[0])
      
      # Step 3: Get K nearest neighbors
      k_nearest = distances[:self.k]
      
      # Step 4: Majority Vote
      # Extract just the labels from the k_nearest
      k_labels = [label for _, label in k_nearest]
          
      most_common = max(set(k_nearest), key=k_labels.count)
      predictions.append(most_common)
    return predictions
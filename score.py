import math
import csv

def calculate_distance(predicted_centroid, ground_truth_centroids):
    distances = []
    for ground_truth_centroid in ground_truth_centroids:
        dist = math.sqrt((predicted_centroid[0] - ground_truth_centroid[0])**2 + (predicted_centroid[1] - ground_truth_centroid[1])**2)
        distances.append(dist)
    return distances

def CsvToList(csvpath):
    data_list = []
    with open(csvpath, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data_list.append([float(row[0]), float(row[1])])  # Assuming CSV format is [x-coordinate, y-coordinate]
    return data_list

def calculate_accuracy(distances):
    correctnum=0
    for i in distances:
        if i<32.5:
            correctnum+=1
    
    accuracy=correctnum/len(distances) if correctnum > 0 else 0.0

    return accuracy,correctnum

def calculate_recall(correctnum,total_positives):
    falsenegatives=total_positives-correctnum
    if (falsenegatives+correctnum)==0:
        return 0.0
    recall=correctnum/(falsenegatives+correctnum)
    return recall


def calculate_F1Score(accuracy,recall):
    if accuracy == 0:
        return 0.0
    F1Score= (2*accuracy*recall)/ (accuracy+recall)
    return F1Score

# # Example usage
# predicted_centroids = [[1160, 1230], [150, 250], [200, 300]]  # Example predicted centroid coordinates
# csv_path = 'D:/PFE/MASTER2/2nd_test/AMIDA13/ground truth/13/06.csv'  # Example CSV file containing ground truth centroid coordinates
# ground_truth_centroids = CsvToList(csv_path)
# print(ground_truth_centroids)


# # Iterate over each predicted centroid and calculate distances to all ground truth centroids
# for i, predicted_centroid in enumerate(predicted_centroids):
#     distances = calculate_distance(predicted_centroid, ground_truth_centroids)
#     accuracy,correctnum=calculate_accuracy(distances)
#     recall=calculate_recall(correctnum,len(ground_truth_centroids))
#     F1_Score=calculate_F1Score(accuracy,recall)
#     print(f"Predicted centroid {i+1}: {predicted_centroid}, Distances to ground truth centroids: {distances}, accuracy = {accuracy}, Recall = {recall} ,F1 Score = {F1_Score}")

import os
import math
import csv

def calculate_distance(predicted_centroid, ground_truth_centroids):
    distances = []
    for ground_truth_centroid in ground_truth_centroids:
        dist = math.sqrt((predicted_centroid[0] - ground_truth_centroid[1])**2 + (predicted_centroid[1] - ground_truth_centroid[0])**2)
        distances.append(dist)
    return distances

def CsvToList(csvpath, has_header=True):
    data_list = []
    with open(csvpath, 'r') as file:
        csv_reader = csv.reader(file)
        if has_header:
            next(csv_reader)  # Skip header if present
        for row in csv_reader:
            data_list.append([float(row[0]), float(row[1])])
    return data_list

def calculate_accuracy(correct_num, total_num):
    accuracy = correct_num / total_num if total_num > 0 else 0.0
    return accuracy

def calculate_precision(correct_num, false_positives):
    precision = correct_num / (correct_num + false_positives) if (correct_num + false_positives) > 0 else 0.0
    return precision

def calculate_recall(correct_num, false_negatives):
    recall = correct_num / (correct_num + false_negatives) if (correct_num + false_negatives) > 0 else 0.0
    return recall

def calculate_F1Score(precision, recall):
    F1Score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return F1Score

def calculate_metrics(predicted_csv_dir, ground_truth_csv_dir):
    all_correct_num = 0
    all_false_positives = 0
    all_false_negatives = 0
    all_total_positives = 0
    print(predicted_csv_dir)
    
    for predicted_csv in os.listdir(predicted_csv_dir):
        #print(predicted_csv)

        if predicted_csv.endswith('.csv'):
            predicted_centroids = CsvToList(os.path.join(predicted_csv_dir, predicted_csv), has_header=True)
            test = predicted_csv[:4] + '.csv'
            ground_truth_csv_path = os.path.join(ground_truth_csv_dir, test)
            print(ground_truth_csv_path)
            if os.path.exists(ground_truth_csv_path):
                ground_truth_centroids = CsvToList(ground_truth_csv_path, has_header=False)

                correct_num = 0
                false_positives = len(predicted_centroids)
                total_positives = len(ground_truth_centroids)
                all_total_positives += total_positives

                print(f"\nProcessing file: {predicted_csv}")
                print(f"Predicted centroids: {predicted_centroids}")
                print(f"Ground truth centroids: {ground_truth_centroids}")

                for predicted_centroid in predicted_centroids:
                    distances = calculate_distance(predicted_centroid, ground_truth_centroids)
                    min_distance = min(distances) if distances else float('inf')
                    print(f"Predicted centroid: {predicted_centroid}, Distances: {distances}, Min distance: {min_distance}")
                    if min_distance < 20.5:
                        correct_num += 1
                        false_positives -= 1

                false_negatives = total_positives - correct_num
                all_correct_num += correct_num
                all_false_positives += false_positives
                all_false_negatives += false_negatives

                accuracy = calculate_accuracy(correct_num, total_positives)
                precision = calculate_precision(correct_num, false_positives)
                recall = calculate_recall(correct_num, false_negatives)
                F1_Score = calculate_F1Score(precision, recall)
                
                print(f"File: {predicted_csv}, Correct num: {correct_num}, False positives: {false_positives}, False negatives: {false_negatives}")
                print(f"File: {predicted_csv}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {F1_Score}")
            else:
                # No corresponding ground truth CSV file
                correct_num = 0
                false_positives = len(predicted_centroids)
                false_negatives = 0

                all_false_positives += false_positives

                accuracy = calculate_accuracy(correct_num, len(predicted_centroids))
                precision = calculate_precision(correct_num, false_positives)
                recall = calculate_recall(correct_num, false_negatives)
                F1_Score = calculate_F1Score(precision, recall)

                print(f"File: {predicted_csv} (no ground truth), Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {F1_Score}")

    overall_accuracy = calculate_accuracy(all_correct_num, all_total_positives)
    overall_precision = calculate_precision(all_correct_num, all_false_positives)
    overall_recall = calculate_recall(all_correct_num, all_false_negatives)
    overall_F1Score = calculate_F1Score(overall_precision, overall_recall)

    print(f"\nOverall Accuracy: {overall_accuracy}, Overall Precision: {overall_precision}, Overall Recall: {overall_recall}, Overall F1 Score: {overall_F1Score}")

# # Example usage
# predicted_csv_dir = 'path_to_predicted_csv_files'  # Directory containing predicted centroids CSV files
# ground_truth_csv_dir = 'path_to_ground_truth_csv_files'  # Directory containing ground truth centroids CSV files

# Example usage
predicted_csv_dir = 'results/csv/'  # Directory containing predicted centroids CSV files
ground_truth_csv_dir = '/home/xlyxs/Desktop/MITOS/code/mitos_swin/test/ground truth/'  # Directory containing ground truth centroids CSV files


calculate_metrics(predicted_csv_dir, ground_truth_csv_dir)

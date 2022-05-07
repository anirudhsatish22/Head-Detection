import csv
from itertools import tee
from tabulate import tabulate

def evaluate():
    '''this function runs evaluation on the bounding boxes that have been computed
    It uses the csv files with coordinates of bounding boxes around heads, and the csv files with truth
    labels for presence of faces to compute a metric for accuracy
    
    A face always indicates a head, so this is a decent measure of accuracy, but since we do not have
    annotated head information, we have to resort to this'''


    movies = ["amadeus", 'argo', 'birdman', 'chicago', 'departed', 'emperor', 'kings', 'gladiator', 'no_country_clip1', 'no_country_clip2', 'saving', 'shakespeare_clip1', 'shakespeare_clip2', 'slumdog', 'unforgiven']
    Accuracy = {}
    for movie in movies:
        truth_path = "Truth/" + movie + "_truth.csv"
        movie_path = "OutputCsv/" + movie + ".csv"
        detectedFaces, totalFrames = compute_metrics(truth_path, movie_path, movie)
        Accuracy[movie] = detectedFaces/totalFrames
    col_width =  max(len(x) for x in Accuracy.keys())
    header = [["Movie", "Accuracy"]]
    pairs = [ [x, Accuracy[x]] for x in Accuracy.keys()]
    print("-----------------  --------")
    print("      Movie      | Accuracy")
    print(tabulate(pairs))
    return Accuracy

def compute_metrics(truth_path, movie_path, movie):
    '''Helper funcion for evaluate: This runs the metric on each movie.
    Returns a Tuple, with counts of number of times a bounding box was made in a frame that contains
    a face, and the total number of frames that had faces for the clip '''

    truth = open(truth_path)
    truth_reader = csv.reader(truth)

    bounding = open(movie_path)
    bounding_reader = csv.reader(bounding)

    header = next(bounding_reader)
    Frame_Map = {}

    for row in bounding_reader:
        Frame_Map[row[0]] = row[1:]

    header = next(truth_reader)

    total_frames_with_faces = 0
    total_heads_detected = 0
    for row in truth_reader:
        frame = row[0]
        frame_converted = frame.rjust(5, '0')

        full_key = movie + "_" + frame_converted + ".png"
        if Frame_Map[full_key] and row[8] == '1':
            total_heads_detected += 1
        if row[8] == '1':
            total_frames_with_faces += 1

    return total_heads_detected, total_frames_with_faces 


evaluate()

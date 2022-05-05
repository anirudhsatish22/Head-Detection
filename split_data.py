import csv

def helper(outputWriter, currentMovie):
    '''helper function to help split the truth values into different files for each movie'''

    truth = open("truth_values.csv")
    truth_reader = csv.reader(truth)
    for row in truth_reader:
        if row[-1] == currentMovie:
            outputWriter.writerow(row)

movies = ["amadeus", 'argo', 'birdman', 'chicago', 'departed', 'emperor', 'kings', 'gladiator', 'no_country_clip1', 'no_country_clip2', 'saving', 'shakespeare_clip1', 'shakespeare_clip2', 'slumdog', 'unforgiven']



row1 = ['frame_num', 'shot_num', 'crane', 'cross_fade', 'cut', 'dia_off_screen', 'dia_on_screen', 'dolly', 'face', 'faces', 'face_other', 'handheld', 'mm_cut', 'motion', 'mult_motion', 'pan_tilt', 'rack_focus', 'text', 'tracking', 'zoom', 'film'] 

print(row1)
for currentMovie in movies:
    outputFP = open(currentMovie + "_truth"+ ".csv", "w")
    outputWriter = csv.writer(outputFP)
    outputWriter.writerow(row1)
    helper(outputWriter, currentMovie)
    outputFP.close()
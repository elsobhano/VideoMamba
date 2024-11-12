import json
import csv
from tqdm import tqdm

W_TO_ID = json.load(open('vocab.json', 'r'))["words_to_id"]

def write_to_csv(output_csv, data):
    with open(output_csv, 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(data)

def extract_sublists(numbers, sublist_size=16, slide_size=8):
    """
    Extract sublists of a given size from a list of numbers, with a sliding window.
    
    Parameters:
    numbers (list): List of consecutive numbers
    sublist_size (int): Size of each sublist to extract
    slide_size (int): Number of elements to shift the window for each step
    
    Returns:
    list: List of extracted sublists
    """
    sublists = []
    for i in range(0, len(numbers), slide_size):
        sublist = numbers[i:i+sublist_size]
        if len(sublist) < sublist_size:
            extra_len = sublist_size - len(sublist)
            sliced = numbers[i-extra_len:i]
            sublist = sliced + sublist
            if (sublist[0], sublist[-1]) not in sublists: sublists.append((sublist[0], sublist[-1]))
        else:
            if (sublist[0], sublist[-1]) not in sublists: sublists.append((sublist[0], sublist[-1]))
    return sublists

def preprocess(input_csv, output_csv):
    with open(input_csv, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        # next(csv_reader)
        for row in tqdm(csv_reader):
            file_name = row[0]
            start_frame = int(row[1])
            end_frame = int(row[2])
            class_name = row[3]
            class_id =  W_TO_ID[class_name]
            if (end_frame - start_frame + 1) > 16:
                sublists = extract_sublists(list(range(start_frame, end_frame + 1)))
                for sublist in sublists:
                    start_frame = sublist[0]
                    end_frame = sublist[1]
                    write_to_csv(output_csv, [file_name, start_frame, end_frame, class_name, class_id])
            else:
                write_to_csv(output_csv, [file_name, start_frame, end_frame, class_name, class_id])

if __name__ == "__main__":
    preprocess("test.csv", "test_refine.csv")
from glob import glob
import pandas as pd


def load_subtitles_dataset(dataset_path):
    # load dataset
    subtitles_path = glob(dataset_path+'/*.ass')
    subtitles_path.sort()

    scripts = []
    episode_num = []

    for path in subtitles_path:

        #read lines
        with open(path, 'r') as file:
            lines = file.readlines()
            lines = lines[27:]           # why removed from 0-26 lines? because those lines contains meta data, which we don't need

            new_lines = []  # Create an empty list to store modified lines

            for line in lines:  # Loop through each line in the 'lines' list
                parts = line.split(",")  # Split the line into a list using ',' as the separator
                selected_parts = parts[9:]  # Keep only the elements from index 9 onward
                new_line = ",".join(selected_parts)  # Join the selected parts back into a string
                new_lines.append(new_line)  # Add the modified line to the new list

            lines = new_lines
        
        # cleaning process
        lines = [line.replace('\\N', ' ') for line in lines]

        script = " ".join(lines)

        episode = int(path.split('-')[-1].split('.')[0].strip())

        scripts.append(script)
        episode_num.append(episode)

    df = pd.DataFrame.from_dict({"episode" : episode_num, "script" : scripts})

    return df
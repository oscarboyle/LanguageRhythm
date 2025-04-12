import csv
import os
import pandas as pd
import glob
import re

dataset = '/home/usuari/Desktop/SMC-Master/AMPLab/LanguageRhythm_DATABASES/folkKern'

language_dict = {'czech':'Czech',
                'danmark':'Danish',
                'deutschl':'German',
                'elsass':'French',
                'england':'English',
                'france':'French',
                'italia':'Italian',
                'lothring':'French',
                'magyar':'Hungarian',
                'nederlan':'Dutch',
                'oesterrh':'German',
                'polska':'Polish',
                'romania':'Romanian',
                'rossiya':'Russian',
                'sverige':'Swedish',
                'ukraina':'Ukranian'
                }

columns = ['id', 'title', 'artist_name', 'genre', 'year', 'language', 'path']
df = pd.DataFrame(columns=columns)

# Get all files from dataset in subfolcers
kern_files = glob.glob(os.path.join(dataset, '**', '*.krn'), recursive=True)

# Read each kern file and extract metadata
for file in kern_files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        title = None
        year = None
        has_ICvox = False

        for line in lines:
            line = line.strip()

            if line.startswith("!!!OTL:"):
                title = line.replace("!!!OTL:", "").strip()

            if "*ICvox" in line:
                voice = True

            if line.startswith("!!") and not line.startswith("!!!") and not year:
                match = re.search(r'\b(1[5-9][0-9]{2}|20[0-2][0-9])\b', line)
                if match:
                    year = int(match.group(0))

        if not voice:
            continue  # Skip files without *ICvox

        # Infer language from folder name
        country = file.split('/')[-2]
        language = language_dict.get(country, "Unknown")

        # Create row
        df.loc[len(df)] = {
            'id': os.path.basename(file).replace(".krn", ""),
            'title': title if title else "Unknown",
            'artist_name': "Unknown",
            'genre': "folk",
            'year': year if year else "Unknown",
            'language': language,
            'path': file
        }

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Filter out rows with missing language
df = df[df['language'] != "Unknown"]
print(df.head())
# Save to CSV
df.to_csv(dataset+'/folkKern_data.csv', index=False)


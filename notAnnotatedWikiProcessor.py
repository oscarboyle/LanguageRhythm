import musicbrainzngs
import csv
import os

FOLDER = "/Users/elenanieto/Downloads/Wikifonia"
musicbrainzngs.set_useragent("MiAppCanciones", "1.0", "micorreo@example.com")

def get_original_song(title, artist):
    try:
        result = musicbrainzngs.search_recordings(recording=title, artist=artist)
        
        if 'recording-list' not in result or not result['recording-list']:
            print(f"[INFO] No recordings found for '{title}' by '{artist}'.")
            return ["Desconocido", "Desconocido"]
        
        recordings_with_isrc = []
        recordings_without_isrc = []

        genres_set = set()  # Set to store genres from all recordings

        for recording in result['recording-list']:
            if not isinstance(recording, dict) or 'release-list' not in recording:
                continue

            for release in recording['release-list']:
                release_group = release.get('release-group', {})
                release_type = release_group.get('type', '')
                secondary_types = release_group.get('secondary-type-list', [])

                # Skip DJ mixes, compilations, live versions, and remixes
                if 'DJ-mix' in secondary_types or 'Compilation' in secondary_types:
                    continue
                if 'Live' in release_type:
                    continue
                if 'remix' in recording.get('disambiguation', '').lower():
                    continue

                score = recording.get('ext:score')
                if not score:
                    continue

                has_isrc = 'isrc-list' in recording and isinstance(recording['isrc-list'], list) and len(recording['isrc-list']) > 0

                entry = (int(score), recording['title'], recording)

                if has_isrc:
                    recordings_with_isrc.append(entry)
                else:
                    recordings_without_isrc.append(entry)
        
                # Collect genres from all recordings
                for tag in recording.get("tag-list", []):
                    genres_set.add(tag['name'])
        
        # Prefer recordings with ISRCs; otherwise, use the fallback
        used_fallback = False
        if recordings_with_isrc:
            recordings = recordings_with_isrc
        elif recordings_without_isrc:
            recordings = recordings_without_isrc
            used_fallback = True
        else:
            return ["Desconocido", "Desconocido"]
        
        # Choose the recording with the highest score
        recordings_sorted = sorted(recordings, key=lambda x: x[0], reverse=True)
        top_recording = recordings_sorted[0][2]

        # Look for the oldest release year
        years = []
        for release in top_recording.get("release-list", []):
            date = release.get("date", "")
            if date and len(date) >= 4 and date[:4].isdigit():
                years.append(int(date[:4]))
        
        # Convert genres_set to a string
        genre_str = ", ".join(genres_set) if genres_set else "Desconocido"

        if years:
            oldest_year = min(years)
            return [oldest_year, genre_str]
        else:
            return ["Desconocido", genre_str]

    except Exception as e:
        print(f"[ERROR] Error retrieving song data: {e}")
        return ["Desconocido", "Desconocido"]


# Process a folder of .mxl files
def process_folder(folder_path, output_csv="results.csv"):
    results = [] 

    for filename in os.listdir(folder_path): 
        if filename.endswith(".mxl") and " - " in filename:
            artist_title = filename[:-4]  # remove .mxl extension
            artist, title = artist_title.split(" - ", 1)
            if "," in artist:
                artist, _ = artist.split(",", 1)
            artist = artist.strip()
            title = title.strip()
            print(f"Processing: '{title}' by '{artist}'")
                    
            # Use relative path for storing file path
            path = os.path.join("Wikifonia", filename)                    
            year, genre = get_original_song(title, artist)
            results.append({"Title": title, "Artist": artist, "Year": year, "Genre": genre, "Path": path})

    # Save results to a CSV file after processing all files
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Title", "Artist", "Year", "Genre", "Path"])
        writer.writeheader()
        writer.writerows(results)

    print(f"CSV saved as {output_csv}")

# Example call
process_folder(FOLDER)

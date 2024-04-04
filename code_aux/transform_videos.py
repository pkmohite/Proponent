import json
from pytube import YouTube, Playlist
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
import csv
from moviepy.editor import VideoFileClip
import urllib.request

def download_shorten_video(json_file, destination_folder, start_time, end_time):
    # Read the JSON file
    with open(json_file) as f:
        data = json.load(f)

    # Iterate over the videos in the JSON data
    for video in data:
        # Get the video URL and title
        video_url = video["url"]
        video_title = video["title"]

        # Create a YouTube object
        yt = YouTube(video_url)

        # Filter the available streams by resolution and select the 720p stream
        stream = yt.streams.filter(res="720p").first()

        # Create the destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Download the video to the destination folder with the specified title
        stream.download(output_path=destination_folder, filename=video_title + ".mp4")

        # Get the path of the downloaded video
        video_path = os.path.join(destination_folder, video_title + ".mp4")

        # Set the path for the shortened video
        shortened_video_folder = os.path.join(destination_folder, "shortened_videos")
        if not os.path.exists(shortened_video_folder):
            os.makedirs(shortened_video_folder)
        shortened_video_path = os.path.join(shortened_video_folder, video_title + ".mp4")

        # Shorten the video using moviepy
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=shortened_video_path)

# # shortening the video
# json_file = "videos.json"
# destination_folder = "videos_titles"
# start_time = 0
# end_time = 60
# download_shorten_video(json_file, destination_folder, start_time, end_time)


def download_playlist(playlist_url, destination_folder, start_time=None, end_time=None):
    # Create a YouTube playlist object
    playlist = Playlist(playlist_url)

    # Iterate over the videos in the playlist
    count = 0
    for video in playlist.videos:
        # Get the video URL and title
        video_url = video.watch_url
        video_title = video.title.replace("|", "-").replace(":", "-")  # Replace "|" and ":" with "-"

        # Create a YouTube object
        yt = YouTube(video_url)

        # Filter the available streams by resolution and select the 480p stream
        stream = yt.streams.filter(res="360p").first()

        # Create the destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Download the video to the destination folder with the specified title
        stream.download(output_path=destination_folder.replace("\\", "/"), filename=video_title + ".mp4")

        if start_time is not None and end_time is not None:
            # Get the path of the downloaded video
            video_path = os.path.join(destination_folder, video_title + ".mp4")

            # Set the path for the shortened video
            shortened_video_folder = os.path.join(destination_folder, "shortened_videos")
            if not os.path.exists(shortened_video_folder):
                os.makedirs(shortened_video_folder)
            shortened_video_path = os.path.join(
                shortened_video_folder, video_title + ".mp4"
            )

            # Shorten the video using moviepy
            ffmpeg_extract_subclip(
                video_path, start_time, end_time, targetname=shortened_video_path
            )


# # downloading the playlist
# start_time = 0
# end_time = 60
# playlist_url = (
#     "https://www.youtube.com/playlist?list=PLutcJfNEwNkSzUWW3NW3858GhsDJ5XYHV"
# )
# destination_folder = "videos".replace("\\", "/")  # Replace backslashes with forward slashes
# download_playlist(playlist_url, destination_folder, start_time, end_time)

def extract_file_names(folder_path, csv_file_path):
    # Get all file names in the folder
    file_names = os.listdir(folder_path)

    # Open the CSV file in write mode
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the file names to the CSV file
        writer.writerow(['File Name'])
        for file_name in file_names:
            writer.writerow([file_name])

# # Example usage
# folder_path = 'monday_videos'
# csv_file_path = 'video_file_names.csv'
# extract_file_names(folder_path, csv_file_path)


def change_file_names_to_lowercase(folder_path):
    # Get all file names in the folder
    file_names = os.listdir(folder_path)

    # Iterate over the file names
    for file_name in file_names:
        # Get the current file path
        current_file_path = os.path.join(folder_path, file_name)

        # Get the new file name in lowercase
        new_file_name = file_name.lower()

        # Get the new file path
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        os.rename(current_file_path, new_file_path)

# # Example usage
# folder_path = 'videos'
# change_file_names_to_lowercase(folder_path)

def delete_nonexistent_videos(folder_path, json_file):
    # Read the JSON file
    with open(json_file) as f:
        data = json.load(f)

    # Get the list of file names from the JSON data
    file_names = [video["videoFile"] for video in data]

    # Get all file names in the folder
    all_files = os.listdir(folder_path)

    # Iterate over the files in the folder
    for file_name in all_files:
        # Check if the file name is not in the list of file names from the JSON data
        if file_name not in file_names:
            # Get the file path
            file_path = os.path.join(folder_path, file_name)

            # Delete the file
            os.remove(file_path)

# # Example usage
# folder_path = 'videos'
# json_file = "mf_embeddings.json"
# delete_nonexistent_videos(folder_path, json_file)


def download_thumbnail_images(playlist_url, destination_folder):
    # Create a YouTube playlist object
    playlist = Playlist(playlist_url)

    # Iterate over the videos in the playlist
    for video in playlist.videos:
        # Get the video URL and title
        video_url = video.watch_url
        video_title = video.title.replace("|", "-").replace(":", "-")  # Replace "|" and ":" with "-"

        # Create a YouTube object
        yt = YouTube(video_url)

        # Get the thumbnail URL
        thumbnail_url = yt.thumbnail_url

        # Create the destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Download the thumbnail image to the destination folder
        thumbnail_path = os.path.join(destination_folder, f"{video_title}.jpg")
        urllib.request.urlretrieve(thumbnail_url, thumbnail_path)

# # Example usage
# playlist_url = (
#     "https://www.youtube.com/playlist?list=PLutcJfNEwNkSzUWW3NW3858GhsDJ5XYHV"
# )
# destination_folder = "thumbnails"
# download_thumbnail_images(playlist_url, destination_folder)


def delete_nonexistent_jpgs(folder_path, json_file):
    # Read the JSON file
    with open(json_file) as f:
        data = json.load(f)

    # Get the list of file names from the JSON data
    file_names = [video["pdfFile"] for video in data]

    # Get all file names in the folder
    all_files = os.listdir(folder_path)

    # Iterate over the files in the folder
    for file_name in all_files:
        # Check if the file name is not in the list of file names from the JSON data
        if file_name not in file_names:
            # Get the file path
            file_path = os.path.join(folder_path, file_name)

            # Delete the file
            os.remove(file_path)


# # Example usage
# folder_path = 'slides2'
# json_file = "mf_embeddings.json"
# delete_nonexistent_jpgs(folder_path, json_file)


def shorten_all_videos(folder_path, destination_folder, start_time, end_time):
    # Get all file names in the folder
    file_names = os.listdir(folder_path)

    # Iterate over the file names
    for file_name in file_names:
        # Get the current file path
        current_file_path = os.path.join(folder_path, file_name)

        # Set the path for the shortened video
        shortened_video_folder = os.path.join(destination_folder, "shortened_videos")
        if not os.path.exists(shortened_video_folder):
            os.makedirs(shortened_video_folder)
        shortened_video_path = os.path.join(shortened_video_folder, file_name)

        # Shorten the video using moviepy
        ffmpeg_extract_subclip(current_file_path, start_time, end_time, targetname=shortened_video_path)
        
# # Example usage
# folder_path = 'videos'
# destination_folder = 'shortened_videos'
# start_time = 0
# end_time = 60
# shorten_all_videos(folder_path, destination_folder, start_time, end_time)
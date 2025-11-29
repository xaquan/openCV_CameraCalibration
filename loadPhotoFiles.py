# Function to load filenames of a folder in photos directory
import os
import cv2

PHOTOS_FOLDER_PATH = "photos"

def load_chessboard_filenames_from_photos_folder(folder_name):

    folder_name = PHOTOS_FOLDER_PATH + '/' + folder_name
    try:
        filenames = os.listdir(folder_name)
        return [get_photo_full_path(folder_name, filename) for filename in filenames if os.path.isfile(os.path.join(folder_name, filename))]
    except FileNotFoundError:
        print(f"Folder not found: {folder_name}")
        return []
    
def load_1cm_chessboard_filenames():
    return load_chessboard_filenames_from_photos_folder('chessboard_1cm')

def load_2cm_chessboard_filenames():
    return load_chessboard_filenames_from_photos_folder('chessboard_2cm')

# Function load a photo return opencv image objects
def load_photo(photo_filename):
    img = cv2.imread(photo_filename)
    if img is not None:
        return img
    else:
        print(f"Failed to load image: {photo_filename}")
        return None
    
# Function get full path of a photo file
def get_photo_full_path(folder_name, photo_filename):
    return os.path.join(folder_name, photo_filename)
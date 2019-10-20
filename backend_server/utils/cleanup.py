import os, shutil


def perform_cleanup():
    clean_folders = ["pics/transformations/full_transformation", "pics/transformations/original", "pics/transformations/weighted_transformation"]

    for folder in clean_folders:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    shutil.copyfile("templates/result/replacement/Result.jpg", "templates/result/Result.jpg")
    shutil.copyfile("templates/backup/index.html", "templates/index.html")

import os


def get_file_to_load():
    file_list = os.listdir("files")
    file = list(filter(lambda x: ".docx" in x or ".doc" in x, file_list))
    if len(file) != 1:
        raise Exception(
            "The word file in the files folder must be only 1 (not counting instructions.txt)"
        )

    file = [os.path.join("files", path) for path in file]

    return file[0]

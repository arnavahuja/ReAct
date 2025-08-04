import os
import json
import shutil
import constants as Constants

class Utils:

    @staticmethod
    def read_json(path):
        f = open(path, "r")
        output = json.load(f)
        f.close()
        return output
    
    @staticmethod
    def save_json(obj, path):
        f = open(path, "w")
        json.dump(obj, f, indent=4)
        f.close()
    
    @staticmethod
    def read_file(path):
        f = open(path, "r")
        output = f.read()
        f.close()
        return output
    
    @staticmethod
    def save_file(string, path):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        f = open(path, "w",  encoding="utf-8")
        f.write(string)
        f.close()

    @staticmethod
    def join_prompt(*args):
        output = ""
        for arg in args:
            output += arg
        return output

    @staticmethod
    def append_file(string, path):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        f = open(path, "a",  encoding="utf-8")
        f.write(string+"\n")
        f.close()
    
    @staticmethod
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)
    
    @staticmethod
    def delete_dir(path, nested=False):
        if os.path.isdir(path):
            try:
                os.rmdir(path)
            except OSError as e:
                if nested:
                    shutil.rmtree(path)
                else:
                    raise e
    
    @staticmethod
    def check_dir(dir_path):
        for file_name in Constants.trajectory_filenames:
            path = os.path.join(dir_path, file_name)
            if not os.path.exists(path):
                print(f"{dir_path} does not have {file_name} file")
    
    @staticmethod
    def check_all_dirs(base_path):
        for direc in os.listdir(base_path):
            Utils.check_dir(os.path.join(base_path, direc))
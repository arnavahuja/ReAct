import os
import json
import shutil


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
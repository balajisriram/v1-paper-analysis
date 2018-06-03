import os
import importlib.machinery
import types

loader = importlib.machinery.SourceFileLoader('OpenEphys',r'C:\Users\bsriram\Desktop\Code\analysis-tools\Python3\OpenEphys.py')
OpenEphys = types.ModuleType(loader.name)
loader.exec_module(OpenEphys)

def pack_and_save_dat(src,des,dref):
    OpenEphys.pack_2(src,des,dref)


if name=="__main__":
    location = r"C:\Users\bsriram\Desktop\Data_V1Paper\Raw"

    isdir = os.path.isdir
    pathjoin = os.path.join

    folders = [f for f in os.listdir(location) if isdir(pathjoin(location,f))]

    for folder in folders:
        OpenEphys.pack_2(pathjoin(location,folder), dref='med')
import os, pickle, time
        
def loadingFiles(filePath, filename):
    loadingPath = os.path.join(filePath, filename)
    print("Loading at..", loadingPath)
    with open(loadingPath, 'rb') as f:
        p = pickle.load(f)
    return p
        

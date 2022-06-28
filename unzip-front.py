from zipfile import ZipFile
import os, re

for i in os.listdir('./'):
    print(i)
    if i.endswith('.zip'):
        print(i)
        with ZipFile(i, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
            zipObj.extractall('Data/Not falling')
            print('File is unzipped in temp folder') 
    
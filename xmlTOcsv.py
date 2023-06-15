import pandas as pd
import xml.etree.ElementTree as xet

from glob import glob

path = glob("./images/*.xml")
labels_dict = dict(filepath=[], xmin=[], xmax=[], ymin=[], ymax=[])
for filename in path:
    info = xet.parse(filename)
    root = info.getroot()
    member_object = root.find("object")
    labels_info = member_object.find("bndbox")
    xmin = int(labels_info.find("xmin").text)
    xmax = int(labels_info.find("xmax").text)
    ymin = int(labels_info.find("ymin").text)
    ymax = int(labels_info.find("ymax").text)
    labels_dict["filepath"].append(filename)
    labels_dict["xmax"].append(xmax)
    labels_dict["xmin"].append(xmin)
    labels_dict["ymax"].append(ymax)
    labels_dict["ymin"].append(ymin)

df = pd.DataFrame(labels_dict)
df.to_csv("labels.csv", index=False)

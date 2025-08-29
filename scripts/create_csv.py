import os, csv, glob
import pandas as pd

ROOT = os.path.join(os.getcwd(), "data")
ann_dir = os.path.join(ROOT, "annotations")
img_dir = os.path.join(ROOT, "images")

columns = ["image_id","image_path","bbox_left","bbox_top","bbox_width","bbox_height","score","object_category"]
rows = []

for ann_file in glob.glob(os.path.join(ann_dir, "*.txt")):
    image_id = os.path.splitext(os.path.basename(ann_file))[0]
    image_path = os.path.join(img_dir, image_id + ".jpg")
    with open(ann_file, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) != 8:
                continue
            try:
                left, top, w, h = map(float, line[:4])
                score = float(line[4])
                cat = int(line[5])
            except:
                continue
            rows.append([image_id, image_path, left, top, w, h, score, cat])

df = pd.DataFrame(rows, columns=columns)
df.dropna(inplace=True)
df = df.sample(n=5000, random_state=42)  # Take only 5000 rows
df.to_csv("output/visdrone_5000.csv", index=False)
print(f"CSV created: output/visdrone_5000.csv with {len(df)} rows")

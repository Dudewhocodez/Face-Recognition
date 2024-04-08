import os

num_skipped = 0
data_dir = r'C:\Users\Devon Scheg\Documents\Academics\Classes\ECE 500\Assignments\MiniProject\data\train'

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    for fname in os.listdir(class_path):
        fpath = os.path.join(class_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")

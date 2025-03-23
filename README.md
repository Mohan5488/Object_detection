This Python script **converts Pascal VOC XML annotations to YOLO format** for object detection. Below is a detailed breakdown of how it works.

---

## **1️⃣ Overview**
- **Pascal VOC format** uses XML files for bounding boxes.
- **YOLO format** represents bounding boxes in a `.txt` file with normalized coordinates.
- The script:
  1. Reads XML files.
  2. Extracts bounding box information.
  3. Converts it to YOLO format.
  4. Saves the results in a `labels/` folder.

---

## **2️⃣ Function Breakdown**
### **🔹 `voc_to_yolo_format(xml_file, image_width, image_height, class_mapping)`**
**📌 Purpose:** Converts a single XML annotation file to YOLO format.

### **👉 Step-by-Step Explanation**
```python
tree = ET.parse(xml_file)
root = tree.getroot()
```
- Parses the XML file and extracts the root element.

```python
yolo_annotations = []
for obj in root.iter('object'):
    class_name = obj.find('name').text
    class_id = class_mapping.get(class_name, -1)
    
    if class_id == -1:
        continue
```
- Loops through each object in the XML file.
- Extracts the **class name** and finds its corresponding **ID** in `class_mapping` (`-1` means unknown class, so it's skipped).

```python
bbox = obj.find('bndbox')
xmin = int(bbox.find('xmin').text)
ymin = int(bbox.find('ymin').text)
xmax = int(bbox.find('xmax').text)
ymax = int(bbox.find('ymax').text)
```
- Extracts bounding box coordinates.

```python
x_center = (xmin + xmax) / 2 / image_width
y_center = (ymin + ymax) / 2 / image_height
width = (xmax - xmin) / image_width
height = (ymax - ymin) / image_height
```
- Converts absolute coordinates to **YOLO format** (normalized values from `0 to 1`).

```python
yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
```
- Stores the **converted annotation** as a formatted string.

🔹 **Example Output for One Object in YOLO Format:**
```
0 0.5 0.5 0.3 0.2
```
*(Class 0, bounding box centered at (0.5,0.5) with width 0.3 and height 0.2)*

---

### **🔹 `convert_voc_to_yolo(dataset_path, image_folder, annotations_folder, class_mapping)`**
**📌 Purpose:** Converts all XML files in a dataset.

```python
image_files = os.listdir(image_folder)
os.makedirs(os.path.join(dataset_path, 'labels'), exist_ok=True)  
```
- **Gets all image filenames**.
- **Creates a `labels/` folder** if it doesn’t exist.

```python
for image_file in image_files:
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        xml_file = os.path.join(annotations_folder, image_file.replace('.jpg', '.xml').replace('.png', '.xml'))
```
- **Finds the corresponding XML file** for each image.

```python
if os.path.exists(xml_file):
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
```
- **Loads the image** to get its dimensions (needed for normalization).

```python
yolo_annotations = voc_to_yolo_format(xml_file, image_width, image_height, class_mapping)
```
- **Converts XML annotations** to YOLO format.

```python
txt_file = os.path.join(dataset_path, 'labels', image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
with open(txt_file, 'w') as f:
    f.write("\n".join(yolo_annotations))
```
- **Saves YOLO annotations** in a `.txt` file.

---

## **3️⃣ Example Walkthrough**
### **📌 Given XML File (`image1.xml`)**
```xml
<annotation>
    <object>
        <name>with_mask</name>
        <bndbox>
            <xmin>50</xmin>
            <ymin>60</ymin>
            <xmax>200</xmax>
            <ymax>220</ymax>
        </bndbox>
    </object>
</annotation>
```

### **📌 Given Image (`image1.jpg`)**
- **Size:** 400 × 400 pixels

### **📌 Class Mapping**
```python
class_mapping = {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrect': 2}
```
- `"with_mask"` is **class 0**.

---

### **🔹 Step 1: Convert Bounding Box**
**XML Bounding Box:**
```
xmin = 50, ymin = 60, xmax = 200, ymax = 220
```

**Convert to YOLO Format:**
```
x_center = (50 + 200) / 2 / 400 = 0.3125
y_center = (60 + 220) / 2 / 400 = 0.35
width = (200 - 50) / 400 = 0.375
height = (220 - 60) / 400 = 0.4
```

---

### **🔹 Step 2: Output YOLO Annotation**
**Saved in `labels/image1.txt`:**
```
0 0.3125 0.35 0.375 0.4
```
*(Class 0, center at (0.3125,0.35), width 0.375, height 0.4)*

---

## **4️⃣ Running the Code**
```python
# Define dataset paths
dataset_path = '/content/Face_detection'
image_folder = '/content/Face_detection/images'
annotations_folder = '/content/Face_detection/annotations'

# Define class mapping
class_mapping = {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrect': 2}

# Convert Pascal VOC to YOLO format
convert_voc_to_yolo(dataset_path, image_folder, annotations_folder, class_mapping)
```

---

## **5️⃣ Final Folder Structure**
After running the script:
```
/Face_detection
  ├── images
  │   ├── image1.jpg
  │   ├── image2.jpg
  ├── annotations
  │   ├── image1.xml
  │   ├── image2.xml
  ├── labels  <-- (Generated)
  │   ├── image1.txt
  │   ├── image2.txt
```
---

## **✅ Summary**
- The script **converts Pascal VOC XML to YOLO format**.
- It **normalizes bounding boxes** based on image size.
- It **saves results in a `labels/` folder** as `.txt` files.

**🚀 Next Step: Train a YOLO Model**
```bash
yolo train model=yolov8n.pt data=/content/Face_detection/dataset.yaml epochs=50 imgsz=640
```

Would you like help with **training YOLO** or **verifying annotation results**? 😊

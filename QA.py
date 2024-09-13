import csv
import os
from PIL import Image

def search_image_by_index(csv_file):
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            file_name = row[0]
            file_index = file_name.split("_")[0]
            image_index = row[1]  # Assuming the image path is in the second column
            image_index = int(int(image_index)/100+1)
            image_path = f"E:/AIC2024ExtractedKeyframes/Keyframes_{file_index}/Keyframes_{file_index}/{file_name}/{file_name}_{image_index}.jpg"
            if os.path.exists(image_path):
                try:
                    command = input("Enter 'y' to show the image, 'n' to skip, 'q' to quit: ")
                    if command == 'y':
                        image = Image.open(image_path)
                        image.show()
                    elif command == 'n':
                        continue
                    elif command == 'q':
                        return None
                except IOError:
                    print(f"Error opening image file: {image_path}")
            else:
                print(f"Image file not found: {image_path}")
    return None

# Usage example:
name_path = "C:/Users/Acer/OneDrive/Desktop/QA-AIC2024/pack-pretest"
name_paths = []
for file in os.listdir(name_path):
    if "qa" in file:
        name_paths.append(file.split(".")[0])

        
    
csv_path = "C:/Users/Acer/OneDrive/Desktop/QA-AIC2024"
for name in name_paths:
    question_file =name + ".txt"
    question_path = os.path.join(name_path, question_file)
    with open(question_path, 'r') as infile:
        question=infile.read()
    print(question)
        
    name = name + ".csv"
    
    if name in os.listdir(csv_path):
        csv_file = os.path.join(csv_path, name)
        search_image_by_index(csv_file)
        # Mở file CSV ở chế độ đọc
        with open(csv_file, 'r') as infile:
            reader = csv.reader(infile)
            # Đọc các dòng dữ liệu từ file cũ
            data = list(reader)
        answer=input("Enter the answer:")
        # Thêm giá trị cho cột mới vào mỗi dòng dữ liệu
        for row in data[0:]:
            row.append(answer)

        # Mở file CSV ở chế độ ghi và ghi lại dữ liệu mới
        with open(csv_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(data)
            
            






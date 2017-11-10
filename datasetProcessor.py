import csv
import cv2
import numpy as np

# CSV description: center,left,right,steering,throttle,brake,speed

# Read whole file
rows = []
with open('./raw_data/driving_log.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        rows.append(row)

# CSV file will be destructed and populated with new data
with open('./raw_data/driving_log.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for row in rows:
        # Update path of central picture
        center_source_path = row[0]
        center_filename = center_source_path.split('/')[-1]
        center_current_path = './raw_data/IMG/' + center_filename
        center_img = cv2.imread(center_current_path)
        # Left picture
        left_source_path = row[1]
        left_filename = left_source_path.split('/')[-1]
        left_current_path = './raw_data/IMG/' + left_filename
        left_img = cv2.imread(left_current_path)
        # Right picture
        right_source_path = row[2]
        right_filename = right_source_path.split('/')[-1]
        right_current_path = './raw_data/IMG/' + right_filename
        right_img = cv2.imread(right_current_path)
        # Other parameters
        steering = float(row[3])
        throttle = float(row[4])
        brake = float(row[5])
        speed = float(row[6])
        writer.writerow([center_current_path, left_current_path,
                         right_current_path, steering,
                         throttle, brake, speed])
        
        # Flip all images
        # Center
        center_flipped_filename = './raw_data/IMG/' + 'F' + center_filename
        center_img_flipped = cv2.flip(center_img, 1)
        cv2.imwrite(center_flipped_filename, center_img_flipped)
        # Left
        left_flipped_filename = './raw_data/IMG/' + 'F' + left_filename
        left_img_flipped = cv2.flip(left_img, 1)  
        cv2.imwrite(left_flipped_filename, left_img_flipped)
        # Right
        right_flipped_filename = './raw_data/IMG/' + 'F' + right_filename
        right_img_flipped = cv2.flip(right_img, 1)  
        cv2.imwrite(right_flipped_filename, right_img_flipped)
        # Other parameters
        steering_flipped = -steering
        # Update csv file
        writer.writerow([center_flipped_filename, left_flipped_filename,
                         right_flipped_filename, steering_flipped,
                         throttle, brake, speed])

        # Random brightness
        # Brightness update was originally developed by Vivek Yadav
        # https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee
        # Center
        center_br_filename = './raw_data/IMG/' + 'B' + center_filename
        center_img_br = cv2.cvtColor(center_img, cv2.COLOR_RGB2HSV)
        center_img_br = np.array(center_img_br, dtype=np.float64)
        random_bright = .5 + np.random.uniform()
        center_img_br[:, :, 2] = center_img_br[:, :, 2] * random_bright
        center_img_br[:, :, 2][center_img_br[:, :, 2] > 255] = 255
        center_img_br = np.array(center_img_br, dtype=np.uint8)
        center_img_br = cv2.cvtColor(center_img_br, cv2.COLOR_HSV2RGB)
        cv2.imwrite(center_br_filename, center_img_br)
        # Left
        left_br_filename = './raw_data/IMG/' + 'B' + left_filename
        left_img_br = cv2.cvtColor(left_img, cv2.COLOR_RGB2HSV)
        left_img_br = np.array(left_img_br, dtype=np.float64)
        random_bright = .5 + np.random.uniform()
        left_img_br[:, :, 2] = left_img_br[:, :, 2] * random_bright
        left_img_br[:, :, 2][left_img_br[:, :, 2] > 255] = 255
        left_img_br = np.array(left_img_br, dtype=np.uint8)
        left_img_br = cv2.cvtColor(left_img_br, cv2.COLOR_HSV2RGB)
        cv2.imwrite(left_br_filename, left_img_br)
        # Right
        right_br_filename = './raw_data/IMG/' + 'B' + right_filename
        right_img_br = cv2.cvtColor(right_img, cv2.COLOR_RGB2HSV)
        right_img_br = np.array(right_img_br, dtype=np.float64)
        random_bright = .5 + np.random.uniform()
        right_img_br[:, :, 2] = right_img_br[:, :, 2] * random_bright
        right_img_br[:, :, 2][right_img_br[:, :, 2] > 255] = 255
        right_img_br = np.array(right_img_br, dtype=np.uint8)
        right_img_br = cv2.cvtColor(right_img_br, cv2.COLOR_HSV2RGB)
        cv2.imwrite(right_br_filename, right_img_br)
        # Other parameters
        steering = float(row[3])
        throttle = float(row[4])
        brake = float(row[5])
        speed = float(row[6])
        # Update csv file
        writer.writerow([center_br_filename, left_br_filename,
                         right_br_filename, steering,
                         throttle, brake, speed])
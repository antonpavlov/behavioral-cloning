import csv
import cv2

#center,left,right,steering,throttle,brake,speed
lines = []
with open('./dataUdacity/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

with open('./dataUdacity/driving_log.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = './dataUdacity/IMG/' + filename
        img = cv2.imread(current_path)
        img = cv2.normalize(img, 0, 255, norm_type=cv2.NORM_MINMAX)#, dtype=cv2.CV_32F)
        img_cropped = img[50:140, 0:320]  # Correct cropping
        cv2.imwrite(current_path, img_cropped)
        filenameFlip = './dataUdacity/IMG/' + 'F' + filename
        fileString = 'IMG/' + 'F' + filename
        img_flipped = cv2.flip(img_cropped, 1) # Correct horizontal flip (do not forget about measurements)
        cv2.imwrite(filenameFlip, img_flipped)
        steering = float(line[3])
        throttle = float(line[4])
        brake = float(line[5])
        speed = float(line[6])
        steering_flipped = -steering
        writer.writerow([fileString, line[1], line[2], steering_flipped, throttle, brake, speed])



# import os
#
import cv2
import numpy as np
# from colorama import init, Fore, Back
from pyzbar import pyzbar


# TODO: lotto do max width check and square check too

img = cv2.imread("/home/nikita-goncharov/Desktop/Failed_checks/12131.jpeg")
red_channel = img[:, :, 2]
mask = red_channel > 120
img[mask] = [255, 255, 255]
print(pyzbar.decode(img, symbols=[pyzbar.ZBarSymbol.QRCODE]))

#
# init(autoreset=True)
#
# numbers_set = set()
# found_count = 0
# non_found_count = 0
#
#
# class DashedNumParser:
#     def __init__(self, img_path):
#         self.img_path = img_path  # name file
#         self.img_filename = os.path.basename(img_path).split(".")[0].strip()
#         self.img_original = cv2.imread(img_path)
#         self.img_height, self.img_width = self.img_original.shape[:2]
#         self.dashed_number = ""
#         self.rotate_and_crop_check()
#         red_channel = self.img_original[:, :, 2]  # Red channel index is 2
#         mask = red_channel > 200
#         self.img_original[mask] = [255, 255, 255]  # Replace pixels where mask is True with white
#
#     @staticmethod
#     def crop_check(img):
#         """Extracting vertical lines through morphology operations
#
#         """
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gray = cv2.bitwise_not(gray)
#         bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
#
#         vertical = np.copy(bw)
#         height = vertical.shape[0]
#         verticalsize = height // 30
#         verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
#         vertical = cv2.erode(vertical, verticalStructure)
#         vertical = cv2.dilate(vertical, verticalStructure)
#
#         # Find vertical check contours
#         contours = cv2.findContours(vertical, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
#         contours = [cv2.boundingRect(cnt) for cnt in contours]
#         contours.sort(key=lambda cnt: cnt[3])
#         # Take two biggest(right and left), check borders
#         check_borders = contours[-2:]
#         check_borders.sort(key=lambda cnt: cnt[0])
#         border1, border2 = check_borders
#
#         result_image = img[
#             0:height,
#             border1[0] + border1[2]:border2[0]
#         ]
#         return result_image
#
#     def rotate_and_crop_check(self):
#         gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
#         gray = cv2.bitwise_not(gray)
#         thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#
#         coords = np.column_stack(np.where(thresh > 0))
#         angle = cv2.minAreaRect(coords)[-1]
#         if angle < 30:
#             angle = -angle
#         else:
#             angle = 90 - angle
#
#         print("CHECK ANGLE", angle)
#
#         h, w = self.img_height, self.img_width
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         rotated = cv2.warpAffine(self.img_original, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#
#         self.img_original = rotated
#
#         if abs(angle) > 1:
#             cropped = self.crop_check(rotated)
#             self.img_original = cropped
#
#         self.img_height, self.img_width = self.img_original.shape[:2]
#
#     def get_dashed_number(self):
#         try:
#             decoded_objects = pyzbar.decode(self.img_original, symbols=[pyzbar.ZBarSymbol.I25])
#             print("Barcode parsed: ", decoded_objects)
#             # Read with no needed "0" at the end
#             self.dashed_number = decoded_objects[0].data.decode("utf-8")[:-1]
#             # For tests
#             # prev_length = len(numbers_set)
#             # numbers_set.add(self.dashed_number)
#             # if len(numbers_set) == prev_length:
#             #     print(Back.GREEN + Fore.WHITE + "DASHED NUMBER DUPLICATED")
#             # global found_count
#             # found_count += 1
#         except Exception as ex:
#             print(Back.RED + Fore.WHITE + "DASHED NUMBER IS NOT FOUND!")
#             print(Back.RED + Fore.WHITE + str(ex))
#             # global non_found_count
#             # non_found_count += 1
#         return self.dashed_number
#
#
# if __name__ == "__main__":
#     path = "../Tickets_img"
#     files = os.listdir(path)
#     # print(files)
#     count_of_files = 0
#     for file_name in files:
#         file_path = os.path.join(path, file_name)
#         if os.path.isfile(file_path):
#             print(Back.BLUE + Fore.WHITE + "Current image: " + file_name)
#             parser = DashedNumParser(file_path)
#             parser.get_dashed_number()
#             count_of_files += 1
#
#     print("COUNT OF FILES:", count_of_files)
#     print("FOUND UNIQUE NUMBER COUNT:", len(numbers_set))
#     print("FOUND COUNT NUMBERS:", found_count)
#     print("NOT FOUND COUNT NUMBERS:", non_found_count)


#
# import cv2
# from pyzbar import pyzbar
# img = cv2.imread("lotto/in_20240116143850636746_011.jpg")  # OR JUST CHECK IF QR CODE EXISTS
#
# # QR CODE NOT FOUND
# # in_20240116164153423024_003.jpg
# # lotto/in_20240116143850636746_010.jpg
# # lotto/in_20240116104124482432_010.jpg
# # lotto/in_20240116143850636746_011.jpg
#
#
# red_channel = img[:, :, 2]  # Red channel index is 2
# mask = red_channel > 150
# img[mask] = [255, 255, 255]  # Replace pixels where mask is True with white
#
# # image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # wb_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
#
# decoded_objects = pyzbar.decode(img, symbols=[pyzbar.ZBarSymbol.QRCODE])
# print(decoded_objects)
#
# import cv2
#
# img = cv2.imread("table.jpg", 0)
#
#
# def find_table_frame_lines(img):
#     h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
#     h_dilated = cv2.dilate(img, h_kernel)
#     h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (71, 1))
#     h_dilated = cv2.dilate(cv2.threshold(h_dilated, 127, 255, cv2.THRESH_BINARY_INV)[1], h_k)
#     contours = cv2.findContours(h_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
#     contours = [cv2.boundingRect(cnt) for cnt in contours]
#     # sort by width, if there are few lines(image cropped with main lines), then needed line will smallest
#     contours.sort(key=lambda cnt: cnt[2])
#     h_line = contours[0]
#
#     v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
#     v_dilated = cv2.dilate(img, v_kernel)
#     k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 71))
#     v_dilated = cv2.dilate(cv2.threshold(v_dilated, 127, 255, cv2.THRESH_BINARY_INV)[1], k)
#     contours = cv2.findContours(v_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
#     contours = [cv2.boundingRect(cnt) for cnt in contours]
#     contours.sort(key=lambda cnt: cnt[3], reverse=True)  # here vertical line will the biggest by height
#     v_line = contours[0]
#
#     # cv2.imshow("h", h_dilated)
#     # cv2.imshow("v", v_dilated)
#     # cv2.waitKey(0)
#     return {"h_line": h_line, "v_line": v_line}  # {"h_line": (x, y, w, h), "v_line": (x, y, w, h)}
#
# height, width = img.shape[:2]
# img = img[   # crop img sides because vertical lines can be there
#     0:height,
#     50:width-50
# ]
# lines = find_table_frame_lines(img)
#
# h_line = lines["h_line"]
# v_line = lines["v_line"]
# regular_img = img[
#     h_line[1]:v_line[1]+v_line[3], # from y horizontal line to last point of vertical line
#     h_line[0]:v_line[0]  # from x of horizontal line to x of vertical_line
# ]
# strong_img = img[
#     h_line[1]:v_line[1]+v_line[3],  # from y horizontal line to last point of vertical line
#     v_line[0]:h_line[0]+h_line[2]  # from x of horizontal line to x of vertical_line
# ]
#
# cv2.imshow("Regular", regular_img)
# cv2.imshow("Strong", strong_img)
# cv2.waitKey(0)
#
#



# LOTTO REGULAR PREV VERSION

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
# dilated_table = cv2.dilate(inverted_table, kernel)
# self.save_pic_debug(dilated_table, f"table/dilated_table.jpg")
# lines_contours = cv2.findContours(dilated_table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# lines_contours = [cv2.boundingRect(cnt) for cnt in lines_contours]
# sorted_lines_contours = [contour for contour in lines_contours if
#                          contour[2] >= 700 and contour[2] <= 800 and contour[3] >= 40]
# sorted_lines_contours.sort(key=lambda line: line[1])
#
# for index, line in enumerate(sorted_lines_contours):
#     self.check_info["table"][f"line_{index + 1}"] = {
#         "regular": [],
#         "strong": [],
#     }
#     x, y, w, h = line
#     line_img = crop_img[
#         y:y + h,
#         x:x + w
#     ]
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 21))
#     k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
#     closing = cv2.morphologyEx(
#         line_img,
#         cv2.MORPH_OPEN,
#         k
#     )
#     self.save_pic_debug(closing, f"table/closing_line({index + 1}).jpg")
#     dilated_line = cv2.dilate(
#         cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY_INV)[1],
#         kernel
#     )
#     self.save_pic_debug(dilated_line, f"table/dilated_line({index+1}).jpg")
#     number_contours = cv2.findContours(dilated_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
#     number_contours = [cv2.boundingRect(cnt) for cnt in number_contours]
#     self.job_log(f"Found numbers contours: {number_contours}")
#     # Contours paired numbers and strong number contour and "()"
#     sorted_number_contours = [contour for contour in number_contours if
#                               contour[2] >= 15 and contour[3] >= 40]
#     self.job_log(f"Remove garbage contours: {sorted_number_contours}")
#     sorted_number_contours.sort(key=lambda cnt: cnt[0])
#     self.job_log(f"Lotto table sorted number contours: {sorted_number_contours}")
#
#     regular_numbers, strong_numbers = self.split_lotto_number_contours(sorted_number_contours)
#     print(regular_numbers, strong_numbers)

# # Find regular numbers
# for j, number in enumerate(regular_numbers):
#     x, y, w, h = number
#     digit1_img = line_img[
#         y:y + h,
#         x:x+w//2
#     ]
#     digit2_img = line_img[
#         y:y + h,
#         x+w//2:x+w
#     ]
#     self.save_pic_debug(digit1_img, f"table/digit1({index + 1}, {j}).jpg")
#     self.save_pic_debug(digit2_img, f"table/digit2({index + 1}, {j}).jpg")
#     digit1 = self.get_value_from_image(digit1_img, "table")
#     digit2 = self.get_value_from_image(digit2_img, "table")
#     self.check_info["table"][f"line_{index + 1}"]["regular"].append(int(f"{digit1}{digit2}"))
#
# # Find strong number, for regular exists only one strong number for line
# x, y, w, h = strong_numbers[0]  # first contour it is strong number
# number_img = line_img[
#     y:y + h,
#     x:x + w
# ]
# table_number = self.get_value_from_image(number_img, "table")
# self.check_info["table"][f"line_{index + 1}"]["strong"].append(int(table_number))

#
# import cv2
# import numpy as np
#
# # Load the image
# image = cv2.imread("/home/nikita-goncharov/Desktop/img2.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Perform edge detection
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#
# # Perform Hough Line Transform
# lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
#
# # Calculate the length of each line and sort them by length
# lines = sorted(lines, key=lambda x: abs(x[0, 1]))
#
# # Extract the longest lines
# longest_lines = lines[-2:]  # Assuming there are two longest lines for the sides of the check
#
# # Draw the longest lines on the image
# for line in longest_lines:
#     print(line)
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))
#
#     # Draw the detected line on the image
#     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
# # Calculate the angle in degrees
# angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
# print("Detected angle:", angle)
# angle = 90 - angle if angle > 45 else angle
# print("Detected angle:", angle)
#
# # Rotate the image
# (h, w) = image.shape[:2]
# center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, angle, 1.0)
# rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#
# # Display the results
# cv2.imshow("Original Image", image)
# cv2.imshow("Rotated Image", rotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# # Current angle finding
# img = cv2.imread("/home/nikita-goncharov/Desktop/Failed_checks/img1.jpg")  # "lotto/in_20240116143850636746_012.jpg")
#
# red_channel = img[:, :, 2]  # Red channel index is 2
# mask = red_channel > 230
# img[mask] = [255, 255, 255]
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.bitwise_not(gray)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#
# coords = np.column_stack(np.where(thresh > 0))
# angle = cv2.minAreaRect(coords)[-1]
# # print("CHECK ANGLE", angle)
# if angle < 30:
#     angle = -angle
# else:
#     angle = 90 - angle
#
# print("CHECK ANGLE", angle)
#
# h, w = img.shape[:2]
# center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, angle, 1.0)
# rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
# cv2.imshow("", rotated)
# cv2.waitKey(0)


# import cv2
# import numpy as np
#
# image = cv2.imread("rotated_check.jpg")
#
#
# def crop_img(img):
#     """Extracting vertical lines through morphology operations
#
#     """
#     h, w = img.shape[:2]
#     hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
#     h_min = np.array((0, 0, 0), np.uint8)
#     h_max = np.array((255, 255, 213), np.uint8)
#     thresh = cv2.inRange(hsv, h_min, h_max)
#
#     wb_img = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY_INV)[1]
#
#     v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
#     wb_img = cv2.dilate(wb_img, v_kernel)
#
#     k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
#     wb_img = cv2.morphologyEx(wb_img, cv2.MORPH_OPEN, k)
#
#     contours = cv2.findContours(cv2.threshold(wb_img, 127, 255, cv2.THRESH_BINARY_INV)[1], cv2.RETR_LIST,
#                                 cv2.CHAIN_APPROX_SIMPLE)[0]
#     contours = [cv2.boundingRect(cnt) for cnt in contours]
#     contours = [cnt for cnt in contours if cnt[3] >= (h // 3) * 2]
#     contours.sort(key=lambda cnt: cnt[3])  # sort by height
#     side_lines = contours[-2:]
#     side_lines.sort(key=lambda cnt: cnt[0])
#     if len(side_lines) == 2:
#         l_line = side_lines[0]
#         r_line = side_lines[1]
#         resulted_img = img[
#                        0:h,
#                        l_line[0] + l_line[2]:r_line[0]
#                        ]
#     else:  # one line
#         if side_lines[0][0] < w // 2:  # if line OX in first half of image then it is left line else right
#             l_line = side_lines[0]
#             resulted_img = img[
#                            0:h,
#                            l_line[0] + l_line[2]:w
#                            ]
#         else:
#             r_line = side_lines[0]
#             resulted_img = img[
#                            0:h,
#                            0:r_line[0]
#                            ]
#     return resulted_img
#
#
# cv2.imshow("", crop_img(image))
# cv2.waitKey(0)
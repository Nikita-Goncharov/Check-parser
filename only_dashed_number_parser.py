import os

import cv2
import numpy as np
from colorama import init, Fore, Back
from pyzbar import pyzbar

init(autoreset=True)

numbers_set = set()
found_count = 0
non_found_count = 0


class DashedNumParser:
    def __init__(self, img_path):
        self.img_path = img_path  # name file
        self.img_filename = os.path.basename(img_path).split(".")[0].strip()
        self.img_original = cv2.imread(img_path)
        self.img_height, self.img_width = self.img_original.shape[:2]
        self.dashed_number = ""
        self.rotate_and_crop_check()
        red_channel = self.img_original[:, :, 2]  # Red channel index is 2
        mask = red_channel > 200
        self.img_original[mask] = [255, 255, 255]  # Replace pixels where mask is True with white

    @staticmethod
    def crop_check(img):
        """Extracting vertical lines through morphology operations

        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        vertical = np.copy(bw)
        height = vertical.shape[0]
        verticalsize = height // 30
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)

        # Find vertical check contours
        contours = cv2.findContours(vertical, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [cv2.boundingRect(cnt) for cnt in contours]
        contours.sort(key=lambda cnt: cnt[3])
        # Take two biggest(right and left), check borders
        check_borders = contours[-2:]
        check_borders.sort(key=lambda cnt: cnt[0])
        border1, border2 = check_borders

        result_image = img[
            0:height,
            border1[0] + border1[2]:border2[0]
        ]
        return result_image

    def rotate_and_crop_check(self):
        gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < 30:
            angle = -angle
        else:
            angle = 90 - angle

        print("CHECK ANGLE", angle)

        h, w = self.img_height, self.img_width
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.img_original, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        self.img_original = rotated

        if abs(angle) > 1:
            cropped = self.crop_check(rotated)
            self.img_original = cropped

        self.img_height, self.img_width = self.img_original.shape[:2]

    def get_dashed_number(self):
        try:
            decoded_objects = pyzbar.decode(self.img_original, symbols=[pyzbar.ZBarSymbol.I25])
            print("Barcode parsed: ", decoded_objects)
            # Read with no needed "0" at the end
            self.dashed_number = decoded_objects[0].data.decode("utf-8")[:-1]
            # For tests
            # prev_length = len(numbers_set)
            # numbers_set.add(self.dashed_number)
            # if len(numbers_set) == prev_length:
            #     print(Back.GREEN + Fore.WHITE + "DASHED NUMBER DUPLICATED")
            # global found_count
            # found_count += 1
        except Exception as ex:
            print(Back.RED + Fore.WHITE + "DASHED NUMBER IS NOT FOUND!")
            print(Back.RED + Fore.WHITE + str(ex))
            # global non_found_count
            # non_found_count += 1
        return self.dashed_number


if __name__ == "__main__":
    path = "../Tickets_img"
    files = os.listdir(path)
    # print(files)
    count_of_files = 0
    for file_name in files:
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            print(Back.BLUE + Fore.WHITE + "Current image: " + file_name)
            parser = DashedNumParser(file_path)
            parser.get_dashed_number()
            count_of_files += 1

    print("COUNT OF FILES:", count_of_files)
    print("FOUND UNIQUE NUMBER COUNT:", len(numbers_set))
    print("FOUND COUNT NUMBERS:", found_count)
    print("NOT FOUND COUNT NUMBERS:", non_found_count)

#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import logging
import logging.config
import datetime
import random

import cv2
import numpy as np
from pyzbar import pyzbar

from elements_coords_in_img import (game_types,
                                    d_all_symbols,
                                    d_table_123_numbers,
                                    d_table_777_numbers,
                                    d_table_lotto_numbers,
                                    cards_clubs, cards_diamonds, cards_hearts, cards_spades,
                                    d_extra_numbers)


def step_decorator(l_name_step):
    """Created just for log status and time of each step of check parser

    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            # Attention! When we pass args and kwargs for object method, first arg will be self(object)
            self = args[0]
            self.StepTimeStart = datetime.datetime.now()
            self.StepNum += 1
            return_value = fn(*args, **kwargs)
            TimeTask = str((datetime.datetime.now() - self.TaskTimeStart))
            TimeTest = str((datetime.datetime.now() - self.StepTimeStart))
            if self.StepLogAdd:
                str_log = f"{self.StepLogAdd}"
                self.StepLog += str_log + "\n"
            self.StepLogAdd = ''
            str_log = f"  {str(self.StepNum).zfill(2)} = {TimeTask} = {TimeTest} = {l_name_step} = {self.StepStatus}"
            self.StepLog += str_log + "\n"
            self.job_error = True if self.StepStatus != "OK" else False
            self.StepStatus = "OK"
            self.job_log(str_log)
            return return_value

        return wrapper

    return decorator


class ParserChecks:
    # Distance from lines to needed elements
    games_elements_distance = {
        "123": {
            "spaced_number": [340, "bottom_line"],
            "date": [55, "bottom_line"],
            "game_id": [20, "bottom_line"],
            "sum": [240, "bottom_line"],
            "game_subtype": [75, "bottom_line"],
            "game_type": [180, "top_line"],
        },
        "777": {
            "spaced_number": [310, "bottom_line"],
            "date": [55, "bottom_line"],
            "game_id": [20, "bottom_line"],
            "sum": [210, "bottom_line"],
            "game_subtype": [75, "bottom_line"],
            "game_type": [180, "top_line"],
        },
        "chance": {
            "spaced_number": [340, "bottom_line"],
            "date": [55, "bottom_line"],
            "game_id": [20, "bottom_line"],
            "sum": [240, "bottom_line"],
            "game_subtype": [75, "bottom_line"],
            "game_type": [190, "top_line"],
        },
        "lotto": {
            "spaced_number": [310, "bottom_line"],
            "date": [55, "bottom_line"],
            "game_id": [20, "bottom_line"],
            "sum": [210, "bottom_line"],
            "game_subtype": [75, "middle_line"],
            "game_type": [000, "top_line"],
        }
    }

    def __init__(self, img_path, log_dir="", debug_img_dir=""):
        self.img_path = img_path  # name file
        self.img_filename = os.path.basename(img_path).split(".")[0].strip()
        # Template images dir
        # The future usage in example: cv2.imread(os.path.join(self.template_images_dir, "image_name.png"))
        self.template_images_dir = "template_images"

        # dir for log
        if log_dir:
            self.log_dir = os.path.join(log_dir, "parser")
        else:
            self.log_dir = os.path.join("..", "log", "parser")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.debug_mode = True  # dir for debug img

        if debug_img_dir:
            self.debug_img_dir = os.path.join(log_dir, "parser", "debug_img", self.img_filename)
        else:
            self.debug_img_dir = os.path.join("..", "log", "parser", "debug_img", self.img_filename)
        if not os.path.exists(self.debug_img_dir):
            os.makedirs(self.debug_img_dir)

        self.StepNum = 0  # num step prog
        # TimeStamp
        self.TaskTimeStart = datetime.datetime.now()
        self.StepTimeStart = datetime.datetime.now()
        self.StepStatus = "OK"  # статус каждого шага программы
        # лог по каждому шагу программы
        self.StepLog = ""
        self.StepLogAdd = ""
        self.StepLogFull = ""  # лог по каждому шагу программы полный
        self.job_error = False
        self.history_job = {}
        self.job_log_text = ""
        self.gl_name = ""
        self.gl_name_short = ""
        self.img_is_valid = False  # flag check img
        self.all_data_not_found = False
        # start log
        self.set_log()
        self.job_log("" + "*" * 40)
        self.job_log(f'***** START {img_path} ******')

    def set_log(self):
        log_file = os.path.join(self.log_dir, f'parser_cabala.log')
        name_logger = f"parser_cabala_{self.img_filename}_{datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S%f')[:-3]}"
        #
        DictLOGGING = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    # 'format': '%(asctime)s; %(levelname)s:%(name)s: %(message)s (%(filename)s:%(lineno)d)',
                    'format': '%(asctime)s; %(levelname)s %(message)s (%(filename)s:%(lineno)d)',
                }
            },
            'handlers': {
                'console': {
                    'level': 'DEBUG',
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                },
                'rotate_file': {
                    'level': 'DEBUG',
                    'formatter': 'standard',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': log_file,
                    'encoding': 'utf8',
                    'maxBytes': 200000,
                    'backupCount': 30,
                }
            }
            ,
            'loggers': {
                '': {
                    'handlers': ['console', 'rotate_file'],
                    # 'handlers': ['rotate_file'],
                    # 'level': 'DEBUG',
                    'level': 'INFO',
                },
            }
        }
        #
        logging.config.dictConfig(DictLOGGING)
        self.log_parser = logging.getLogger(name_logger)

    @step_decorator("set_param")
    def set_param(self):
        self.qr_code_info = {}
        self.main_data_contours = {
            "game_ids_count": 0,
            "game_id": (),
            "date": (),
            "sum": (),
            "spaced_number": (),
            "dashed_number": ()
        }
        # will like {
        # "top_border_line": {...},
        # "top_line": {"min_x": min_x, "max_x": max_x, "y": y},
        # "middle_line": {...},
        # "bottom_line": {"min_x": min_x, "max_x": max_x, "y": y}
        # }
        self.longest_lines = {}
        # self.bottom_oy_border_for_table not zero if subtype exists between content lines
        self.bottom_oy_border_for_table = 0

        self.check_info = {
            "qr_code_link": "",
            "date": "",
            "game_type": "",
            "game_subtype": "",
            "is_double": False,
            "game_id": "",
            "spent_on_ticket": 0.0,
            "dashed_number": "",
            "spaced_number": "",
            "extra": False,
            "extra_number": "",
            "is_table_pattern": False,
            "cards": {
                "hearts": [],
                "diamonds": [],
                "spades": [],
                "clubs": []
            },
            "table": {}
        }
        # if here exists pairs, then that data was parsed incorrect, example: "dashed_number": True
        self.is_incorrect_check_info = {}
        self.qrcode_found = False

    def job_log(self, l_text=""):
        self.log_parser.info(l_text)
        self.StepLogFull += "\n" + l_text

    def save_pic_debug(self, pic, l_work_name):
        try:
            if self.debug_mode:
                dirs, file_name = os.path.split(l_work_name)
                check_path = os.path.join(self.debug_img_dir, dirs)
                if not os.path.exists(check_path):
                    os.makedirs(check_path)
                m_pic_path = os.path.join(self.debug_img_dir, l_work_name)
                cv2.imwrite(m_pic_path, pic)
        except Exception as ex:
            print("Exception  m = ", ex)

    def qr_code_read(self):
        # read qrcode
        next_red_channel = 220
        removed_red_img = self.img_original.copy()
        # Decrease red channel of pixels, while qr code is not decoded
        while next_red_channel > 30:
            decoded_objects = pyzbar.decode(removed_red_img, symbols=[pyzbar.ZBarSymbol.QRCODE])
            if decoded_objects:
                break

            red_channel = removed_red_img[:, :, 2]
            mask = red_channel > next_red_channel
            removed_red_img[mask] = [255, 255, 255]
            next_red_channel -= 20

        print("******************")
        print("decoded_objects = ", decoded_objects)
        print("******************")
        '''
        decoded_objects = [Decoded(
            data=b'https://www.pais.co.il/qr.aspx?q=eNozMjUAA0MwaW5sbIGgQMDEwMLAss7AHKjAqM7CzMAAAPMwCiwAAAAAAAAAAAAAAA==\n',
            type='QRCODE', 
            rect=Rect(left=320, top=1646, width=326, height=328),
            polygon=[
                Point(x=320, y=1646), 
                Point(x=320, y=1974), 
                Point(x=646, y=1972), 
                Point(x=646, y=1646)
            ],
            quality=1,
            orientation='DOWN')]
        '''
        if len(decoded_objects) > 0:
            for item_decoded_objects in decoded_objects:
                self.qrcode_data = item_decoded_objects.data.decode('utf-8')
                self.qrcode_type = item_decoded_objects.type
                self.qrcode_rect = item_decoded_objects.rect
                self.qrcode_rect_left = self.qrcode_rect.left
                self.qrcode_rect_top = self.qrcode_rect.top
                self.qrcode_rect_width = self.qrcode_rect.width
                self.qrcode_rect_height = self.qrcode_rect.height
                self.qrcode_polygon = item_decoded_objects.polygon
                self.qrcode_point_left_top_x   = self.qrcode_polygon[0].x
                self.qrcode_point_left_top_y   = self.qrcode_polygon[0].y
                self.qrcode_point_left_down_x  = self.qrcode_polygon[1].x
                self.qrcode_point_left_down_y  = self.qrcode_polygon[1].y
                self.qrcode_point_right_down_x = self.qrcode_polygon[2].x
                self.qrcode_point_right_down_y = self.qrcode_polygon[2].y
                self.qrcode_point_right_top_x  = self.qrcode_polygon[3].x
                self.qrcode_point_right_top_y  = self.qrcode_polygon[3].y
                self.qrcode_point_centr_x  = self.qrcode_rect_left + int(self.qrcode_rect_width / 2)
                self.qrcode_point_centr_y  = self.qrcode_rect_top + int(self.qrcode_rect_height / 2)
                self.qrcode_quality = item_decoded_objects.quality
                self.qrcode_orientation = item_decoded_objects.orientation

                print("qrcode_data               = ", self.qrcode_data)
                print("qrcode_type               = ", self.qrcode_type)
                print("qrcode_rect               = ", self.qrcode_rect)
                print("qrcode_rect_left          = ", self.qrcode_rect_left)
                print("qrcode_rect_top           = ", self.qrcode_rect_top)
                print("qrcode_rect_width         = ", self.qrcode_rect_width)
                print("qrcode_rect_height        = ", self.qrcode_rect_height)
                print("qrcode_polygon            = ", self.qrcode_polygon)
                print("qrcode_point_left_top_x   = ", self.qrcode_point_left_top_x)
                print("qrcode_point_left_top_y   = ", self.qrcode_point_left_top_y)
                print("qrcode_point_left_down_x  = ", self.qrcode_point_left_down_x)
                print("qrcode_point_left_down_y  = ", self.qrcode_point_left_down_y)
                print("qrcode_point_right_down_x = ", self.qrcode_point_right_down_x)
                print("qrcode_point_right_down_y = ", self.qrcode_point_right_down_y)
                print("qrcode_point_right_top_x  = ", self.qrcode_point_right_top_x)
                print("qrcode_point_right_top_y  = ", self.qrcode_point_right_top_y)
                print("qrcode_point_centr_x      = ", self.qrcode_point_centr_x)
                print("qrcode_point_centr_y      = ", self.qrcode_point_centr_y)
                print("qrcode_quality            = ", self.qrcode_quality)
                print("qrcode_orientation        = ", self.qrcode_orientation)

                if "pais.co.il" in self.qrcode_data:
                    print("QRcode is valid ***********************")
                    self.qrcode_found = True
                    break

    @staticmethod
    def resize_img(img, target_width=972):
        height, width = img.shape[:2]
        ratio = target_width / width
        target_height = int(height * ratio)

        resized_image = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        return resized_image

    @step_decorator("img_prep")
    def img_prep(self):
        try:
            self.StepLogAdd = f'  file = {self.img_path}'
            self.img_original = cv2.imread(self.img_path)
            self.img_height, self.img_width = self.img_original.shape[:2]
            self.img_height_centr, self.img_width_centr = int(self.img_height/2), int(self.img_width/2)
            self.StepLogAdd = f'\n  img_height, img_width = {self.img_height}, {self.img_width}'
            self.StepLogAdd = f'\n  img_height_center, img_width_center = {self.img_height_centr}, {self.img_width_centr}'

            self.save_pic_debug(self.img_original, f"img_original_01.jpg")

            # img check
            self.qr_code_read()
            if self.qrcode_found:
                if self.img_height > self.img_width:
                    qrcode_position_gv = "vertical"
                    if self.qrcode_point_centr_y > self.img_height_centr:
                        qrcode_position = "down"  # если вверх ногами - поворот 180 rotateCode=1 180 градусов по часовой
                        img_rotate = 1
                    else:
                        qrcode_position = "top"
                        img_rotate = None
                else:
                    qrcode_position_gv = "gorizont"
                    if self.qrcode_point_centr_x > self.img_width_centr:
                        qrcode_position = "right"  # если повернут 90  - поворот 270 rotateCode=2 270 градусов по часовой
                        img_rotate = 2
                    else:
                        qrcode_position = "left"  # если повернут 270 - поворот 90  rotateCode=0 90 градусов по часовой
                        img_rotate = 0
                print("qrcode_position_gv = ", qrcode_position_gv)
                print("qrcode_position    = ", qrcode_position)
                print("img_rotate         = ", img_rotate)

                if img_rotate:
                    self.img_original = cv2.rotate(self.img_original, img_rotate)

                self.rotate_and_crop_check()
                self.img_original = self.resize_img(self.img_original)
                self.qr_code_read()

                self.img_height, self.img_width = self.img_original.shape[:2]
                self.img_height_centr, self.img_width_centr = int(self.img_height / 2), int(self.img_width / 2)
                print("self.img_height, self.img_width = ", self.img_height, self.img_width)
                print("self.img_height_centr, self.img_width_centr = ", self.img_height_centr, self.img_width_centr)

                self.save_pic_debug(self.img_original, f"img_original_02.jpg")

                # после поворота нужны новые координаты QRcode
                # link
                # corners_coords = [
                #     [corners_coords[0], corners_coords[1]],
                #     [corners_coords[2], corners_coords[1]],
                #     [corners_coords[2], corners_coords[3]],
                #     [corners_coords[0], corners_coords[3]]
                # ]

                self.qr_code_info = {
                        "link": self.qrcode_data,
                        "top_left": (self.qrcode_point_left_top_x, self.qrcode_point_left_top_y),
                        "top_right": (self.qrcode_point_right_top_x, self.qrcode_point_right_top_x),
                        "bottom_right": (self.qrcode_point_right_down_x, self.qrcode_point_right_down_y),
                        "bottom_left": (self.qrcode_point_left_down_x, self.qrcode_point_left_down_y),
                        "middle_line": self.qrcode_point_right_top_x - self.qrcode_point_left_top_x  # value by OX
                    }
                self.check_info["qr_code_link"] = self.qrcode_data

                red_channel = self.img_original[:, :, 2]  # Red channel index is 2
                mask = red_channel > 200
                self.img_original[mask] = [255, 255, 255]  # Replace pixels where mask is True with white

                image = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
                wb_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
                blured_img_main = cv2.bilateralFilter(wb_image, 1, 75, 75)
                self.wb_blured_img = cv2.threshold(blured_img_main, 170, 255, cv2.THRESH_TOZERO)[1]
                self.save_pic_debug(self.wb_blured_img, f"img_wb_blured_03.jpg")

                print("FINISH img_prep")

                self.img_is_valid = True
            else:
                self.img_is_valid = False
                self.all_data_not_found = True
        except Exception as ex:
            self.img_is_valid = False
            self.all_data_not_found = True
            self.job_log(f'  check_prepare_pict_02 = {ex} ')
            self.StepStatus = ex

    @staticmethod
    def crop_img(img):
        """Extracting vertical lines through morphology operations

        """
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
        h_min = np.array((0, 0, 0), np.uint8)
        h_max = np.array((255, 255, 213), np.uint8)
        thresh = cv2.inRange(hsv, h_min, h_max)

        wb_img = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY_INV)[1]

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
        wb_img = cv2.dilate(wb_img, v_kernel)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
        wb_img = cv2.morphologyEx(wb_img, cv2.MORPH_OPEN, k)

        contours = cv2.findContours(cv2.threshold(wb_img, 127, 255, cv2.THRESH_BINARY_INV)[1], cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [cv2.boundingRect(cnt) for cnt in contours]
        contours = [cnt for cnt in contours if cnt[3] >= (h//3)*2]
        contours.sort(key=lambda cnt: cnt[3])  # sort by height
        side_lines = contours[-2:]
        print("CHECK CROP SIDE LINES", side_lines)
        side_lines.sort(key=lambda cnt: cnt[0])
        if len(side_lines) == 2:
            l_line = side_lines[0]
            r_line = side_lines[1]
            resulted_img = img[
                0:h,
                l_line[0] + l_line[2]:r_line[0]
            ]
        elif len(side_lines) == 1:  # one line
            if side_lines[0][0] < w//2:  # if line OX in first half of image then it is left line else right
                l_line = side_lines[0]
                resulted_img = img[
                    0:h,
                    l_line[0] + l_line[2]:w
                ]
            else:
                r_line = side_lines[0]
                resulted_img = img[
                    0:h,
                    0:r_line[0]
                ]
        else:
            resulted_img = img
        return resulted_img

    def rotate_and_crop_check(self):
        red_channel = self.img_original[:, :, 2]  # Red channel index is 2
        mask = red_channel > 230
        self.img_original[mask] = [255, 255, 255]

        # bordered_img = cv2.copyMakeBorder(
        #     self.img_original,
        #     top=15,
        #     bottom=15,
        #     left=15,
        #     right=15,
        #     value=[179, 171, 182],
        #     borderType=cv2.BORDER_CONSTANT
        # )

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

        self.save_pic_debug(rotated, f"rotated_check.jpg")

        if abs(angle) >= 0.2:
            cropped = self.crop_img(rotated)
            self.img_original = cropped
            self.save_pic_debug(cropped, f"cropped_check.jpg")
        self.img_height, self.img_width = self.img_original.shape[:2]

    @staticmethod
    def _unique_contours(contours, data_type="any_else_data"):
        unique_contours = []
        for index, contour in enumerate(contours):
            if index == 0:
                unique_contours.append(contour)
                continue
            else:
                # Append contour if contour OX more than prev by (minimal_distance)px
                last_unique_contour = unique_contours[-1]
                if data_type == "cards":
                    # For cards OY bigger than prev card coords then it is new card
                    condition = contour[1] - last_unique_contour[1] >= 30
                else:
                    condition = contour[0] - last_unique_contour[0] >= 13
                if condition:
                    unique_contours.append(contour)
                else:
                    # Else check square, if current bigger then swap contour in unique_contours
                    last_unique_contour_square = last_unique_contour[2] * last_unique_contour[3]
                    contour_square = contour[2] * contour[3]
                    if contour_square > last_unique_contour_square:
                        unique_contours.pop()
                        unique_contours.append(contour)

        return unique_contours

    def merge_lines_contours(self, contours):
        contours = self._order_contours_in_lines(contours)
        result_lines_points = []
        for i, line in enumerate(contours):
            line.sort(key=lambda cnt: cnt[0])
            if len(line) == 1:
                result_lines_points.append(line[0])
            else:
                # resulted_line = []
                # for j, segment in enumerate(line):
                #     if j == 0:
                #         resulted_line = segment
                #     else:
                #         if segment[0] - (resulted_line[0]+resulted_line[2]) < 70:
                #             x = resulted_line[0]
                #             y = resulted_line[1]
                #             width = segment[0]+segment[2] - resulted_line[0]
                #             height = resulted_line[3]
                #             resulted_line = (x, y, width, height)
                # result_lines_points.append(resulted_line)

                first_point = line[0]
                last_point = line[-1]
                line_width = last_point[0] - first_point[0] + last_point[2]
                result_lines_points.append((first_point[0], first_point[1], line_width, first_point[3]))
        return result_lines_points

    @staticmethod
    def _denoise_wb_img(img, noise_width=3, noise_height=3):
        contours = cv2.findContours(cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]  # CHAIN_APPROX_NONE
        contours = [cv2.boundingRect(cnt) for cnt in contours]
        contours = [cnt for cnt in contours if cnt[2] < noise_width and cnt[3] < noise_height]
        print("Noise contours", contours)
        denoised_img = img.copy()
        for cnt in contours:
            x, y, w, h = cnt
            denoised_img = cv2.rectangle(denoised_img, (x, y), (x+w, y+h), (255, 255, 255), -1)
        return denoised_img

    @staticmethod
    def _find_missed_contours(contours):
        updated_contours = []
        for i, cnt in enumerate(contours):
            if len(updated_contours) == 8:
                break
            if i == 0:
                if cnt[0] >= 10:
                    updated_contours.append((7, 0, 15, 23))  # average symbol contour data
                updated_contours.append(cnt)
            else:
                last_symbol_cnt = updated_contours[-1]
                if cnt[0] - last_symbol_cnt[0] > 25:
                    # (x, y, w, h), not_found_cnt, (x, y, w, h)
                    contours_distance = cnt[0] - last_symbol_cnt[0] - last_symbol_cnt[2]  # distance between two found contours
                    new_cnt_width = contours_distance - 2 - 2  # margin from perv and next contours

                    new_cnt_x = last_symbol_cnt[0] + last_symbol_cnt[2] + 2
                    updated_contours.append(
                        (new_cnt_x, last_symbol_cnt[1], new_cnt_width, last_symbol_cnt[3])
                    )

                updated_contours.append(cnt)

        if len(updated_contours) < 8:
            last_symbol_cnt = updated_contours[-1]
            updated_contours.append(
                (last_symbol_cnt[0] + 2, last_symbol_cnt[1], last_symbol_cnt[2], last_symbol_cnt[3])
            )
        return updated_contours

    @staticmethod
    def _order_contours_in_lines(contours):
        """ Sort contours by OY, and create two-dimensional array """
        resulted_lines = []
        lines_count = 0
        prev_cnt_oy = 0
        contours.sort(key=lambda cnt: cnt[1])
        for i, cnt in enumerate(contours):
            if i == 0:
                resulted_lines.append([cnt])
                prev_cnt_oy = cnt[1]
                lines_count += 1
            else:
                if cnt[1] - prev_cnt_oy > 10:
                    resulted_lines.append([cnt])
                    prev_cnt_oy = cnt[1]
                    lines_count += 1
                else:
                    resulted_lines[lines_count-1].append(cnt)
        return resulted_lines  # [[(x, y, w, h), (x, y, w, h)], [...], [...]]

    def _find_table_frame_lines(self, img):  # for lotto tables(finding regular and strong sections)
        height, width = img.shape[:2]
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1))
        h_dilated = cv2.dilate(img, h_kernel)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (71, 1))
        closing = cv2.morphologyEx(h_dilated, cv2.MORPH_OPEN, k)

        inverted_line = cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY_INV)[1]

        contours = cv2.findContours(inverted_line, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [cv2.boundingRect(cnt) for cnt in contours]
        contours = self.merge_lines_contours(contours)
        print("Lotto table lines: ", contours)
        # Remove lines with small OY = [0, 30] or with big OY = [height-30, height]
        contours = list(filter(lambda cnt: cnt[1] not in range(0, 30) and cnt[1] not in range(height-30, height+1), contours))
        # sort by width, if there are few lines(image cropped with main lines), then needed line will smallest
        contours.sort(key=lambda cnt: cnt[2])
        h_line = contours[0]

        # IMAGE SHOULD BE CROPPED BY RIGHT AND LEFT SIDES, FOR CORRECT FINDING VERTICAL LINE
        img = img[
            0:height,
            h_line[0]:h_line[0]+h_line[2]  # crop by width of horizontal line
        ]
        print("All horizontal lines:", contours)
        print("Horizontal line contour data:", h_line)
        self.save_pic_debug(img, f"table/cropped_sides_by_horizontal_line.jpg")

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
        v_dilated = cv2.dilate(img, v_kernel)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 71))
        v_dilated = cv2.dilate(cv2.threshold(v_dilated, 127, 255, cv2.THRESH_BINARY_INV)[1], k)
        contours = cv2.findContours(v_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [cv2.boundingRect(cnt) for cnt in contours]
        contours.sort(key=lambda cnt: cnt[3], reverse=True)  # here vertical line will the biggest by height
        v_line = contours[0]

        self.save_pic_debug(inverted_line, f"table/h_line_dilated.jpg")
        self.save_pic_debug(v_dilated, f"table/v_line_dilated.jpg")

        regular_img = img[
            h_line[1]:v_line[1] + v_line[3],  # from y horizontal line to last point of vertical line
            h_line[0]-20:v_line[0]  # from x of horizontal line to x of vertical_line
        ]
        strong_img = img[
            h_line[1]:v_line[1] + v_line[3],  # from y horizontal line to last point of vertical line
            v_line[0]:h_line[0] + h_line[2]  # from x of horizontal line to x of vertical_line
        ]
        self.save_pic_debug(regular_img, f"table/regular_img.jpg")
        self.save_pic_debug(strong_img, f"table/strong_img.jpg")

        return regular_img, strong_img

    @staticmethod
    def remove_digit_white_space(digit_img):
        img = digit_img.copy()
        h, w = img.shape[:2]
        contours = cv2.findContours(cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)
        info_of_contours.sort(key=lambda cnt: cnt[0])  # by OX
        # print("DIGIT contours", info_of_contours)
        start_x = info_of_contours[0][0]  # first contour OX

        # Here we should take as end OX contour with the biggest sum(OX and width)
        info_of_contours.sort(key=lambda cnt: cnt[0]+cnt[2])
        end_x = info_of_contours[-1][0]+info_of_contours[-1][2]  # last contour OX + last contour width
        return img[
            0:h,
            start_x:end_x
        ]

    @staticmethod
    def get_unreaded_number_position(contours_lines):
        # TODO: do for strong and systematic too
        incomplete_lines = list(filter(lambda line: len(line) < 6, contours_lines))
        ideal_line = list(filter(lambda line: len(line) == 6, contours_lines))[0]

        if len(incomplete_lines) == 0:
            return -1
        else:
            line = incomplete_lines[0]
            if len(line) == 5:
                prev_number_contour = 0
                for i, number_contour in enumerate(line):
                    if i == 0:
                        prev_number_contour = number_contour
                    else:
                        print(prev_number_contour)
                        print(number_contour)
                        print(number_contour[0] - (prev_number_contour[0]+prev_number_contour[2]))
                        if number_contour[0] - (prev_number_contour[0]+prev_number_contour[2]) > 50:
                            return i+1  # because i starts from 0
                        prev_number_contour = number_contour
                # First or last(6) element
                # take incomplete line first number
                # and then compare it x1 with x2 of ideal line first number, if abs more than 10,
                # then first number contour was not found else it was last
                incomplete_line_first_contour = line[0]
                ideal_line_first_contour = ideal_line[0]
                if abs(incomplete_line_first_contour[0] - ideal_line_first_contour[0]) > 10:
                    return 1
                else:
                    return 6

            else:
                return -1

    @staticmethod
    def add_missed_table_number_like_pattern(contours_lines, position):
        for i, line in enumerate(contours_lines):
            if len(line) == 5:
                if position == 1:
                    contours_lines[i] = ["*", line]
                elif position == 6:
                    contours_lines[i] = [line, "*"]
                else:
                    contours_lines[i] = [*line[0:position-1], "*", *line[position-1:]]
            elif len(line) == 6:
                if position == 1:
                    contours_lines[i] = ["*", line[1:]]
                elif position == 6:
                    contours_lines[i] = [line[:-1], "*"]
                else:
                    contours_lines[i] = [*line[0:position-1], "*", *line[position:]]
        return contours_lines

    def data_incorrect_parsed_log(self, stringed_error, data_key):
        print(stringed_error)
        self.job_log(stringed_error)
        self.is_incorrect_check_info.update({data_key: True})

    @step_decorator("get_coords_of_main_lines")
    def get_coords_of_main_lines(self):
        try:
            blured_img_lines = cv2.GaussianBlur(self.wb_blured_img, [5, 5], 0)  # self.img_grayscale
            img = cv2.threshold(blured_img_lines, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            self.save_pic_debug(img, f"wb_inverted/wb_inverted.jpg")

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            img_dilation = cv2.dilate(img, kernel, iterations=2)
            detect_horizontal = cv2.morphologyEx(img_dilation, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            self.save_pic_debug(detect_horizontal, f"detect_lines/detect_lines.jpg")

            contours = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours = [cv2.boundingRect(cnt) for cnt in contours]
            lines_contours = self.merge_lines_contours(contours)
            print(lines_contours)

            # We need only two(three or four) lines around cards/table
            # So this two lines have the most points count(width)
            line_threshold = self.img_width - 100  # if line width >= threshold then we take this line
            lines = []

            # Remove lines which OY coord in range from 0 to 1200 and from img.height to img.height-500
            # Because card/table lines not in those diapason
            # and if x2 - x1 >= threshold
            for line in lines_contours:
                min_x = line[0]
                max_x = line[0]+line[2]
                y = line[1]
                condition = max_x - min_x >= line_threshold and y not in range(0, 1200) and y not in range(
                    self.img_height - 500, self.img_height)
                if condition:
                    lines.append({"min_x": min_x, "max_x": max_x, "y": y})
            print("MAIN LINES OF CHECK:", lines)
            if len(lines) == 0:
                self.data_incorrect_parsed_log("Main content lines were not found", "main_content_lines")
                raise Exception()
            elif len(lines) == 1:
                self.data_incorrect_parsed_log("Not all main content lines were found", "main_content_lines")
                raise Exception()
            else:
                # Here in lines can be more than four lines now
                # But the four needed ones are the longest
                lines = sorted(lines, key=lambda line: line["max_x"] - line["min_x"], reverse=True)
                lines = lines[:4]
                lines.sort(key=lambda line_data: line_data["y"])
                if len(lines) == 2:  # 123, 777, chance
                    self.longest_lines = {
                        "top_border_line": {},
                        "top_line": lines[0],
                        "middle_line": {},
                        "bottom_line": lines[1]
                    }
                elif len(lines) == 3:  # For lotto, if three main lines
                    self.longest_lines = {
                        "top_border_line": {},
                        "top_line": lines[0],
                        "middle_line": lines[1],
                        "bottom_line": lines[2]
                    }
                else:  # For lotto, if four main lines
                    self.longest_lines = {
                        "top_border_line": lines[0],
                        "top_line": lines[1],
                        "middle_line": lines[2],
                        "bottom_line": lines[3]
                    }
        except:
            self.all_data_not_found = True
            self.data_incorrect_parsed_log("Main content lines were not found, error occurred", "main_content_lines")
            self.StepStatus = "FAIL"
        return self.longest_lines

    @step_decorator("get_main_data_contours")
    def get_main_data_contours(self):
        try:
            def split_contours_by_width(contours):
                index_of_first_small_contour = 0
                for index, contour in enumerate(contours):
                    if contour[2] < 330:
                        index_of_first_small_contour = index
                        break

                large_contours = contours[:index_of_first_small_contour]
                small_contours = contours[index_of_first_small_contour::]

                return large_contours, small_contours

            # For remove no needed gray contours(garbage)
            blured_img = cv2.bilateralFilter(self.wb_blured_img, 1, 75, 75)
            wb_img = cv2.threshold(blured_img, 140, 255, cv2.THRESH_BINARY)[1]
            bottom_line = self.longest_lines["bottom_line"]
            check_bottom_part = wb_img[
                bottom_line["y"]:self.img_height,
                0:self.img_width
            ]

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
            img = cv2.morphologyEx(check_bottom_part, cv2.MORPH_OPEN, kernel)

            self.save_pic_debug(img, f"main_data_contours/main_data_blocks.png")
            contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours = [cv2.boundingRect(cnt) for cnt in contours]

            contours = [cnt for cnt in contours if cnt[3] > 20 and cnt[3] < 45]
            contours.sort(key=lambda cnt: cnt[2], reverse=True)  # by width
            # large     small
            l_contours, s_contours = split_contours_by_width(contours)
            # sort by OY
            l_contours.sort(key=lambda cnt: cnt[1])
            game_id_date_spaced_dashed_numbers = l_contours[-4:]
            date_line = game_id_date_spaced_dashed_numbers[1]

            s_contours.sort(key=lambda cnt: cnt[1])
            # Remove game_id contours if exists few game id in check
            s_contours = list(filter(lambda cnt: cnt[1] > date_line[1], s_contours))

            # In s_contours can be few no needed contours found from barcode + foreground bold inscription
            # OY of this contours will more than needed sum contours by ~200px

            prev_contour_y = 0
            for index, cnt in enumerate(s_contours):
                contour_y = cnt[1]
                if index == 0:
                    prev_contour_y = contour_y
                else:
                    if contour_y - prev_contour_y > 60:
                        del s_contours[index:]  # remove current and all next elements
                        break
                    else:
                        prev_contour_y = contour_y

            self.job_log(f"MAIN DATA CONTOURS. LARGE: {l_contours}. SMALL: {s_contours}")
            check_sum_line = s_contours[-2:]
            check_sum_line.sort(key=lambda cnt: cnt[0])  # by OX

            if len(game_id_date_spaced_dashed_numbers) == 4:
                game_id_line, date_line, *spaced_dashed_numbers = game_id_date_spaced_dashed_numbers
            else:
                # can be three
                game_id_line, date_line, dashed_numbers = game_id_date_spaced_dashed_numbers
                sum_contour = check_sum_line[0]
                spaced_number_y = sum_contour[1] + sum_contour[3] + 73  # sum_y + sum_h + 73(fixed distance)
                spaced_dashed_numbers = [(150, spaced_number_y, 700, 27), dashed_numbers]

            game_id_line = (
                game_id_line[0],
                game_id_line[1],
                # if few game_ids then width will smaller
                game_id_line[2] - 327 if game_id_line[2] > 500 else game_id_line[2] - 175,
                game_id_line[3]
            )
            date_line = (
                date_line[0],
                date_line[1],
                date_line[2] - 167,
                date_line[3]
            )
            self.main_data_contours["game_ids_count"] = len(l_contours[:-3])
            self.main_data_contours["game_id"] = game_id_line
            self.main_data_contours["date"] = date_line
            self.main_data_contours["sum"] = check_sum_line[0]
            self.main_data_contours["spaced_number"] = spaced_dashed_numbers[0]
            self.main_data_contours["dashed_number"] = spaced_dashed_numbers[1]
            self.job_log(f"END of method 'get_main_data_contours()'")
        except:
            self.StepStatus = "FAIL"

    @step_decorator("get_game_type")
    def get_game_type(self):
        try:
            game_type_img = self.wb_blured_img[
                self.qr_code_info["bottom_left"][1] + 310:self.qr_code_info["bottom_left"][1] + 310 + 300,
                self.qr_code_info["top_left"][0] - 158:self.qr_code_info["top_right"][0] + 158
            ]
            self.save_pic_debug(game_type_img, f"game_type/game_type_zone.jpg")
            edges = cv2.Canny(game_type_img, 10, 200)
            contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Contours to list of tuples
            info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)
            # Sort contours in right way
            sorted_contours = sorted(info_of_contours, key=lambda contour: contour[0])
            # Remove garbage contours
            sorted_contours = [contour for contour in sorted_contours if
                               contour[2] >= 30 and contour[3] >= 70]
            unique_contours = self._unique_contours(sorted_contours)
            first_contour = unique_contours[0]
            last_contour = unique_contours[-1]
            exactly_game_type_img = game_type_img[
                last_contour[1]: first_contour[1] + first_contour[3],
                first_contour[0]:last_contour[0] + last_contour[2]
            ]
            self.save_pic_debug(exactly_game_type_img, f"game_type/game_type.jpg")
            self.check_info["game_type"] = self.get_value_from_image(exactly_game_type_img, "game_type")
            if self.check_info["game_type"] == "":
                self.data_incorrect_parsed_log(f"Check game type is not found(get_game_type)", "game_type")
        except:
            self.data_incorrect_parsed_log(f"Check game type is not found, error occurred(get_game_type)", "game_type")
            self.StepStatus = "FAIL"
        return self.check_info["game_type"]

    @step_decorator("get_game_subtype")
    def get_game_subtype(self):
        try:
            img = cv2.imread(os.path.join(self.template_images_dir, "no_needed_repeat_game.png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            game_type = self.check_info["game_type"]
            distance_line = self.games_elements_distance[game_type]["game_subtype"]
            distance, line = distance_line[0], self.longest_lines[distance_line[1]]

            crop_img = self.wb_blured_img[
                line["y"] - distance:line["y"],
                0:self.img_width
            ]

            res = cv2.matchTemplate(crop_img, img, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.7)  # THRESHOLD
            if len(loc[0].tolist()) != 0:
                crop_img = self.wb_blured_img[
                    line["y"] - 115:line["y"] - 50,
                    self.qr_code_info["top_left"][0] - 158:self.qr_code_info["top_right"][0] + 158
                ]
                if self.check_info["game_type"] == "lotto":
                    self.bottom_oy_border_for_table = self.longest_lines["middle_line"]["y"] - 55
                else:
                    self.bottom_oy_border_for_table = self.longest_lines["bottom_line"]["y"] - 55

            self.save_pic_debug(crop_img, f"subtype/subtype.jpg")
            parsed_subtype = self.get_value_from_image(crop_img, "game_subtype")
            if self.longest_lines["top_border_line"] != {}:
                self.check_info["is_double"] = True
            self.check_info["game_subtype"] = parsed_subtype
            if parsed_subtype == "":
                self.data_incorrect_parsed_log("Game subtype was not found(get_game_subtype)", "game_subtype")
        except Exception as ex:
            self.data_incorrect_parsed_log(f"Game subtype was not found, error occurred(get_game_subtype): {str(ex)}", "game_subtype")
            self.StepStatus = "FAIL"
        return self.check_info["game_subtype"]

    @step_decorator("get_spent_money")
    def get_spent_money(self):
        try:
            bottom_line = self.longest_lines["bottom_line"]
            sum_contour = self.main_data_contours["sum"]  # (x, y, w, h)
            crop_img = self.wb_blured_img[
                bottom_line["y"] + sum_contour[1]:bottom_line["y"] + sum_contour[1] + sum_contour[3],
                sum_contour[0]:sum_contour[0] + sum_contour[2]
            ]

            self.save_pic_debug(crop_img, f"spent_money/spent_money.jpg")
            parsed_numbers = self.get_value_from_image(crop_img, "sum")

            if parsed_numbers:
                numbers = ""
                for symbol in parsed_numbers:
                    if symbol.isnumeric():
                        numbers += symbol
                numbers = numbers[0:-2] + "." + numbers[-2:]
                self.check_info["spent_on_ticket"] = float(numbers[0:len(numbers)])
            else:
                self.data_incorrect_parsed_log("Check sum was not found(get_spent_money)", "spent_on_ticket")
                self.check_info["spent_on_ticket"] = 0
        except:
            self.data_incorrect_parsed_log("Check sum was not found, error occurred(get_spent_money)", "spent_on_ticket")
            self.StepStatus = "FAIL"
        return self.check_info["spent_on_ticket"]

    @step_decorator("get_dashed_number")
    def get_dashed_number(self):
        try:
            decoded_objects = pyzbar.decode(self.img_original, symbols=[pyzbar.ZBarSymbol.I25])
            print("DASHED NUMBER CODE: ", decoded_objects)
            self.check_info["dashed_number"] = decoded_objects[0].data.decode("utf-8")[:-1]  # Read with no needed "0" at the end
        except Exception as ex:
            self.data_incorrect_parsed_log(
                f"The length of the number is not correct, error occurred(dashed_number): {ex}",
                "dashed_number"
            )
            self.StepStatus = "FAIL"
        return self.check_info["dashed_number"]

    @step_decorator("get_spaced_number")
    def get_spaced_number(self):
        try:
            bottom_line = self.longest_lines["bottom_line"]
            spaced_number_contour = self.main_data_contours["spaced_number"]  # (x, y, w, h)
            crop_img = self.wb_blured_img[
                bottom_line["y"] + spaced_number_contour[1]:bottom_line["y"] + spaced_number_contour[1] + spaced_number_contour[3],
                spaced_number_contour[0]:spaced_number_contour[0] + spaced_number_contour[2]
            ]
            self.save_pic_debug(crop_img, f"spaced_number/spaced_number.jpg")
            numbers = self.get_value_from_image(crop_img, "numbers", parse_just_in_numbers=False)
            if len(numbers) == 26:
                spaced_number = f"{numbers[:8]} {numbers[8:17]} {numbers[17:]}"
                self.check_info["spaced_number"] = spaced_number
            else:
                self.data_incorrect_parsed_log(
                    f"The length of the number is not correct(spaced_number)\nNumber: {numbers}",
                    "spaced_number"
                )
                self.check_info["spaced_number"] = numbers
        except:
            self.data_incorrect_parsed_log(
                f"The length of the number is not correct, error occurred(spaced_number)",
                "spaced_number"
            )
            self.StepStatus = "FAIL"
        return self.check_info["spaced_number"]

    @step_decorator("get_cards")
    def get_cards(self):
        try:
            top_line = self.longest_lines["top_line"]
            bottom_line = self.longest_lines["bottom_line"]

            cards_img = self.wb_blured_img[
                top_line["y"]:bottom_line["y"],
                0:self.img_width
            ]

            self.save_pic_debug(cards_img, f"cards/cards.jpg")
            # edges = cv2.Canny(cards_img, 10, 200)
            # contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            closing = cv2.morphologyEx(cards_img, cv2.MORPH_OPEN, k)
            self.save_pic_debug(closing, f"cards/closing_cards.jpg")
            img_dilated = cv2.dilate(cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY_INV)[1], kernel)
            self.save_pic_debug(img_dilated, f"cards/dilated_cards.jpg")

            contours = cv2.findContours(img_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            # Contours to list of tuples
            info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)
            # Sort contours in right way
            sorted_contours = sorted(info_of_contours, key=lambda contour: contour[0])
            # Remove garbage contours
            sorted_contours = [contour for contour in sorted_contours if
                               contour[2] >= 100 and contour[2] <= 200 and contour[3] >= 150 and contour[3] <= 230]

            # For four card suits(spades, hearts, ...)
            unique_contours = []
            ox_coord_range_suit = {
                "0": range(280),  # spades
                "1": range(280, 455),  # hearts
                "2": range(455, 645),  # diamonds
                "3": range(645, self.img_width)  # clubs
            }
            # Here we split cards contours by suits, sort by OY coord and remove duplicates
            for i in range(4):
                one_suit_contours = filter(lambda contour: contour[0] in ox_coord_range_suit[str(i)], sorted_contours)
                one_suit_contours = sorted(one_suit_contours, key=lambda contour: contour[1])  # sort by OY
                unique_suit_contours = self._unique_contours(one_suit_contours, data_type="cards")
                unique_contours.extend(unique_suit_contours)

            for i, contour in enumerate(unique_contours):
                one_card_img = cards_img[
                    contour[1]:contour[1] + contour[3],
                    contour[0]:contour[0] + contour[2],
                ]

                x_card_coord = contour[0]
                if x_card_coord in ox_coord_range_suit["0"]:
                    card_type = "spades"
                elif x_card_coord in ox_coord_range_suit["1"]:
                    card_type = "hearts"
                elif x_card_coord in ox_coord_range_suit["2"]:
                    card_type = "diamonds"
                else:
                    card_type = "clubs"
                self.save_pic_debug(one_card_img, f"cards/one_card_img_{i}.jpg")
                self.get_value_from_image(one_card_img, f"card_{card_type}")
        except Exception as ex:
            self.data_incorrect_parsed_log(
                f"Cards were not found, error occurred(get_cards): {ex}",
                "cards"
            )
            self.StepStatus = "FAIL"
        return self.check_info["cards"]

    @step_decorator("get_table_123_777")
    def get_table_123_777(self):
        try:
            game_type = self.check_info["game_type"]
            if self.bottom_oy_border_for_table == 0:
                bottom_oy_border = self.longest_lines["bottom_line"]["y"]
            else:
                # finding from top_line to top of subtype if exists
                bottom_oy_border = self.bottom_oy_border_for_table

            crop_img = self.wb_blured_img[
                self.longest_lines["top_line"]["y"]:bottom_oy_border,
                0:self.img_width
            ]
            if self.bottom_oy_border_for_table != 0:  # Add some space at the bottom if image was cropped
                crop_img = cv2.copyMakeBorder(
                    crop_img,
                    top=0,
                    bottom=10,
                    right=0,
                    left=0,
                    value=[255, 255, 255],
                    borderType=cv2.BORDER_CONSTANT
                )

            self.save_pic_debug(crop_img, f"table/table.jpg")
            crop_img_original = crop_img.copy()

            if game_type == "123":
                crop_img_inv = cv2.threshold(crop_img, 200, 255, cv2.THRESH_BINARY_INV)[1]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 9))
                dilated_img = cv2.dilate(crop_img_inv, kernel)
                self.save_pic_debug(dilated_img, f"table/dilated_table.jpg")

                contours = cv2.findContours(dilated_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
                info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)

                # Remove garbage contours
                min_width = 40
                max_width = 60
                min_height = 40

                sorted_contours = [contour for contour in info_of_contours if
                                   contour[2] >= min_width and contour[2] <= max_width and contour[3] >= min_height]
                print(info_of_contours, len(info_of_contours))
                print(sorted_contours, len(sorted_contours))
                if len(sorted_contours) % 3 != 0:
                    raise Exception("Not all numbers were found(table_123)")

                # sort contours in right way n11, n21, n31 n12, n22, n32
                sorted_by_OY = sorted(sorted_contours, key=lambda contour: contour[1])

                # group numbers by 3 and sort by OX
                groups_of_numbers = []
                i = 0
                while len(groups_of_numbers) != len(sorted_by_OY) / 3:
                    next_three_numbers = sorted_by_OY[i:i + 3]
                    next_three_numbers.sort(key=lambda contour: contour[0])  # sort by OX
                    groups_of_numbers.append(next_three_numbers)
                    i += 3

                for index, group in enumerate(groups_of_numbers):
                    self.check_info["table"][f"line_{index + 1}"] = {
                        "regular": []
                    }
                    for j, number in enumerate(group):
                        crop_number = crop_img_original[
                            number[1]:number[1] + number[3],
                            number[0]:number[0] + number[2]
                        ]
                        crop_number = self.remove_digit_white_space(crop_number)
                        self.save_pic_debug(crop_number, f"table/cropped_number{j}, {index}.jpg")
                        table_number = self.get_value_from_image(crop_number, "table")
                        self.check_info["table"][f"line_{index + 1}"]["regular"].append(int(table_number))

            elif game_type == "777":
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1))
                closing = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, k)
                self.save_pic_debug(closing, f"table/closing_img.jpg")

                img_dilated = cv2.dilate(cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY_INV)[1], kernel)

                contours = cv2.findContours(img_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
                info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)
                self.save_pic_debug(img_dilated, f"table/dilated_table.jpg")

                game_subtype = self.check_info["game_subtype"]
                count_nums_by_sybtype = {
                    "777_regular": 7,
                    "777_col8": 8,
                    "777_col9": 9
                }
                if game_subtype == "777_regular":
                    min_width = 60
                    max_width = 75
                    min_height = 35
                    max_height = 50
                elif game_subtype == "777_col8":
                    min_width = 55
                    max_width = 70
                    min_height = 34
                    max_height = 50
                else:  # 777_col9
                    min_width = 47
                    max_width = 60
                    min_height = 35
                    max_height = 45

                sorted_contours = [contour for contour in info_of_contours if
                                   contour[2] >= min_width and contour[2] <= max_width and contour[3] >= min_height and contour[3] <= max_height]
                print(info_of_contours, len(info_of_contours))
                print(sorted_contours, len(sorted_contours))

                numbers_in_lines = self._order_contours_in_lines(sorted_contours)

                for index, line in enumerate(numbers_in_lines):
                    line.sort(key=lambda number: number[0])
                    numbers_in_lines[index] = line

                for index, line in enumerate(numbers_in_lines):
                    for j, number in enumerate(line):
                        # At the end of every line exist number of that line. And its contour can be in "numbers_in_lines"
                        # So we should break loop
                        if j >= count_nums_by_sybtype[game_subtype]:
                            break
                        x, y, w, h = number
                        OX_between_digits = w // 2
                        number_img = crop_img_original[
                            y:y+h,
                            x:x+w
                        ]
                        self.save_pic_debug(number_img, f"table/number({j}, {index}).jpg")

                        digit1_img = crop_img_original[
                            y:y+h,
                            x:x+OX_between_digits
                        ]
                        digit2_img = crop_img_original[
                            y:y + h,
                            x+OX_between_digits:x+w
                        ]
                        if game_subtype == "777_col8":
                            digit1_h, digit1_w = digit1_img.shape[:2]
                            digit1_img = cv2.resize(digit1_img, (digit1_w + 2, digit1_h))
                            digit2_h, digit2_w = digit2_img.shape[:2]
                            digit2_img = cv2.resize(digit2_img, (digit2_w + 2, digit2_h))
                        elif game_subtype == "777_col9":
                            digit1_h, digit1_w = digit1_img.shape[:2]
                            digit1_img = cv2.resize(digit1_img, (digit1_w + 5, digit1_h))
                            digit2_h, digit2_w = digit2_img.shape[:2]
                            digit2_img = cv2.resize(digit2_img, (digit2_w + 5, digit2_h))

                        digit1_img = cv2.threshold(digit1_img, 127, 255, cv2.THRESH_BINARY)[1]
                        digit2_img = cv2.threshold(digit2_img, 127, 255, cv2.THRESH_BINARY)[1]

                        digit1_img = self.remove_digit_white_space(digit1_img)
                        digit2_img = self.remove_digit_white_space(digit2_img)

                        self.save_pic_debug(digit1_img, f"table/digit1({j}, {index}).jpg")
                        self.save_pic_debug(digit2_img, f"table/digit2({j}, {index}).jpg")

                        digit1 = self.get_value_from_image(digit1_img, "table")
                        digit2 = self.get_value_from_image(digit2_img, "table")

                        if not self.check_info["table"].get(f"line_{index + 1}", False):  # If line does not exist - create
                            self.check_info["table"][f"line_{index + 1}"] = {
                                "regular": []
                            }
                        self.check_info["table"][f"line_{index + 1}"]["regular"].append(int(f"{digit1}{digit2}"))
                    count_found_line_nums = len(self.check_info["table"][f"line_{index + 1}"]["regular"])
                    count_needed_line_nums = count_nums_by_sybtype[game_subtype]
                    if count_found_line_nums != count_needed_line_nums:
                        print("Error. Not All numbers in line were found")
        except Exception as ex:
            self.data_incorrect_parsed_log(
                f"Table was not found, error occurred(get_table_123_777): {ex}",
                "table_123_777"
            )
            self.StepStatus = "FAIL"
        return self.check_info["table"]

    @step_decorator("get_table_lotto")
    def get_table_lotto(self):
        try:
            if self.bottom_oy_border_for_table == 0:
                bottom_oy_border = self.longest_lines["middle_line"]["y"]
            else:
                # finding from top_line to top of subtype if exists
                bottom_oy_border = self.bottom_oy_border_for_table

            crop_img = self.wb_blured_img[
                self.longest_lines["top_line"]["y"]:bottom_oy_border,  # 90px it is cropping without table header
                0:self.img_width
            ]
            crop_img = self._denoise_wb_img(crop_img)
            self.save_pic_debug(crop_img, f"table/table.jpg")

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            regular_closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 11))
            strong_closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

            regular_img, strong_img = self._find_table_frame_lines(crop_img)
            regular_img = cv2.copyMakeBorder(
                regular_img,
                top=0,
                bottom=10,
                right=0,
                left=20,
                value=(255, 255, 255),
                borderType=cv2.BORDER_CONSTANT
            )
            strong_img = cv2.copyMakeBorder(
                strong_img,
                top=0,
                bottom=10,
                right=0,
                left=0,
                value=(255, 255, 255),
                borderType=cv2.BORDER_CONSTANT
            )

            closed_regular = cv2.morphologyEx(regular_img, cv2.MORPH_OPEN, regular_closing_kernel)
            dilated_regular = cv2.dilate(
                cv2.threshold(closed_regular, 127, 255, cv2.THRESH_BINARY_INV)[1],
                kernel
            )
            self.save_pic_debug(closed_regular, f"table/closed_regular.jpg")
            self.save_pic_debug(dilated_regular, f"table/dilated_regular.jpg")

            closed_strong = cv2.morphologyEx(strong_img, cv2.MORPH_OPEN, strong_closing_kernel)
            dilated_strong = cv2.dilate(
                cv2.threshold(closed_strong, 127, 255, cv2.THRESH_BINARY_INV)[1],
                kernel
            )
            self.save_pic_debug(closed_strong, f"table/closed_strong.jpg")
            self.save_pic_debug(dilated_strong, f"table/dilated_strong.jpg")

            number_contours = cv2.findContours(dilated_regular, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            number_contours = [cv2.boundingRect(cnt) for cnt in number_contours]
            regular_number_contours = [contour for contour in number_contours if
                                       contour[2] >= 40 and contour[2] <= 60 and contour[3] >= 30 and contour[3] <= 40]

            number_contours = cv2.findContours(dilated_strong, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            number_contours = [cv2.boundingRect(cnt) for cnt in number_contours]
            strong_number_contours = [contour for contour in number_contours if
                                      contour[2] >= 20 and contour[2] <= 30 and contour[3] >= 30 and contour[3] <= 40]

            if self.check_info["game_subtype"] == "lotto_regular":
                regular_lines = self._order_contours_in_lines(regular_number_contours)
                for i, line in enumerate(regular_lines):
                    line.sort(key=lambda cnt: cnt[0])
                    regular_lines[i] = line
                strong_lines = self._order_contours_in_lines(strong_number_contours)

                position = self.get_unreaded_number_position(regular_lines)
                print("POSITION OF NOT FOUND TABLE NUMBER IS:", position)
                if position != -1:
                    self.check_info["is_table_pattern"] = True
                    regular_lines = self.add_missed_table_number_like_pattern(regular_lines, position)

                for i, line in enumerate(regular_lines):
                    self.check_info["table"][f"line_{i+1}"] = {
                        "regular": [],
                        "strong": [],
                    }
                    # Find regular numbers
                    for j, number in enumerate(line):
                        if number == "*":
                            self.check_info["table"][f"line_{i + 1}"]["regular"].append("*")
                            continue
                        x, y, w, h = number
                        digit1_img = regular_img[
                            y:y + h,
                            x:x + w // 2
                        ]
                        digit2_img = regular_img[
                            y:y + h,
                            x + w // 2:x + w
                        ]
                        self.save_pic_debug(digit1_img, f"table/digit1({i + 1}, {j}).jpg")
                        self.save_pic_debug(digit2_img, f"table/digit2({i + 1}, {j}).jpg")
                        digit1 = self.get_value_from_image(digit1_img, "table")
                        digit2 = self.get_value_from_image(digit2_img, "table")
                        self.check_info["table"][f"line_{i + 1}"]["regular"].append(int(f"{digit1}{digit2}"))
                    self.check_info["table"][f"line_{i + 1}"]["regular"] = self.check_info["table"][f"line_{i + 1}"]["regular"][::-1]

                    # Find strong number, for regular exists only one strong number for line
                    strong_line = strong_lines[i]
                    strong_line.sort(key=lambda cnt: cnt[0])
                    x, y, w, h = strong_line[0]  # first contour it is strong number
                    number_img = strong_img[
                        y:y + h,
                        x:x + w
                    ]
                    table_number = self.get_value_from_image(number_img, "table")
                    self.check_info["table"][f"line_{i + 1}"]["strong"].append(int(table_number))

            elif self.check_info["game_subtype"] in ["lotto_strong", "lotto_systematic"]:
                self.check_info["table"][f"line_1"] = {
                    "regular": [],
                    "strong": [],
                }
                for i, number in enumerate(regular_number_contours):
                    x, y, w, h = number
                    digit_width = w // 2

                    digit1_img = regular_img[
                        y:y + h,
                        x:x + digit_width
                    ]
                    digit2_img = regular_img[
                        y:y + h,
                        x + digit_width:x+w
                    ]
                    self.save_pic_debug(digit1_img, f"table/digit1({i}).jpg")
                    self.save_pic_debug(digit2_img, f"table/digit2({i}).jpg")
                    digit1 = self.get_value_from_image(digit1_img, "table")
                    digit2 = self.get_value_from_image(digit2_img, "table")
                    self.check_info["table"][f"line_1"]["regular"].append(int(f"{digit1}{digit2}"))
                self.check_info["table"][f"line_1"]["regular"].sort(key=lambda number: number)

                strong_lines = self._order_contours_in_lines(strong_number_contours)
                if self.check_info["game_subtype"] == "lotto_systematic":
                    # For lotto systematic we take first contour from first line(there is only one strong number)
                    x, y, w, h = strong_lines[0][0]
                    digit_img = strong_img[
                        y:y+h,
                        x:x+w
                    ]
                    self.save_pic_debug(digit_img, f"table/strong_digit({i}).jpg")
                    digit = self.get_value_from_image(digit_img, "table")
                    self.check_info["table"][f"line_1"]["strong"].append(int(digit))
                else:  # lotto strong
                    # Here can be from 4 to 7 strong numbers
                    # So first line always will have 3 needed numbers, second line can have 1, 2 or 3 nums
                    # And if there are 7 strong numbers were selected, then we will have third line with one number
                    for line in strong_lines:
                        line.sort(key=lambda cnt: cnt[0])
                        for i, number_contour in enumerate(line):
                            if i <= 2:
                                x, y, w, h = number_contour
                                digit_img = strong_img[
                                    y:y + h,
                                    x:x + w
                                ]
                                self.save_pic_debug(digit_img, f"table/strong_digit({i}).jpg")
                                digit = self.get_value_from_image(digit_img, "table")
                                self.check_info["table"][f"line_1"]["strong"].append(int(digit))

                        self.check_info["table"][f"line_1"]["strong"].sort(key=lambda number: number)
        except Exception as ex:
            self.data_incorrect_parsed_log(
                f"Table was not found, error occurred(get_table_lotto): {str(ex)}",
                "table_lotto"
            )
            self.StepStatus = "FAIL"
        return self.check_info["table"]

    @step_decorator("get_extra")
    def get_extra(self):
        try:
            middle_line = self.longest_lines["middle_line"]
            bottom_line = self.longest_lines["bottom_line"]

            crop_img = self.wb_blured_img[
                middle_line["y"]:bottom_line["y"],
                0:self.img_width
            ]
            self.save_pic_debug(crop_img, f"extra/extra.jpg")

            w, h = crop_img.shape[:2]
            extra_crop = crop_img[
                0:130,
                self.img_width // 2:self.img_width
            ]
            self.save_pic_debug(extra_crop, f"extra/is_extra.jpg")

            extra_number_crop = crop_img[
                140:h,
                0:self.img_width
            ]

            is_extra = self.get_value_from_image(extra_crop, "is_extra")
            self.check_info["extra"] = is_extra
            if is_extra:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 1))
                # Dilate image by OX, for make one solid block
                img_dilated = cv2.dilate(
                    cv2.threshold(extra_number_crop, 127, 255, cv2.THRESH_BINARY_INV)[1],
                    kernel
                )

                contours = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                contours = [cv2.boundingRect(cnt) for cnt in contours]
                # Find number block contour by min width and height
                number_block_contour = [cnt for cnt in contours if cnt[2] >= 150 and cnt[3] >= 25][0]

                self.save_pic_debug(img_dilated, f"extra/number_horizontal_dilated.jpg")

                # Crop only number
                x, y, w, h = number_block_contour
                extra_number_exactly = extra_number_crop[
                    y:y+h,
                    x+10:x+w-10
                ]
                self.save_pic_debug(extra_number_exactly, f"extra/extra_number.jpg")

                # -20 because after dilation number block bigger, divide by 6 because there are 6 digits
                digit_width = 23
                next_digit_OX = 3  # 3 from third pixel
                for i in range(6):
                    digit_img = extra_number_exactly[
                        0:h,
                        next_digit_OX:next_digit_OX+digit_width
                    ]
                    self.save_pic_debug(digit_img, f"extra/digit({i}).jpg")
                    print("Extra num found coords: ")
                    extra_number = self.get_value_from_image(digit_img, "extra_numbers")
                    self.check_info["extra_number"] += extra_number
                    next_digit_OX += digit_width + 7  # 7 it is space between numbers
        except Exception as ex:
            self.data_incorrect_parsed_log(f"Extra was not found, error occurred(get_extra): {str(ex)}", "extra")
            self.StepStatus = "FAIL"
        return self.check_info["extra"], self.check_info["extra_number"]

    @step_decorator("get_date")
    def get_date(self):
        try:
            bottom_line = self.longest_lines["bottom_line"]
            date_contour = self.main_data_contours["date"]  # (x, y, w, h)
            crop_img = self.wb_blured_img[
                bottom_line["y"] + date_contour[1]:bottom_line["y"] + date_contour[1] + date_contour[3],
                date_contour[0]:date_contour[0] + date_contour[2]
            ]

            self.save_pic_debug(crop_img, f"date/date.jpg")
            numbers = self.get_value_from_image(crop_img, "date")
            if len(numbers) == 12:
                date = f"{numbers[:2]}:{numbers[2:4]}:{numbers[4:6]} {numbers[6:8]}.{numbers[8:10]}.{numbers[10:]}"
                datetime.datetime.strptime(date, "%H:%M:%S %d.%m.%y")  # if there is no error then add datetime string
                self.check_info["date"] = date
            else:
                self.data_incorrect_parsed_log(f"The length of the number is not correct(date)\nNumber: {numbers}", "date")
                self.check_info["date"] = numbers
        except:
            self.data_incorrect_parsed_log(f"The length of the number is not correct, error occurred(date)", "date")
            self.StepStatus = "FAIL"
        return self.check_info["date"]

    @step_decorator("get_user_passport")
    def get_user_passport(self):
        # TODO: Find user passport code before qr code
        pass

    @step_decorator("get_game_id")
    def get_game_id(self):
        try:
            bottom_line = self.longest_lines["bottom_line"]
            game_id_contour = self.main_data_contours["game_id"]  # (x, y, w, h)
            crop_img = self.wb_blured_img[
                bottom_line["y"] + game_id_contour[1]:bottom_line["y"] + game_id_contour[1] + game_id_contour[3],
                game_id_contour[0]:game_id_contour[0] + game_id_contour[2]
            ]

            self.save_pic_debug(crop_img, f"game_id/game_id.jpg")
            numbers = self.get_value_from_image(crop_img, "game_id")
            self.check_info["game_id"] = numbers

            if self.check_info["game_id"] == "":
                self.data_incorrect_parsed_log(
                    f"Game id is incorrect(get_game_id)",
                    "game_id"
                )
        except Exception as ex:
            self.data_incorrect_parsed_log(
                f"Game id is not found, error occurred(get_game_id): {ex}",
                "game_id"
            )
            self.StepStatus = "FAIL"
        return self.check_info["game_id"]

    def get_value_from_image(self, cropped_img, data_type, parse_just_in_numbers=True):
        result = ""

        if data_type == "table":
            game_type = self.check_info["game_type"]
            if game_type == "123":
                img = cv2.imread(os.path.join(self.template_images_dir, "table_123_numbers.png"))
                d_table_numbers = d_table_123_numbers
            elif game_type == "777":
                img = cv2.imread(os.path.join(self.template_images_dir,"table_777_numbers.png"))
                d_table_numbers = d_table_777_numbers
            elif game_type == "lotto":
                img = cv2.imread(os.path.join(self.template_images_dir, "table_lotto_numbers.png"))
                d_table_numbers = d_table_lotto_numbers

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(img, cropped_img, cv2.TM_CCOEFF_NORMED)
            y, x = np.unravel_index(res.argmax(), res.shape)
            print(x, y)
            for key, value in d_table_numbers.items():
                if x in range(*value[0]) and y in range(*value[1]):
                    return key

        elif data_type == "extra_numbers":
            img = cv2.imread(os.path.join(self.template_images_dir, "extra_numbers.png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cropped_img = cv2.copyMakeBorder(
                cropped_img,
                top=1,
                bottom=1,
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )

            res = cv2.matchTemplate(img, cropped_img, cv2.TM_CCOEFF_NORMED)
            y, x = np.unravel_index(res.argmax(), res.shape)
            print(x, y)
            for key, value in d_extra_numbers.items():
                if x in range(*value[0]) and y in range(*value[1]):
                    return key

        elif data_type == "game_type":
            img = cv2.imread(os.path.join(self.template_images_dir, "game_types.png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(
                img,
                cropped_img,
                cv2.TM_CCOEFF_NORMED
            )
            y, x = np.unravel_index(res.argmax(), res.shape)
            print("GAME TYPE COORDS:", x, y)
            for key, value in game_types.items():
                if x in range(*value[0]) and y in range(*value[1]):
                    return key
            return "Not found"

        elif data_type == "game_subtype":
            game_type = self.check_info["game_type"]
            if game_type == "777":
                img = cv2.imread(os.path.join(self.template_images_dir, "777_systematic_subtype.png"))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                res = cv2.matchTemplate(cropped_img, img, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.7)  # THRESHOLD
                if len(loc[0].tolist()) == 0:
                    result = "777_regular"
                else:
                    if self.bottom_oy_border_for_table != 0:
                        self.bottom_oy_border_for_table -= 60  # if no needed data exists
                    else:
                        self.bottom_oy_border_for_table = self.longest_lines["bottom_line"]["y"] - 60
                    img = cv2.imread(os.path.join(self.template_images_dir, "777_systematic_subtype_col8.png"))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    res = cv2.matchTemplate(cropped_img, img, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= 0.7)  # THRESHOLD
                    if len(loc[0].tolist()) == 0:
                        result = "777_col9"
                    else:
                        result = "777_col8"

            elif game_type == "chance":
                img_type_in_subtype = cv2.imread(os.path.join(self.template_images_dir, "chance_subtype.png"))
                img = cv2.cvtColor(img_type_in_subtype, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(cropped_img, img, cv2.TM_CCOEFF_NORMED)
                y, x = np.unravel_index(res.argmax(), res.shape)
                # Crop exactly subtype
                cropped_subtype = cropped_img[
                    y:y+50,
                    x-180:x+101+70
                ]
                self.save_pic_debug(cropped_subtype, "subtype/subtype_cropped.png")

                img_subtype_multi = cv2.imread(os.path.join(self.template_images_dir, "chance_multi_subtype.png"))
                img_subtype_multi = cv2.cvtColor(img_subtype_multi, cv2.COLOR_BGR2GRAY)

                img_subtype_multi_not_full = cv2.imread(os.path.join(self.template_images_dir, "chance_multi_subtype_not_full.png"))
                img_subtype_multi_not_full = cv2.cvtColor(img_subtype_multi_not_full, cv2.COLOR_BGR2GRAY)

                img_subtype_systematic = cv2.imread(os.path.join(self.template_images_dir, "chance_systematic_subtype.png"))
                img_subtype_systematic = cv2.cvtColor(img_subtype_systematic, cv2.COLOR_BGR2GRAY)

                res_suffix_subtype = cv2.matchTemplate(cropped_subtype, img_subtype_multi, cv2.TM_CCOEFF_NORMED)
                loc_suffix = np.where(res_suffix_subtype >= 0.65)  # THRESHOLD

                res_suffix_subtype_not_full = cv2.matchTemplate(cropped_subtype, img_subtype_multi_not_full, cv2.TM_CCOEFF_NORMED)
                loc_suffix_not_full = np.where(res_suffix_subtype_not_full >= 0.65)  # THRESHOLD

                res_prefix_subtype = cv2.matchTemplate(cropped_subtype, img_subtype_systematic, cv2.TM_CCOEFF_NORMED)
                loc_prefix = np.where(res_prefix_subtype >= 0.65)  # THRESHOLD

                if len(loc_prefix[0].tolist()) != 0:  # systematic first because here can be systematic type_5
                    result = "chance_systematic"
                elif len(loc_suffix[0].tolist()) != 0 or len(loc_suffix_not_full[0].tolist()) != 0:
                    result = "chance_multi"
                else:
                    result = "chance_regular"

                if self.bottom_oy_border_for_table != 0:
                    self.bottom_oy_border_for_table -= 60  # if no needed data exists
                else:
                    self.bottom_oy_border_for_table = self.longest_lines["bottom_line"]["y"] - 60

            elif game_type == "123":
                if self.bottom_oy_border_for_table != 0:
                    self.bottom_oy_border_for_table -= 60  # if no needed data exists
                else:
                    self.bottom_oy_border_for_table = self.longest_lines["bottom_line"]["y"] - 60
                result = "123_regular"

            elif game_type == "lotto":
                lotto_strong_subtype = cv2.imread(os.path.join(self.template_images_dir, "lotto_strong_subtype.png"))
                lotto_strong_subtype = cv2.cvtColor(lotto_strong_subtype, cv2.COLOR_BGR2GRAY)

                lotto_systematic_subtype = cv2.imread(os.path.join(self.template_images_dir, "lotto_systematic_subtype.png"))
                lotto_systematic_subtype = cv2.cvtColor(lotto_systematic_subtype, cv2.COLOR_BGR2GRAY)

                res_strong_subtype = cv2.matchTemplate(cropped_img, lotto_strong_subtype, cv2.TM_CCOEFF_NORMED)
                loc_strong_subtype = np.where(res_strong_subtype >= 0.7)  # THRESHOLD

                res_systematic_subtype = cv2.matchTemplate(cropped_img, lotto_systematic_subtype, cv2.TM_CCOEFF_NORMED)
                loc_systematic_subtype = np.where(res_systematic_subtype >= 0.7)  # THRESHOLD

                if len(loc_strong_subtype[0].tolist()) != 0:
                    result = "lotto_strong"
                elif len(loc_systematic_subtype[0].tolist()) != 0:
                    result = "lotto_systematic"
                else:
                    result = "lotto_regular"

                if result != "lotto_regular":
                    if self.bottom_oy_border_for_table != 0:
                        self.bottom_oy_border_for_table -= 60  # if no needed data exists
                    else:
                        self.bottom_oy_border_for_table = self.longest_lines["middle_line"]["y"] - 60

            return result

        elif data_type in ["numbers", "date", "sum", "game_id"]:
            print("numbers", "date", "sum", "game_id")
            if data_type == "sum":
                height, width = cropped_img.shape[:2]
                find_symbol_img = cv2.imread(os.path.join(self.template_images_dir, "sum_symbol_border.png"))
                find_symbol_img = cv2.cvtColor(find_symbol_img, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(cropped_img, find_symbol_img, cv2.TM_CCOEFF_NORMED)
                y, x = np.unravel_index(res.argmax(), res.shape)

                cropped_img = cropped_img[
                    0:height,
                    x + 20:width
                ]

            symbols_numbers_img = cv2.imread(os.path.join(self.template_images_dir, "numbers_and_letters.png"))
            numbers_img = cv2.imread(os.path.join(self.template_images_dir, "numbers.png"))

            cropped_img = cv2.copyMakeBorder(
                cropped_img,
                top=1,
                bottom=1,
                left=30,  # this border need for img dilation
                right=10,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )

            # In sum, date image symbols are located far from each other(so we need make bigger kernel)
            width_of_kernel = 30 if data_type in ["sum", "date"] else 15
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width_of_kernel, 3))
            dilation = cv2.dilate(255 - cropped_img, rect_kernel, iterations=1)

            self.save_pic_debug(
                dilation,
                f"blocks_of_needed_data/data_block({data_type}, {parse_just_in_numbers}).png"
            )

            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            contours = list(map(lambda cnt: cv2.boundingRect(cnt), contours))
            contours.sort(key=lambda contour: contour[0])

            for index, cnt in enumerate(contours):
                block_x, block_y, block_w, block_h = cnt
                min_block_width = 70 if data_type == "sum" else 100
                if block_w < min_block_width:
                    continue

                block = cropped_img[
                    block_y:block_y + block_h,
                    block_x:block_x + block_w
                ]

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
                img_dilated = cv2.dilate(cv2.threshold(block, 127, 255, cv2.THRESH_BINARY_INV)[1], kernel)

                if not parse_just_in_numbers and data_type == "numbers":
                    self.save_pic_debug(
                        img_dilated,
                        f"symbols_blocks_in_data/spaced_number(block coord{block_x}).png"
                    )
                else:
                    self.save_pic_debug(
                        img_dilated,
                        f"symbols_blocks_in_data/{data_type}(block coord{block_x}).png"
                    )

                lines_contours = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                lines_contours = [cv2.boundingRect(cnt) for cnt in lines_contours]

                lines_contours = list(filter(lambda cnt: cnt[2] > 6, lines_contours))

                def minimal_black_pixels_count(contour_tuple):
                    x, y, w, h = contour_tuple
                    cropped_contour = block[
                        y:y + h,
                        x:x + w
                    ]
                    symbol_width, symbol_height = cropped_contour.shape[:2]
                    count_non_zero = cv2.countNonZero(cropped_contour)

                    # Find rects in which count of black pixels less than...
                    minimal_count_black_pixels = 60
                    if symbol_width * symbol_height - count_non_zero < minimal_count_black_pixels:
                        return False
                    return True

                if parse_just_in_numbers:
                    lines_contours = list(filter(minimal_black_pixels_count, lines_contours))

                # In blocks of spaced number after dilation some can symbols can be in one contour
                if data_type == "numbers":
                    if not parse_just_in_numbers:
                        block_symbols_count = 8 if index == 0 else 9
                    else:
                        # dashed number
                        block_symbols_count = 19  # {"0": 4, "1": 9, "2": 6}[str(index)]  # OR just 19 ?

                    if len(lines_contours) < block_symbols_count:
                        for cnt in lines_contours:
                            if len(lines_contours) >= block_symbols_count:
                                break
                            x, y, w, h = cnt
                            if w > 20:
                                # +1 because we have this contour
                                count_needed_symbols = block_symbols_count - len(lines_contours) + 1
                                # print("DEBUG COUNT NEEDED SYMBOLS:", count_needed_symbols)
                                average_symbol_width = w // count_needed_symbols
                                OX_first_merged_symbol = x
                                for i in range(count_needed_symbols):
                                    lines_contours.append((OX_first_merged_symbol, y, average_symbol_width, h))
                                    OX_first_merged_symbol += average_symbol_width
                        lines_contours = list(filter(lambda cnt: cnt[2] < 23, lines_contours))

                lines_contours.sort(key=lambda cnt: cnt[0])
                lines_contours = self._unique_contours(lines_contours)

                # If in first block of spaced number less than 8 symbol contours. Then find it by hands in loop
                if data_type == "numbers" and not parse_just_in_numbers and index == 0 and len(lines_contours) < 8:
                    lines_contours = self._find_missed_contours(lines_contours)

                for cnt in lines_contours:
                    symbol_x, symbol_y, symbol_w, symbol_h = cnt
                    cropped_contour = block[
                        symbol_y:symbol_y + symbol_h,
                        symbol_x:symbol_x + symbol_w
                    ]

                    cropped_contour = cv2.copyMakeBorder(
                        cropped_contour,
                        top=2,
                        bottom=2,
                        left=2,
                        right=2,
                        borderType=cv2.BORDER_CONSTANT,
                        value=[255, 255, 255]
                    )
                    if not parse_just_in_numbers and data_type == "numbers" and index == 0:
                        img = symbols_numbers_img
                    else:
                        img = numbers_img
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    res = cv2.matchTemplate(img, cropped_contour, cv2.TM_CCOEFF_NORMED)
                    found_symbol_y, found_symbol_x = np.unravel_index(res.argmax(), res.shape)
                    for key, value in d_all_symbols.items():
                        if found_symbol_x in range(*value[0]) and found_symbol_y in range(*value[1]):
                            if key != "(" and key != ")":
                                result += key

            print("result = ", result)
            return result

        elif data_type == "is_extra":
            img = cv2.imread(os.path.join(self.template_images_dir, "extra_true.png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(cropped_img, img, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.7)  # THRESHOLD
            if len(loc[0].tolist()) != 0:
                return True
            return False

        elif data_type.startswith("card"):
            card_type = data_type.split("_")[1]
            img = cv2.imread(os.path.join(self.template_images_dir, f"cards_{card_type}.png"), 0)
            img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]

            resized_card = self.resize_img(cropped_img, target_width=128)
            resized_card = cv2.threshold(resized_card, 200, 255, cv2.THRESH_BINARY)[1]
            h, w = resized_card.shape[:2]

            # There is card with white pixels by sides. We should crop them
            resized_card = resized_card[
                20:h-20,
                20:w-20
            ]
            self.save_pic_debug(resized_card, f"cards/resized_card_img_{random.randint(0, 10_000)}.jpg")

            res = cv2.matchTemplate(img, resized_card, cv2.TM_CCOEFF_NORMED)
            y, x = np.unravel_index(res.argmax(), res.shape)
            print(x, y)
            cards = {
                "spades": cards_spades,
                "hearts": cards_hearts,
                "diamonds": cards_diamonds,
                "clubs": cards_clubs
            }.get(card_type)

            for key, value in cards.items():
                if x in range(*value[0]) and y in range(*value[1]):
                    card = key
                    self.check_info["cards"][card_type].append(card)
                    return card

    def get_result(self):
        # All should be called in this way
        self.set_param()
        self.img_prep()

        if self.img_is_valid:
            print("Image is valid.")
            self.get_coords_of_main_lines()
            self.job_log(f"Main check lines: {self.longest_lines}")

            game_type = self.get_game_type()
            self.get_game_subtype()
            self.get_main_data_contours()
            self.get_game_id()
            self.get_date()
            self.get_spent_money()
            self.get_dashed_number()
            self.get_spaced_number()

            if game_type == "chance":
                self.get_cards()
            elif game_type == "lotto":
                self.get_table_lotto()
                self.get_extra()
            else:
                self.get_table_123_777()
        else:
            print("Image is not valid. Data was not parsed.")
            self.all_data_not_found = True
        return self.check_info, self.all_data_not_found, self.is_incorrect_check_info, self.StepLogFull


# TODO: find "autofilling" data near subtype

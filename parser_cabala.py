#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import logging
import logging.config
import datetime

import cv2
import cv2.cuda
import numpy as np
from pyzbar import pyzbar

from elements_coords_in_img import (game_types,
                                    d_all_symbols,
                                    d_table_123_numbers,
                                    d_table_777_numbers,
                                    d_table_lotto_numbers,
                                    cards_clubs, cards_diamonds, cards_hearts, cards_spades,
                                    d_extra_numbers)


class ParserChecks:
    def __init__(self, img_path, log_dir="", debug_img_dir=""):
        self.img_path = img_path  # name file
        self.img_filename = os.path.basename(img_path).split(".")[0].strip()
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
        self.job_log(f'***** START job {img_path} ******')
        #

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

    def set_param(self):
        # Distance from lines to needed elements
        self.games_elements_distance = {
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
            "game_id": "",
            "spent_on_ticket": 0.0,
            "dashed_number": "",
            "spaced_number": "",
            "extra": False,
            "extra_number": "",
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

    def step_start(self):
        self.StepTimeStart = datetime.datetime.now()
        self.StepNum += 1

    def step_stop(self, l_name_step=""):
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
        decoded_objects = pyzbar.decode(self.img_original, symbols=[pyzbar.ZBarSymbol.QRCODE])
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
    def resize_img(img):
        height, width = img.shape[:2]
        if width > height:
            # If image inverted
            target_height = 972
            ratio = target_height / height
            target_width = int(width * ratio)
        else:
            # If image OK
            target_width = 972
            ratio = target_width / width
            target_height = int(height * ratio)

        resized_image = cv2.resize(img, (target_width, target_height))
        return resized_image

    def img_prep(self):
        try:
            self.StepLogAdd = f'  file = {self.img_path}'
            self.img_original = cv2.imread(self.img_path)
            self.img_height, self.img_width = self.img_original.shape[:2]
            self.img_height_centr, self.img_width_centr = int(self.img_height/2), int(self.img_width/2)
            self.StepLogAdd = f'\n  img_height, img_width = {self.img_height}, {self.img_width}'
            self.StepLogAdd = f'\n  img_height_center, img_width_center = {self.img_height_centr}, {self.img_width_centr}'

            self.save_pic_debug(self.img_original, f"img_original_01.jpg")

            img_rotate = None
            qrcode_position = "down"
            qrcode_position = "left"
            qrcode_position = "right"
            qrcode_position = "top"

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

                # self.img_grayscale = self.img_original.copy()
                # self.img_grayscale = cv2.cvtColor(self.img_grayscale, cv2.COLOR_BGR2GRAY)
                # self.save_pic_debug(self.img_grayscale, f"img_grayscale_01.jpg")
                # # copy wb
                # #
                # contrast = .99  # (0-127)
                # brightness = .1  # (0-100)
                # contrasted_img = cv2.addWeighted(
                #     self.img_grayscale,
                #     contrast,
                #     self.img_grayscale, 0,
                #     brightness
                # )
                # blured_img_main = cv2.bilateralFilter(contrasted_img, 1, 75, 75)
                # self.save_pic_debug(blured_img_main, f"img_wb_blured_01.jpg")
                # im_bw = cv2.threshold(blured_img_main, 170, 255, cv2.THRESH_TOZERO)[1]
                # self.wb_blured_img = cv2.threshold(im_bw, 10, 255, cv2.THRESH_BINARY)[1]
                # self.save_pic_debug(self.wb_blured_img, f"img_wb_blured_02.jpg")
                # self.wb_blured_img = self._denoise_wb_img(self.wb_blured_img)


                # Extract the red channel
                red_channel = self.img_original[:, :, 2]  # Red channel index is 2
                # Create a mask where red channel values are greater than 200
                mask = red_channel > 200
                # Replace pixels where mask is True with white
                self.img_original[mask] = [255, 255, 255]

                image = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
                wb_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
                blured_img_main = cv2.bilateralFilter(wb_image, 1, 75, 75)
                self.wb_blured_img = cv2.threshold(blured_img_main, 170, 255, cv2.THRESH_TOZERO)[1]

                # self.wb_blured_img = cv2.threshold(im_bw, 10, 255, cv2.THRESH_BINARY)[1]
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

    def rotate_and_crop_check(self):
        gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        # print(angle)
        if angle < 30:
            angle = -angle
        else:
            angle = 90 - angle

        print("CHECK ANGLE", angle)

        h, w = self.img_height, self.img_width
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.img_original, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # TODO: remove it, make cropping more accurate
        self.rotated_img = self.img_original = rotated

        self.save_pic_debug(rotated, f"rotated_check.jpg")

        if abs(angle) > 1:
            img = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
            thresh = cv2.inRange(img, (0, 0, 0), (255, 255, 232))
            thresh = cv2.threshold(thresh, 200, 255, cv2.THRESH_BINARY_INV)[1]

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            big_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(big_contour)

            cropped = rotated[
                0:thresh.shape[0],
                x:x + w
            ]
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

    @staticmethod
    def split_lotto_number_contours(number_contours, sort_by_oy=False):
        prev_number_contour = 0
        index_of_first_strong = 0
        for index, contour in enumerate(number_contours):
            if prev_number_contour != 0 and contour[0] - prev_number_contour[0] >= 100:
                index_of_first_strong = index
                break
            prev_number_contour = contour

        regular_contour_nums = number_contours[:index_of_first_strong]
        strong_contour_nums = number_contours[index_of_first_strong::]
        # because here can be numbers in few lines, so we should sort them in right way(for lotto strong and systematic)
        if sort_by_oy:
            regular_contour_nums.sort(key=lambda cnt: [1])
            strong_contour_nums.sort(key=lambda cnt: [1])
        return regular_contour_nums, strong_contour_nums

    @staticmethod
    def _denoise_wb_img(cropped_img, noise_width=3, noise_height=3):
        contours = cv2.findContours(cropped_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  # CHAIN_APPROX_NONE
        contours = [cv2.boundingRect(cnt) for cnt in contours]

        contours = sorted(contours, key=lambda cnt: cnt[2] * cnt[3])
        for cnt in contours:
            x, y, w, h = cnt
            if w < noise_width or h < noise_height:
                for i in range(h):
                    for j in range(w):
                        cropped_img[y + i, x + j] = 255
            else:
                break
        return cropped_img

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
    def resize_cropped_img(img, target_width):
        height, width = img.shape[:2]
        ratio = target_width / width
        target_height = int(height * ratio)
        resized_img = cv2.resize(img, (target_width, target_height))
        return resized_img

    def data_incorrect_parsed_log(self, stringed_error, data_key):
        print(stringed_error)
        self.job_log(stringed_error)
        self.is_incorrect_check_info.update({data_key: True})

    def get_game_type(self):
        try:
            game_type_img = self.wb_blured_img[
                self.qr_code_info["bottom_left"][1] + 310:self.qr_code_info["bottom_left"][1] + 310 + 300,
                self.qr_code_info["top_left"][0] - 158:self.qr_code_info["top_right"][0] + 158
            ]
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

            contours_points = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # One of elements it is data what we do not really need
            # contours_points - list of numpy arrays with elements: [[[x, y]], [[x, y]], [[x, y]]]
            contours_points = contours_points[0] if len(contours_points) == 2 else contours_points[1]
            contours_points = list(contours_points)

            # Filter None values
            sorted_contours_points = filter(lambda line_array: line_array is not None, contours_points)

            # Numpy array to python list
            lines_x_y_points = list(map(lambda line_array: line_array.tolist(), sorted_contours_points))

            # Removing unwanted nesting
            # From [[[x, y]], [[x, y]], [[x, y]]] to [[x, y], [x, y], [x, y]]  (One element it is points for one line)
            unnested_lines_points = list(
                map(lambda list_of_x_y_pairs: list(
                    map(lambda x_y_pair: x_y_pair[0], list_of_x_y_pairs)
                ), lines_x_y_points)
            )

            # Now we have lists with points of lines, but needed lines can be splited
            # So we should merge lines with +- same OY
            # unnested_lines_points already sorted from bigger to smaller
            result_lines_points = []
            for index, line_points in enumerate(unnested_lines_points):
                if index == 0:
                    result_lines_points.append(line_points)
                else:
                    last_line = result_lines_points[-1]
                    last_point_y = last_line[-1][1]
                    if last_point_y - line_points[-1][1] <= 17:
                        result_lines_points.pop()
                        result_lines_points.append([*last_line, *line_points])
                    else:
                        result_lines_points.append(line_points)

            # We need only two lines around cards/table
            # So this two lines have the most points count(width)
            line_threshold = self.img_width - 100  # if line width >= threshold then we take this line
            lines = []

            for line_points in result_lines_points:
                min_x, max_x, sum_y = line_points[0][0], 0, 0
                for x, y in line_points:
                    sum_y += y
                    if min_x > x:
                        min_x = x
                    if max_x < x:
                        max_x = x
                y = sum_y // len(line_points)

                # print("MAIN LINES BEFORE WIDTH AND POSITION CHECK:", {"min_x": min_x, "max_x": max_x, "y": y})
                # Remove lines which OY coord in range from 0 to 1200 and from img.height to img.height-500
                # Because card/table lines not in those diapason
                # and if x2 - x1 >= threshold
                condition = max_x - min_x >= line_threshold and y not in range(0, 1200) and y not in range(
                    self.img_height - 500, self.img_height)
                if condition:
                    lines.append({"min_x": min_x, "max_x": max_x, "y": y})

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
            kernel = np.ones((2, 2), np.uint8)
            bold_check_bottom_part = cv2.erode(check_bottom_part, kernel, iterations=1)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 5))
            img = cv2.morphologyEx(bold_check_bottom_part, cv2.MORPH_OPEN, kernel)
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

    def get_game_subtype(self):
        try:
            img = cv2.imread("template_images/no_needed_repeat_game.png")
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
            self.check_info["game_subtype"] = parsed_subtype
            if parsed_subtype == "":
                self.data_incorrect_parsed_log("Game subtype was not found(get_game_subtype)", "game_subtype")
        except:
            self.data_incorrect_parsed_log("Game subtype was not found, error occurred(get_game_subtype)", "game_subtype")
            self.StepStatus = "FAIL"
        return self.check_info["game_subtype"]

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

    def get_dashed_number(self):
        try:
            decoded_objects = pyzbar.decode(self.rotated_img, symbols=[pyzbar.ZBarSymbol.I25])
            self.check_info["dashed_number"] = decoded_objects[0].data.decode("utf-8")[:-1]  # Read with no needed "0" at the end
        except:
            self.data_incorrect_parsed_log(
                f"The length of the number is not correct, error occurred(dashed_number)",
                "dashed_number"
            )
            self.StepStatus = "FAIL"
        return self.check_info["dashed_number"]

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

    def get_cards(self):
        try:
            top_line = self.longest_lines["top_line"]
            bottom_line = self.longest_lines["bottom_line"]

            cards_img = self.wb_blured_img[
                top_line["y"]:bottom_line["y"],
                0:self.img_width
            ]
            # cards_img = cv2.threshold(cards_img, 200, 255, cv2.THRESH_BINARY)[1]
            # cards_img = cv2.GaussianBlur(cards_img, (3, 3), 0)
            self.save_pic_debug(cards_img, f"cards/cards.jpg")

            edges = cv2.Canny(cards_img, 10, 200)
            contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Contours to list of tuples
            info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)
            # Sort contours in right way
            sorted_contours = sorted(info_of_contours, key=lambda contour: contour[0])
            # Remove garbage contours
            sorted_contours = [contour for contour in sorted_contours if
                               contour[2] >= 30 and contour[3] >= 70]

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
            # print(sorted_contours)
            print(unique_contours)
            for i, contour in enumerate(unique_contours):
                one_card_img = cards_img[
                    contour[1]:contour[1]+contour[3],
                    contour[0]:contour[0] + contour[2],
                ]
                # TODO: refactor
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
            self.save_pic_debug(crop_img, f"table/table.jpg")
            crop_img_original = crop_img.copy()
            crop_img_blured = cv2.GaussianBlur(crop_img, (41, 41), 1)
            crop_img = cv2.threshold(crop_img_blured, 200, 255, cv2.THRESH_BINARY)[1]
            self.save_pic_debug(crop_img, f"table/table_blured.jpg")
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            img_dilated = cv2.dilate(cv2.threshold(crop_img, 127, 255, cv2.THRESH_BINARY_INV)[1], kernel)
            # cv2.imshow("", img_dilated)
            # cv2.waitKey(0)
            # exit()

            contours = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)

            # Remove garbage contours
            min_width = 30 if game_type == "123" else 17
            max_width = 60 if game_type == "123" else 35
            min_height = 35 if game_type == "123" else 35

            sorted_contours = [contour for contour in info_of_contours if contour[2] >= min_width and contour[2] <= max_width and contour[3] >= min_height]

            if game_type == "123":
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
                        self.save_pic_debug(crop_number, f"table/cropped_number{j}, {index}.jpg")

                        table_number = self.get_value_from_image(crop_number, "table")
                        self.check_info["table"][f"line_{index + 1}"]["regular"].append(int(table_number))

            elif game_type == "777":
                current_line = 0
                numbers_in_lines = []
                prev_number_OY = 0
                sorted_by_OY = sorted(sorted_contours, key=lambda contour: contour[1])
                numbers_in_lines.append([])

                for number in sorted_by_OY:
                    if prev_number_OY == 0:
                        prev_number_OY = number[1]

                    if number[1] - prev_number_OY > 10:
                        current_line += 1
                        numbers_in_lines.append([])

                    numbers_in_lines[current_line].append(number)

                    prev_number_OY = number[1]

                for index, line in enumerate(numbers_in_lines):
                    line.sort(key=lambda number: number[0])
                    numbers_in_lines[index] = line

                for index, line in enumerate(numbers_in_lines):
                    stringed_numbers = ""
                    for number in line:
                        crop_number = crop_img_original[
                            number[1]:number[1] + number[3],
                            number[0]:number[0] + number[2]
                        ]
                        table_number = self.get_value_from_image(crop_number, "table")
                        stringed_numbers += table_number

                    # merge numbers by two
                    i = 0
                    while i < len(stringed_numbers):
                        try:
                            make_resulted_number = f"{stringed_numbers[i]}{stringed_numbers[i + 1]}"
                        except:
                            make_resulted_number = f"{stringed_numbers[i]}"
                            print(f"Not all regular numbers were found in table line")

                        if not self.check_info["table"].get(f"line_{index + 1}", False):
                            self.check_info["table"][f"line_{index + 1}"] = {
                                "regular": []
                            }
                        self.check_info["table"][f"line_{index + 1}"]["regular"].append(int(make_resulted_number))
                        i += 2
        except Exception as ex:
            self.data_incorrect_parsed_log(
                f"Table was not found, error occurred(get_table_123_777): {ex}",
                "table_123_777"
            )
            self.StepStatus = "FAIL"
        return self.check_info["table"]

    def get_table_lotto(self):
        try:
            if self.bottom_oy_border_for_table == 0:
                bottom_oy_border = self.longest_lines["middle_line"]["y"]
            else:
                # finding from top_line to top of subtype if exists
                bottom_oy_border = self.bottom_oy_border_for_table

            crop_img = self.wb_blured_img[
                self.longest_lines["top_line"]["y"]+90:bottom_oy_border,  # 90px it is cropping without table header
                0:self.img_width
            ]
            self.save_pic_debug(crop_img, f"table/table.jpg")

            inverted_table = cv2.threshold(crop_img, 127, 255, cv2.THRESH_BINARY_INV)[1]
            if self.check_info["game_subtype"] == "lotto_regular":
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
                dilated_table = cv2.dilate(inverted_table, kernel)
                lines_contours = cv2.findContours(dilated_table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                lines_contours = [cv2.boundingRect(cnt) for cnt in lines_contours]
                sorted_lines_contours = [contour for contour in lines_contours if
                                         contour[2] >= 700 and contour[2] <= 800 and contour[3] >= 40]
                sorted_lines_contours.sort(key=lambda line: line[1])

                for index, line in enumerate(sorted_lines_contours):
                    self.check_info["table"][f"line_{index + 1}"] = {
                        "regular": [],
                        "strong": [],
                    }
                    x, y, w, h = line
                    line_img = crop_img[
                        y:y + h,
                        x:x + w
                    ]

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    dilated_line = cv2.dilate(
                        cv2.threshold(line_img, 127, 255, cv2.THRESH_BINARY_INV)[1],
                        kernel
                    )
                    number_contours = cv2.findContours(dilated_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    number_contours = [cv2.boundingRect(cnt) for cnt in number_contours]

                    sorted_number_contours = [contour for contour in number_contours if
                                              contour[2] >= 10 and contour[3] >= 25]
                    sorted_number_contours.sort(key=lambda cnt: cnt[0])
                    self.job_log(f"Lotto table sorted number contours: {sorted_number_contours}")

                    regular_numbers, strong_numbers = self.split_lotto_number_contours(sorted_number_contours)

                    # Find regular numbers
                    stringed_numbers = ""
                    for number in regular_numbers:
                        x, y, w, h = number
                        number_img = line_img[
                            y:y + h,
                            x:x + w
                        ]
                        table_number = self.get_value_from_image(number_img, "table")
                        stringed_numbers += table_number

                    self.job_log(f"Stringed regular numbers: {stringed_numbers}")

                    # merge numbers by two
                    i = 0
                    while i < len(stringed_numbers):
                        try:
                            make_resulted_number = f"{stringed_numbers[i]}{stringed_numbers[i + 1]}"
                        except:
                            make_resulted_number = f"{stringed_numbers[i]}"
                            print(f"Not all regular numbers were found in table line")

                        self.check_info["table"][f"line_{index + 1}"]["regular"].append(int(make_resulted_number))
                        i += 2

                    # Find strong number
                    for number in strong_numbers:
                        x, y, w, h = number
                        number_img = line_img[
                            y:y + h,
                            x:x + w
                        ]
                        table_number = self.get_value_from_image(number_img, "table")
                        if table_number == "(":
                            break
                        self.check_info["table"][f"line_{index + 1}"]["strong"].append(int(table_number))

            # TODO: do parsing for those types
            elif self.check_info["game_subtype"] in ["lotto_strong", "lotto_systematic"]:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                dilated_table = cv2.dilate(inverted_table, kernel)
                number_contours = cv2.findContours(dilated_table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                number_contours = [cv2.boundingRect(cnt) for cnt in number_contours]

                sorted_number_contours = [contour for contour in number_contours if
                                          contour[2] >= 10 and contour[3] >= 25]
                sorted_number_contours.sort(key=lambda cnt: cnt[0])

                for number in sorted_number_contours:
                    x, y, w, h = number
                    number_img = crop_img[
                        y:y + h,
                        x:x + w
                    ]
                    table_number = self.get_value_from_image(number_img, "table")
                    self.check_info["table"].append(table_number)
        except:
            self.data_incorrect_parsed_log(
                f"Table was not found, error occurred(get_table_lotto)",
                "table_lotto"
            )
            self.StepStatus = "FAIL"
        return self.check_info["table"]

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
            self.save_pic_debug(extra_number_crop, f"extra/extra_number.jpg")

            is_extra = self.get_value_from_image(extra_crop, "is_extra")
            if is_extra:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                img_dilated = cv2.dilate(
                    cv2.threshold(extra_number_crop, 127, 255, cv2.THRESH_BINARY_INV)[1],
                    kernel
                )

                contours = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)
                sorted_contours = [contour for contour in info_of_contours if
                                   contour[2] >= 17 and contour[2] <= 30 and contour[3] >= 25]
                sorted_by_OY = sorted(sorted_contours, key=lambda contour: contour[0])
                for number in sorted_by_OY:
                    crop_number = extra_number_crop[
                        number[1]:number[1] + number[3],
                        number[0]:number[0] + number[2]
                    ]
                    extra_number = self.get_value_from_image(crop_number, "extra_numbers")
                    self.check_info["extra_number"] += extra_number

            self.check_info["extra"] = is_extra
        except:
            self.data_incorrect_parsed_log(f"Extra was not found, error occurred(get_extra)", "extra")
            self.StepStatus = "FAIL"
        return self.check_info["extra"], self.check_info["extra_number"]

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
                img = cv2.imread("template_images/table_123_numbers.png")
                d_table_numbers = d_table_123_numbers
            elif game_type == "777":
                img = cv2.imread("template_images/table_777_numbers.png")
                d_table_numbers = d_table_777_numbers
            elif game_type == "lotto":
                img = cv2.imread("template_images/table_lotto_numbers.png")
                d_table_numbers = d_table_lotto_numbers

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(img, cropped_img, cv2.TM_CCOEFF_NORMED)
            y, x = np.unravel_index(res.argmax(), res.shape)
            print(x, y)
            for key, value in d_table_numbers.items():
                if x in range(*value[0]) and y in range(*value[1]):
                    return key

        elif data_type == "extra_numbers":
            img = cv2.imread("template_images/extra_numbers.png")
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
            for key, value in d_extra_numbers.items():
                if x in range(*value[0]) and y in range(*value[1]):
                    return key

        elif data_type == "game_type":
            img = cv2.imread("template_images/game_types.png")
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
                img = cv2.imread("template_images/777_systematic_subtype.png")
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
                    img = cv2.imread("template_images/777_systematic_subtype_col8.png")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    res = cv2.matchTemplate(cropped_img, img, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= 0.7)  # THRESHOLD
                    if len(loc[0].tolist()) == 0:
                        result = "777_col9"
                    else:
                        result = "777_col8"

            elif game_type == "chance":
                img_type_in_subtype = cv2.imread("template_images/chance_subtype.png")
                img = cv2.cvtColor(img_type_in_subtype, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(cropped_img, img, cv2.TM_CCOEFF_NORMED)
                y, x = np.unravel_index(res.argmax(), res.shape)
                # Crop exactly subtype
                cropped_subtype = cropped_img[
                    y:y+50,
                    x-180:x+101+70
                ]

                img_subtype_multi = cv2.imread("template_images/chance_multi_subtype.png")
                img_subtype_multi = cv2.cvtColor(img_subtype_multi, cv2.COLOR_BGR2GRAY)

                img_subtype_systematic = cv2.imread("template_images/chance_systematic_subtype.png")
                img_subtype_systematic = cv2.cvtColor(img_subtype_systematic, cv2.COLOR_BGR2GRAY)

                res_suffix_subtype = cv2.matchTemplate(cropped_subtype, img_subtype_multi, cv2.TM_CCOEFF_NORMED)
                loc_suffix = np.where(res_suffix_subtype >= 0.7)  # THRESHOLD

                res_prefix_subtype = cv2.matchTemplate(cropped_subtype, img_subtype_systematic, cv2.TM_CCOEFF_NORMED)
                loc_prefix = np.where(res_prefix_subtype >= 0.7)  # THRESHOLD

                if len(loc_suffix[0].tolist()) != 0:
                    result = "chance_multi"
                elif len(loc_prefix[0].tolist()) != 0:
                    result = "chance_systematic"
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

            elif game_type == "lotto":  # TODO: subtype
                lotto_strong_subtype = cv2.imread("template_images/lotto_strong_subtype.png")
                lotto_strong_subtype = cv2.cvtColor(lotto_strong_subtype, cv2.COLOR_BGR2GRAY)

                lotto_systematic_subtype = cv2.imread("template_images/lotto_systematic_subtype.png")
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
                        self.bottom_oy_border_for_table = self.longest_lines["bottom_line"]["y"] - 60

            return result

        elif data_type in ["numbers", "date", "sum", "game_id"]:
            print("numbers", "date", "sum", "game_id")
            if data_type == "sum":
                height, width = cropped_img.shape[:2]
                find_symbol_img = cv2.imread("template_images/sum_symbol_border.png")
                find_symbol_img = cv2.cvtColor(find_symbol_img, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(cropped_img, find_symbol_img, cv2.TM_CCOEFF_NORMED)
                y, x = np.unravel_index(res.argmax(), res.shape)

                cropped_img = cropped_img[
                    0:height,
                    x + 20:width
                ]

            symbols_numbers_img = cv2.imread("template_images/numbers_and_letters.png")
            numbers_img = cv2.imread("template_images/numbers.png")

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

                def minimal_black_pixels_count(contour_tuple):  # TODO: change from function to method
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
            img = cv2.imread("template_images/extra_true.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(cropped_img, img, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.7)  # THRESHOLD
            if len(loc[0].tolist()) != 0:
                return True
            return False

        elif data_type.startswith("card"):
            card_type = data_type.split("_")[1]
            img = cv2.imread(f"template_images/cards_{card_type}.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # TODO: change to staticmethod
            height, width = cropped_img.shape[:2]
            target_width = 127
            ratio = target_width / width
            target_height = int(height * ratio)
            resized_card = cv2.resize(cropped_img, (target_width, target_height))
            resized_card = cv2.threshold(resized_card, 200, 255, cv2.THRESH_BINARY)[1]

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
        self.step_start()
        self.set_param()
        self.step_stop("set_param")

        self.step_start()
        self.img_prep()
        self.step_stop("img_prep")

        if self.img_is_valid:
            print("Image is valid.")
            self.step_start()
            self.get_coords_of_main_lines()
            self.step_stop("get_coords_of_main_lines")
            self.job_log(f"Main check lines: {self.longest_lines}")

            self.step_start()
            self.get_game_type()
            self.step_stop("get_game_type")
            game_type = self.check_info["game_type"]

            self.step_start()
            self.get_game_subtype()
            self.step_stop("get_game_subtype")

            self.step_start()
            self.get_main_data_contours()
            self.step_stop("get_main_data_contours")

            self.step_start()
            self.get_game_id()
            self.step_stop("get_game_id")

            self.step_start()
            self.get_date()
            self.step_stop("get_date")

            self.step_start()
            self.get_spent_money()
            self.step_stop("get_spent_money")

            self.step_start()
            self.get_dashed_number()
            self.step_stop("get_dashed_number")

            self.step_start()
            self.get_spaced_number()
            self.step_stop("get_spaced_number")

            if game_type == "chance":
                self.step_start()
                self.get_cards()
                self.step_stop("get_cards")
            elif game_type == "lotto":
                self.step_start()
                self.get_table_lotto()
                self.step_stop("get_table_lotto")
                self.step_start()
                self.get_extra()
                self.step_stop("get_extra")
            else:
                self.step_start()
                self.get_table_123_777()
                self.step_stop("get_table_123_777")
        else:
            print("Image is not valid. Data was not parsed.")
            self.all_data_not_found = True

        return self.check_info, self.all_data_not_found, self.is_incorrect_check_info, self.StepLogFull



# TODO: check 123, 777
# TODO: change in get_result() from step_start and step_stop to decorator
# TODO: REFACTOR !!!!
# TODO: find "autofilling" data near subtype
# TODO: Type hints ???

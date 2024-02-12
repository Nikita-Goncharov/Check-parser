# TODO: Maybe move it to json ???

game_types = {
    "lotto": [[0, 550], [290, 450]],
    "123": [[710, 1200], [290, 480]],
    "chance": [[0, 700], [0, 250]],
    "777": [[710, 1225], [0, 250]],
}

cards_spades = {
    # "A": [[40, 185], [40, 260]],
    # "K": [[40, 185], [280, 505]],
    # "Q": [[40, 185], [515, 760]],
    # "J": [[40, 185], [780, 1020]],
    # "10": [[40, 185], [1040, 1280]],
    # "9": [[40, 185], [1300, 1540]],
    # "8": [[40, 185], [1560, 1800]],
    # "7": [[40, 185], [1820, 2060]],
    "A": [[35, 185], [35, 240]],
    "K": [[35, 185], [250, 440]],
    "Q": [[35, 185], [450, 645]],
    "J": [[35, 185], [655, 855]],
    "10": [[35, 185], [860, 1060]],
    "9": [[35, 185], [1070, 1270]],
    "8": [[35, 185], [1280, 1480]],
    "7": [[35, 185], [1490, 1690]],
}

cards_hearts = {
    "A": [[35, 185], [35, 240]],
    "K": [[35, 185], [250, 440]],
    "Q": [[35, 185], [450, 645]],
    "J": [[35, 185], [655, 855]],
    "10": [[35, 185], [860, 1060]],
    "9": [[35, 185], [1070, 1270]],
    "8": [[35, 185], [1280, 1480]],
    "7": [[35, 185], [1490, 1690]],
}

cards_diamonds = {
    "A": [[35, 185], [35, 240]],
    "K": [[35, 185], [250, 440]],
    "Q": [[35, 185], [450, 645]],
    "J": [[35, 185], [655, 855]],
    "10": [[35, 185], [860, 1060]],
    "9": [[35, 185], [1070, 1270]],
    "8": [[35, 185], [1280, 1480]],
    "7": [[35, 185], [1490, 1690]],
}

cards_clubs = {
    "A": [[35, 185], [35, 240]],
    "K": [[35, 185], [250, 440]],
    "Q": [[35, 185], [450, 645]],
    "J": [[35, 185], [655, 855]],
    "10": [[35, 185], [860, 1060]],
    "9": [[35, 185], [1070, 1270]],
    "8": [[35, 185], [1280, 1480]],
    "7": [[35, 185], [1490, 1690]],
}

# All coords should be 3+x1, x2+3 and 3+y1, y2+3
d_table_777_numbers = {
    "0": [[0, 30], [0, 50]],
    "1": [[35, 70], [0, 50]],
    "2": [[72, 105], [0, 50]],
    "3": [[110, 145], [0, 50]],
    "4": [[150, 185], [0, 50]],
    "5": [[190, 225], [0, 50]],
    "6": [[230, 265], [0, 50]],
    "7": [[267, 300], [0, 50]],
    "8": [[305, 340], [0, 50]],
    "9": [[345, 380], [0, 50]],
}

d_table_123_numbers = {
    "0": [[0, 45], [0, 50]],
    "1": [[46, 95], [0, 50]],
    "2": [[98, 145], [0, 50]],
    "3": [[148, 193], [0, 50]],
    "4": [[195, 240], [0, 50]],
    "5": [[242, 288], [0, 50]],
    "6": [[290, 325], [0, 50]],
    "7": [[330, 380], [0, 50]],
    "8": [[383, 426], [0, 50]],
    "9": [[430, 480], [0, 50]],
}

d_table_lotto_numbers = {
    "0": [[0, 30], [0, 35]],
    "1": [[35, 65], [0, 35]],
    "2": [[70, 95], [0, 35]],
    "3": [[100, 130], [0, 35]],
    "4": [[135, 165], [0, 35]],
    "5": [[170, 200], [0, 35]],
    "6": [[205, 235], [0, 35]],
    "7": [[240, 265], [0, 35]],
    "8": [[270, 300], [0, 35]],
    "9": [[305, 335], [0, 35]],
    "(": [[340, 360], [0, 35]],
    ")": [[365, 380], [0, 35]],
}

d_extra_numbers = {
    "1": [[0, 35], [0, 35]],
    "2": [[40, 65], [0, 35]],
    "3": [[70, 100], [0, 35]],
    "4": [[105, 135], [0, 35]],
    "5": [[140, 170], [0, 35]],
    "6": [[175, 200], [0, 35]],
    "7": [[205, 235], [0, 35]],
}


d_numbers = {
    "0": [[0, 29], [0, 30]],
    "1": [[32, 63], [0, 30]],
    "2": [[64, 90], [0, 30]],
    "3": [[95, 125], [0, 30]],
    "4": [[132, 160], [0, 30]],
    "5": [[166, 195], [0, 30]],
    "6": [[202, 230], [0, 30]],
    "7": [[237, 265], [0, 30]],
    "8": [[270, 295], [0, 30]],
    "9": [[301, 330], [0, 30]],
    "(": [[340, 355], [0, 35]],
    ")": [[365, 380], [0, 35]],
}

d_letters = {
    "B": [[0, 25], [40, 80]],
    "Q": [[35, 65], [40, 80]],
    "P": [[72, 100], [40, 80]],
    "V": [[110, 135], [40, 80]],
    "L": [[145, 175], [40, 80]],
    "C": [[185, 210], [40, 80]],
    "S": [[215, 240], [40, 80]],
    "F": [[245, 270], [40, 80]],
    "Y": [[280, 305], [40, 80]],
    "G": [[310, 335], [40, 80]],
    "H": [[345, 370], [40, 80]],
    "J": [[373, 400], [40, 80]],
    "R": [[403, 430], [40, 80]],
    "W": [[435, 462], [40, 80]],
    "K": [[465, 490], [40, 80]],
    "D": [[495, 520], [40, 80]],
    "T": [[525, 552], [40, 80]],
    "N": [[555, 580], [40, 80]],
    "Z": [[582, 610], [40, 80]],
    "X": [[615, 640], [40, 80]],
    "M": [[643, 670], [40, 80]],
}

d_special_symbols = {
    "&": [[0, 30], [83, 120]],
    "%": [[35, 63], [83, 120]],
    "#": [[65, 90], [83, 120]],
    "@": [[95, 125], [83, 120]],
    "-": [[130, 158], [83, 120]],
    "*": [[160, 180], [83, 120]],
    # "(": [[190, 210], [83, 120]],
    # ")": [[215, 230], [83, 120]],
}

d_all_symbols = {}
d_all_symbols.update(d_numbers)
d_all_symbols.update(d_letters)
d_all_symbols.update(d_special_symbols)

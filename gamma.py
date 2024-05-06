"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from ex1_utils import LOAD_GRAY_SCALE, LOAD_RGB
import numpy as np
import cv2


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    def gammaChange(val: int):
        gamma = val / 100
        img_cpy = np.round(np.power(img_norm, gamma)*255).astype(np.uint8)
        cv2.imshow('Gamma Correction', img_cpy)

    img = cv2.imread(img_path)
    if rep == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_norm = img / 255

    cv2.namedWindow('Gamma Correction', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Gamma value', 'Gamma Correction', 0, 200, gammaChange)

    cv2.waitKey(0)


def main():
    gammaDisplay('water_bear.png', LOAD_RGB)


if __name__ == '__main__':
    main()

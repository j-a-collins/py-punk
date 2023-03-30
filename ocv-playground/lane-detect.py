# """
# A place to play with OpenCV and lane detection. Mostly playing around
# with the Hough transform and Canny edge detection before I get into
# the deep end with the the self-driving car for Cyberpunk 2077

# Author: J-A-Collins
# Date: 2020-08-01
# """

# # Imports
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# def canny_method(img):
#     """A function that takes in an image and implements
#     the Canny edge detection method. Converts the image to
#     grayscale, reduces noise with a Gaussian blur, and then
#     finally applies the Canny edge detection method.

#     Parameters
#     ----------
#     img : numpy.ndarray
#         The image to be processed.

#     Returns
#     -------
#     canny_img : numpy.ndarray
#         The processed image.
#     """
#     grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred_img = cv2.GaussianBlur(grayscale, (5, 5), 0)
#     canny_img = cv2.Canny(blurred_img, 15, 45)  # (10, 30), (15, 45), (20, 60)
#     return canny_img


# def triangle_view(img):
#     """A function that creates a triangle view in the image
#     to focus on the lane lines.
#     Parameters
#     ----------
#     img : numpy.ndarray
#         The image to be processed.
#     Returns
#     -------
#     triangle : numpy.ndarray
#         The processed image.
#     """
#     max_h = 700  # img.shape[0]  # height, -e_y
#     max_x = img.shape[1]  # width, -e_x
#     polygons = list(
#         np.array([[(80, max_h), (max_x + 200, max_h), (630, 410)]], dtype=np.int32)
#     )
#     masking = np.zeros_like(img)
#     cv2.fillPoly(masking, polygons, 255)  # polygon colour white (255)
#     mask_img = cv2.bitwise_and(img, masking)
#     return mask_img


# def make_coordinates(img, line_parameters):
#     """A function that takes in an image and line parameters
#     and returns the coordinates of the line.

#     Parameters
#     ----------
#     img : numpy.ndarray
#         The image to be processed.
#         line_parameters : numpy.ndarray
#         The line parameters to be processed.

#     Returns
#     -------
#     coordinates : numpy.ndarray
#     The processed image."""
#     slope, intercept = line_parameters
#     y1 = img.shape[0]
#     y2 = int(y1 * (3 / 5))
#     x1 = int((y1 - intercept) / slope)
#     x2 = int((y2 - intercept) / slope)
#     coordinates = np.array([x1, y1, x2, y2])
#     return coordinates


# def average_slope_intercept(img, lines):
#     """A function that takes in an image and lines and
#     returns the average slope and intercept of the lines.

#     Parameters
#     ----------
#     img : numpy.ndarray
#         The image to be processed.
#     lines : numpy.ndarray
#         The lines to be averaged.

#     Returns
#     -------
#     average_lines : numpy.ndarray
#         The processed image.
#     """
#     left_fit, right_fit = [], []
#     for line in lines:
#         x1, y1, x2, y2 = line.reshape(4)
#         parameters = np.polyfit((x1, x2), (y1, y2), 1)
#         slope = parameters[0]
#         intercept = parameters[1]
#         if slope < 0:
#             left_fit.append((slope, intercept))
#         else:
#             right_fit.append((slope, intercept))
#     left_fit_average = np.average(left_fit, axis=0)
#     right_fit_average = np.average(right_fit, axis=0)
#     left_line = make_coordinates(img, left_fit_average)
#     right_line = make_coordinates(img, right_fit_average)
#     average_lines = np.array([left_line, right_line])
#     return average_lines


# def display_lines(img, lines):
#     """A function that takes in an image and lines and
#     displays the lines on the image.

#     Parameters
#     ----------
#     img : numpy.ndarray
#         The image to be processed.
#     lines : numpy.ndarray
#         The lines to be displayed.
#     Returns
#     -------
#     line_img : numpy.ndarray
#         The processed image.
#     """
#     line_img = np.zeros_like(img)
#     if lines.all():  # Not None, .any()?
#         for line in lines:
#             x1, y1, x2, y2 = line.reshape(4)
#             cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
#     return line_img


# if __name__ == "__main__":
#     print("Lane detection playground")
#     # Load the image:
# lane_img = cv2.imread(
#     r"C:\Users\jacka\Documents\GitHub\py-punk\ocv-playground\inputs\test_frame.jpg"
# )
# # Create a copy of the image, for the thresholding:
# lane_detect = np.copy(lane_img)
# # Apply edge detection:
# canny_img = canny_method(lane_detect)
# cropped_img = triangle_view(canny_img)
# lines = cv2.HoughLinesP(
#     cropped_img, 2, np.pi / 180, 100, np.array([]), minLineLength=5, maxLineGap=20
# )
# mean_lines = average_slope_intercept(lane_detect, lines)
# masked_lane_detect = triangle_view(lane_detect)  # Mask the original image
# line_img = display_lines(
#     masked_lane_detect, mean_lines
# )  # Draw the lines on the masked image
# threshold_img = cv2.addWeighted(lane_detect, 0.8, line_img, 1, 1)
#     # Display the image:
#     cv2.imshow("output", threshold_img)  # Show the threshold_img directly
#     cv2.waitKey(0)

import cv2
import numpy as np


class LaneDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.processed_image = None


    def canny_method(self, img):
        """A function that takes in an image and implements
        the Canny edge detection method. Converts the image to
        grayscale, reduces noise with a Gaussian blur, and then
        finally applies the Canny edge detection method.

        Parameters
        ----------
        img : numpy.ndarray
            The image to be processed.

        Returns
        -------
        canny_img : numpy.ndarray
            The processed image.
        """
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(grayscale, (5, 5), 0)
        canny_img = cv2.Canny(blurred_img, 15, 45)  # (10, 30), (15, 45), (20, 60)
        return canny_img

    def average_slope_intercept(self, img, lines):
        """A function that takes in an image and lines and
        returns the average slope and intercept of the lines.

        Parameters
        ----------
        img : numpy.ndarray
            The image to be processed.
        lines : numpy.ndarray
            The lines to be averaged.

        Returns
        -------
        average_lines : numpy.ndarray
            The processed image.
        """
        left_fit, right_fit = [], []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = self.make_coordinates(img, left_fit_average)
        right_line = self.make_coordinates(img, right_fit_average)
        average_lines = np.array([left_line, right_line])
        return average_lines

    def make_coordinates(self, img, line_parameters):
        """A function that takes in an image and line parameters
        and returns the coordinates of the line.

        Parameters
        ----------
        img : numpy.ndarray
            The image to be processed.
        line_parameters : numpy.ndarray
            The line parameters to be processed.

        Returns
        -------
        coordinates : numpy.ndarray
        The processed image."""
        slope, intercept = line_parameters
        y1 = img.shape[0]
        y2 = int(y1 * (3 / 5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        coordinates = np.array([x1, y1, x2, y2])
        return coordinates

    def triangle_view(self, img):
        """A function that creates a triangle view in the image
        to focus on the lane lines.
        Parameters
        ----------
        img : numpy.ndarray
            The image to be processed.
        Returns
        -------
        triangle : numpy.ndarray
            The processed image.
        """
        max_h = 700  # img.shape[0]  # height, -e_y
        max_x = img.shape[1]  # width, -e_x
        polygons = list(
            np.array([[(80, max_h), (max_x + 200, max_h), (630, 410)]], dtype=np.int32)
        )
        masking = np.zeros_like(img)
        cv2.fillPoly(masking, polygons, 255)  # polygon colour white (255)
        mask_img = cv2.bitwise_and(img, masking)
        return mask_img

    def display_lines(self, img, lines):
        """A function that takes in an image and lines and
        displays the lines on the image.

        Parameters
        ----------
        img : numpy.ndarray
            The image to be processed.
        lines : numpy.ndarray
            The lines to be displayed.
        Returns
        -------
        line_img : numpy.ndarray
            The processed image.
        """
        line_img = np.zeros_like(img)
        if lines.all():  # Not None, .any()?
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_img

    def detect_lane(self):
        # Create a copy of the image, for the thresholding:
        lane_detect = np.copy(self.image)
        # Apply edge detection:
        canny_img = self.canny_method(lane_detect)
        cropped_img = self.triangle_view(canny_img)
        lines = cv2.HoughLinesP(
            cropped_img,
            2,
            np.pi / 180,
            100,
            np.array([]),
            minLineLength=5,
            maxLineGap=20,
        )
        mean_lines = self.average_slope_intercept(lane_detect, lines)
        masked_lane_detect = self.triangle_view(lane_detect)  # Mask the original image
        line_img = self.display_lines(
            masked_lane_detect, mean_lines
        )  # Draw the lines on the masked image
        threshold_img = cv2.addWeighted(lane_detect, 0.8, line_img, 1, 1)
        self.processed_image = threshold_img  # Add this line to store the result


    def show_result(self):
        if self.processed_image is not None:
            cv2.imshow("output", self.processed_image)
            cv2.waitKey(0)
        else:
            print("No processed image to display. Run detect_lane() first.")


if __name__ == "__main__":
    image_path = r"C:\Users\jacka\Documents\GitHub\py-punk\ocv-playground\inputs\test_frame.jpg"
    lane_detector = LaneDetector(image_path)
    lane_detector.detect_lane()
    lane_detector.show_result()


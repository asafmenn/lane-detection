# import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


def reject_outliers(data, m=2, min_data_length=4):

    if len(data) > min_data_length:
        data = np.array(data)

        # filter intercept outliers
        intercepts = data[:, 1]
        filter_data = data[abs(intercepts - np.mean(intercepts)) < m * np.std(intercepts)]

        # filter slope outliers
        slopes = filter_data[:, 0]
        filter_data = filter_data[abs(slopes - np.mean(slopes)) < m * np.std(slopes)]
        return filter_data.tolist()
    else:
        return data


def is_outlier(point, data, m=2, min_data_length=4):

    if len(data) <= min_data_length:
        return False

    data = np.array(data)

    slopes = data[:, 0]
    intercepts = data[:, 1]

    slope_diff = np.abs(point[0] - np.mean(slopes))
    intercept_diff = np.abs(point[1] - np.mean(intercepts))

    slope_std = m * np.std(slopes)
    intercept_std = m * np.std(intercepts)

    is_outlier = slope_diff >= slope_std or intercept_diff >= intercept_std

    return is_outlier




def get_mean_line_from_buffer(buffer, frames, y_min, y_max):

    # get the mean line from the frame buffer
    mean_line = np.mean(np.array(buffer[-frames:]), axis=0).tolist()
    mean_slope = mean_line[0]
    mean_intercept = mean_line[1]
    if(mean_slope==0):
        mean_slope=1
    # calculate the X coordinates of the line
    x1 = int((y_min - mean_intercept) / mean_slope)
    x2 = int((y_max - mean_intercept) / mean_slope)
    return x1, x2


# buffer cache through frames
line_low_avg_cache = []
line_high_avg_cache = []

# calculate max y coordinate for drawing and constructing the line
factor_height = 0.62



def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #fill the ROI
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # return the image where masked pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=8):

    global line_low_avg_cache
    global line_high_avg_cache

    # get min y and max y for drawing the line
    y_min = img_height = img.shape[0]
    y_max = int(img_height * factor_height)

    lines_left = []
    lines_right = []

    for line in lines:
        num_of_slopes = 0
        for x1, y1, x2, y2 in line:
            num_of_slopes = num_of_slopes + 1

            # calculate slope
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - x1 * slope
            # exclude non-valid slopes
            if (abs(slope) < 0.5 and abs(slope) > 0.8):
                lines_right.append([slope, intercept])
            if (math.isnan(slope) or math.isinf(slope)):
                continue
            # calculate intercept
            intercept = y1 - x1 * slope

            if (slope < 0):
                # it's right line
                lines_left.append([slope, intercept])
            # if(slope == np.nan):
                continue
            else:
                # it's left line
                lines_right.append([slope, intercept])


    # clean left lines from noise
    lines_low = reject_outliers(lines_right, m=1.7)

    # clean right lines from noise
    lines_high = reject_outliers(lines_left, m=1.7)

    # add left lines to the frame buffer only if they are not outliers inside
    if lines_high:
        for element in lines_high:
            if not is_outlier(element, line_high_avg_cache, m=2.6):
                line_high_avg_cache.append(element)

    # add right lines to the frame buffer only if they are not outliers inside
    if lines_low:
        for element in lines_low:
            if not is_outlier(element, line_low_avg_cache, m=2.6):
                line_low_avg_cache.append(element)

    if line_high_avg_cache:
        x1, x2 = get_mean_line_from_buffer(buffer=line_high_avg_cache,
                                           frames=20,
                                           y_min=y_min,
                                           y_max=y_max)
        # line extrapolation
        cv2.line(img, (x1, y_min), (x2, y_max), color, thickness)

    if line_low_avg_cache:
        x1, x2 = get_mean_line_from_buffer(buffer=line_low_avg_cache,
                                           frames=20,
                                           y_min=y_min,
                                           y_max=y_max)
        # line extrapolation
        cv2.line(img, (x1, y_min), (x2, y_max), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, first=1.5, sec=1, third=0.):

    return cv2.addWeighted(initial_img, first, img, sec, third)

low_threshold = 50
high_threshold = 150
kernel_size = 5

# hough transform params
rho = 1
theta = np.pi /180
threshold = 25
min_line_length = 5
max_line_gap = 40

# set mask scale factor
mask_scale_factor = 0.6
mask_width_factor = 0.5


def detect_segments(image):
    # get image shape
    img_width = image.shape[1]
    img_height = image.shape[0]

    # apply greyscale transform
    grayscale_transform_image = grayscale(image)

    # apply gausian transform
    gausian_transform_image = gaussian_blur(grayscale_transform_image, kernel_size)

    canny_transform_image = canny(gausian_transform_image, low_threshold, high_threshold)
    # get mask x, y coordinates
    mask_y = int(img_height * mask_scale_factor)
    mask_x = int(img_width * mask_width_factor)
    print(mask_y)
    print(mask_x)
    # mask vertices
    # vertices = np.array([[(0, img_height), (mask_x, mask_y), (mask_x, mask_y), (img_width, img_height)]])
    vertices = np.array([[(500, 370), (720, 370), (1200, 700), (60, 700)]])
    masked_edges = region_of_interest(canny_transform_image, vertices)
    lines_transform_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    result_image = weighted_img(image, lines_transform_image)
    return result_image



def clear_cache():
    line_low_avg_cache.clear()
    line_high_avg_cache.clear()

def process_image(image):
    result = detect_segments(image)
    return result

cap = cv2.VideoCapture('camvideo.mp4')

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', process_image(frame))
    key = cv2.waitKey(5)

cap.release()
cv2.destroyAllWindows()
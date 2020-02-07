import argparse
import time
import numpy as np
from collections import defaultdict
from scipy import stats
import cv2
import sys


def cartoonize(image):
    """
    convert image into cartoon-like image
    image: input PIL image
    """

    output = np.array(image)
    x, y, c = output.shape
    
    for i in xrange(c):
        output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 5, 50, 50)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
    cv2.imshow("grayscale",gray)
    cv2.waitKey(0)
    
    #canny_original
    edge = cv2.Canny(output, 100, 200)
    cv2.imshow("canny_original",edge)
    cv2.waitKey(0)
    cv2.imwrite("Edge_Canny_"+file_name, cv2.bitwise_not(edge))

    #canny_gray
    edge_gray = cv2.Canny(gray, 100, 200)
    cv2.imshow("canny_gray",edge_gray)
    cv2.waitKey(0)

    #canny_gray_noise_removed
    edge_gray_nr = cv2.Canny(img_gaussian, 115, 220)
    cv2.imshow("canny_gray_noise_removed",edge_gray_nr)
    cv2.waitKey(0)

    edge_gray_nr_invert= cv2.bitwise_not(edge_gray_nr)
    cv2.imwrite("Edge_gray_nr_"+file_name, edge_gray_nr_invert)
    cv2.imshow("edge_gray_nr_invert",edge_gray_nr_invert)
    cv2.waitKey(0)

    #sobel
    img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=3)
    img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=3)
    img_sobel = img_sobelx + img_sobely
    cv2.imshow("sobel",img_sobel)
    cv2.waitKey(0)

    #laplacian
    laplacian = cv2.Laplacian(img_gaussian,cv2.CV_8U)
    cv2.imshow("laplacian",laplacian)
    cv2.waitKey(0)

    #Adaptive Threshold
    gray_blur = cv2.medianBlur(gray, 5)
    ATedges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    cv2.imshow("adaptiveThreshold",ATedges)
    cv2.waitKey(0)
    cv2.imwrite("Edge_Adaptive_"+file_name, ATedges)

    output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)

    hists = []
    #H
    hist, _ = np.histogram(output[:, :, 0], bins=np.arange(180+1))
    hists.append(hist)
    #S
    hist, _ = np.histogram(output[:, :, 1], bins=np.arange(256+1))
    hists.append(hist)
    #V
    hist, _ = np.histogram(output[:, :, 2], bins=np.arange(256+1))
    hists.append(hist)

    C = []
    for h in hists:
        C.append(choose_k(h))
    #print("centroids: {0}".format(C))

    output = output.reshape((-1, c))
    for i in xrange(c):
        channel = output[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - C[i]), axis=1)
        output[:, i] = C[i][index]
    output = output.reshape((x, y, c))
    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

    #This is the colour smoothened output
    cv2.imshow("smooth",output)
    cv2.waitKey(0)
    cv2.imwrite("Smooth_K_Means"+file_name, output)

    #Edges using adaptive threshold
    cartoonAT = cv2.bitwise_and(output, output, mask=ATedges)
    cv2.imshow("cartoonAT", cartoonAT)
    cv2.waitKey(0)

    #Edges using canny_gray
    edge_gray_invert= cv2.bitwise_not(edge_gray)
    cartoon_canny_gray = cv2.bitwise_and(output, output, mask=edge_gray_invert)
    cv2.imwrite("cartoon_canny_gray"+file_name, cartoon_canny_gray)
    cv2.imshow("cartoon_canny_gray", cartoon_canny_gray)
    cv2.waitKey(0)

    #Edges using canny_gray_nr
    cartoon_canny_gray_nr = cv2.bitwise_and(output, output, mask=edge_gray_nr_invert)
    cv2.imwrite("cartoon_canny_gray_nr"+file_name, cartoon_canny_gray_nr)
    cv2.imshow("cartoon_canny_gray_nr", cartoon_canny_gray_nr)
    cv2.waitKey(0)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cartoonAT


def find_centroids(centroids, histogram):
    """
    Keep on updating centroids until centroids do not change
    """
    new_centroids = np.array([])
    while (not np.array_equal(centroids, new_centroids)):
        length = len(histogram)
        i = 0
        assigned_pixels = defaultdict(list)
        if (not np.array_equal(new_centroids, np.array([]))):
            centroids = new_centroids
        while (i < length):
            if (histogram[i] != 0):
                distance = centroids - i
                assigned_centroid = np.argmin(np.abs(distance))
                assigned_pixels[assigned_centroid].append(i)
            i = i + 1
        new_centroids = np.array(centroids)
        for index, pixels in assigned_pixels.items():
            if (np.sum(histogram[pixels]) != 0):
                pixels_sum = np.sum(pixels * histogram[pixels])
                pixels_centroid = int(pixels_sum/np.sum(histogram[pixels]))
                new_centroids[index] = pixels_centroid
    return centroids, assigned_pixels


def choose_k(histogram):
    """
    Choose the best K for k-means and get the centroids
    """
    alpha = 0.01     # p-value threshold for normaltest
    min_group_size = 20 # minimum group size for normaltest
    centroids = np.array([128])

    new_centroids = set()

    while (len(new_centroids) != len(centroids)):
        if (not (new_centroids == set())):
            centroids = np.array(sorted(new_centroids))

        centroids, assigned_pixels = find_centroids(centroids, histogram)
        new_centroids = set() # set will avoid adding the same centroid value which already exists

        for index, pixels in assigned_pixels.items():
            num_pixels = len(pixels)

            # if number of pixels assigned to the centroid are more than minimum size, try to separate centroid
            if (num_pixels > min_group_size):
                statistic, pvalue = stats.normaltest(histogram[pixels])
                
                # This is the case when the group does not follow normal distribution --> separate
                if (pvalue < alpha):

                    # left and right help to determine the new centroids into which the original
                    # centroid will be broken into
                    if (index == 0):
                        left = 0
                        if (len(centroids) > 1):
                            right = centroids[index + 1]
                        else:
                            right = len(histogram) - 1
                    elif (index == len(centroids) - 1):
                        left = centroids[index - 1]
                        right = len(histogram) - 1
                    else:
                        left = centroids[index - 1]
                        right = centroids[index + 1]


                    difference = right - left
                    if difference >= 3:
                        # this condition ensures that at least one of the two centroids to be added to the
                        # list of centroids is different from the original centroid
                        new_centroids.add((centroids[index] + left)/2)
                        new_centroids.add((centroids[index] + right)/2)
                    else:
                        new_centroids.add(centroids[index])
                else:
                    # if not enough elements in the group, no need to separate the centroid
                    new_centroids.add(centroids[index])
            else:
                # This is the case when the group follows normal distribution, no need to separate
                new_centroids.add(centroids[index])
    return centroids

file_name = sys.argv[1] #File_name will come here
print(file_name)

img = cv2.imread(file_name)
output = cartoonize(img)
cv2.imshow("original",img)
cv2.waitKey(0)

cv2.imwrite("Cartoon_"+file_name, output)
cv2.imshow("Cartoon version", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
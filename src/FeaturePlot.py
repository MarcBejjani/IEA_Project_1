from matplotlib import pyplot as plt


def profilePlot (x_axis, y_axis, bottom, top, left, right):
    plt.subplot(2, 2, 1)
    plt.bar(x_axis, bottom, color='red', width=0.5)
    plt.title("bottom profile")

    plt.subplot(2, 2, 2)
    plt.bar(x_axis, top, color='red', width=0.5)
    plt.title("top profile")

    plt.subplot(2, 2, 3)
    plt.bar(y_axis, right, color='red', width=0.5)
    plt.title("right profile")

    plt.subplot(2, 2, 4)
    plt.bar(y_axis, left, color='red', width=0.5)
    plt.title("left profile")
    plt.show()

def projectionHistogramPlot(image, x_axis, column_sum, row_sum):
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    # cv2.imshow("final image", image)

    plt.subplot(2, 2, 3)
    plt.bar(x_axis, column_sum, color='red', width=0.4)
    plt.title("Vertical projection histogram")

    plt.subplot(2,2,4)
    plt.bar(x_axis, row_sum, color='red', width=0.4)
    plt.title("Horizental projection histogram")
    plt.show()

def hogPlot(hog_image):
    plt.axis("off")
    plt.imshow(hog_image, cmap="gray")
    plt.show()


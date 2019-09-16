import cv2

input = cv2.imread('../DATA/bricks.jpg')

cv2.imshow("First Output", input)
 
#image shape
print(input.shape)

#width of the image
print("Height is: ",input.shape[0], " pixels")

#width of the image
print("Width is: ",input.shape[1], " pixels")


cv2.waitKey()

cv2.destroyAllWindows()


# Simply use 'imwrite' specificing the file name and the image to be saved
#imwrite() is used to save an image, 
#1st parameter = file name, 2nd parameter = image to be saved
cv2.imwrite('output1.jpg', input)
cv2.imwrite('output2.png', input)
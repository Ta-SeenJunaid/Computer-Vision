
#importing opencv
import cv2

#imread() is used to load images
#We need to be carefull about path setup
#If it can not load a image due to path, it'll take null and so you can not see 
#any error message
input = cv2.imread('../DATA/bricks.jpg')

#imshow() is used for output an image, 
#The first parameter will be title, The second parameter is the image varialbe
cv2.imshow("First Output", input)

#waitKey() = input information when a image window is open
#blank parameter = waits for anykey to be pressed before continuing
# By placing milliseconds numbers, we can specify abouthow long we keep the window open 
cv2.waitKey()

# destroys all open windows 
cv2.destroyAllWindows()
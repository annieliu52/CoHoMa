from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2

def decode(im) :
  # Find barcodes and QR codes
  decodedObjects = pyzbar.decode(im)

  # Print results
  for obj in decodedObjects:
    if str(obj.type) == 'QRCODE' :
      print('Type : ', obj.type)
      print('Data : ', str(obj.data),'\n')

  return decodedObjects


# Display barcode and QR code location
def display(im, decodedObjects, writer):

  # Loop over all decoded objects
  for decodedObject in decodedObjects:
    if str(decodedObject.type) == 'QRCODE' :
      points = decodedObject.polygon
      
      # If the points do not form a quad, find convex hull
      if len(points) > 4 :
        hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
        hull = list(map(tuple, np.squeeze(hull)))
      else :
        hull = points;

    # Number of points in the convex hull
    n = len(hull)

    # Draw the convext hull
    for j in range(0,n):
      cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)

  # Display results
  cv2.imshow("Results", im);
  writer.write(im)
  #cv2.waitKey(0);


# Main
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer= cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))

    while True:
        

        # Read image
        _, im = cap.read()
        
        decodedObjects = decode(im)
        display(im, decodedObjects, writer)

        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
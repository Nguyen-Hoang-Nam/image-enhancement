import os
import argparse
import cv2 as cv

import contrast_image as ci 

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--test", action = "store", type=str, help = "Test all images in test directory")

  args = parser.parse_args()

  if args.test:
    test_directory = "test"
    result_directory = args.test
    testcase_list = os.listdir(test_directory)

    for testcase in testcase_list:
      testcase_path = test_directory + '/' + testcase

      image = cv.imread(testcase_path)

      if args.test == "THE":
        result = ci.THE(image)
      elif args.test == "BBHE":
        result = ci.BBHE(image)
      elif args.test == "DSIHE":
        result = ci.DSIHE(image)
      elif args.test == "MMBEBHE":
        result = ci.MMBEBHE(image)
      elif args.test == "RMSHE":
        result = ci.RMSHE(image, 2)

      cv.imwrite(result_directory + '/' + testcase, result)

if __name__ == '__main__':
  main()
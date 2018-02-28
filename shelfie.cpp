//
//  shelfie.cpp
//  Opencv Test
//
//  Created by Jason Hogan on 08/02/2017.
//  Copyright © 2017 Jason Hogan. All rights reserved.
//

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <string>

#pragma clang pop

using namespace std;
using namespace cv;

/*  Global variables for colours in RGB colour space.  */
Vec3b white = { 255, 255, 255 };
Vec3b blue = { 220, 80, 0 };
Vec3b red = { 0, 0, 255 };
Vec3b black = { 0, 0, 0 };

/* Other global variables */
int numberOfShelves;
int screenHeight;
int screenWidth;
Mat spine;
float bookCoords[5] = {-1, -1, -1, -1, -1};  // Book coordinates initialised with error values.


/**
 *  Function to return the location of the most recent book to be searched for.
 *  @return x and y coordinates of book in image.
 */
static float * bookLocation(){
    return bookCoords;
}


/**
 *  Function to carry out template matching in bookshelf scene using book spine as template.
 *
 *  @params Template image and target image
 *  @return Tuple containing location, size, and confidence of best match
 */
static tuple<double, int, int, cv::Point> templateMatchSpine(Mat bookshelf, Mat greySpine){
    // Initialise variables
    double maxConfidence = 0;
    cv::Point maxConfidenceLocation;
    int maxConfidenceWidth = 0;
    int maxConfidenceHeight = 0;
    Mat templateCopy = greySpine.clone();
    Mat bookshelfCopy = bookshelf.clone();
    
    // Scale shelf to a height of 300 pixels in order to speed up template matching.
    double scaleRatio = 300.0/(double)bookshelfCopy.rows;
    resize(bookshelf, bookshelfCopy, cv::Size(bookshelfCopy.cols*scaleRatio, bookshelfCopy.rows*scaleRatio));
    
    // Scale template to intial height, which is same height as shelf.
    double templateAspectRatio = ((double)greySpine.cols)/((double)greySpine.rows);
    double widthScale = bookshelfCopy.rows * templateAspectRatio;
    double heightScale = bookshelfCopy.rows;
    resize(templateCopy, templateCopy, cv::Size(widthScale, heightScale));
    
    // If shelf is smaller than template image after scaling then shelf is an error case so return all zeros.
    if(templateCopy.cols > bookshelfCopy.cols){
        return make_tuple(0,0,0,cv::Point(0,0));
    }
    
    // Search loop. This is repeated for each scale of the template
    for(;templateCopy.rows > bookshelfCopy.rows/2;){
        // Local variables to house template matching result data.
        Mat templateMatchResult;
        double minVal, maxVal;
        cv::Point minLocation, maxLocation;
        
        // Carry out normalised correlation coefficient template matching using OpenCV function.
        matchTemplate(bookshelfCopy, templateCopy, templateMatchResult, CV_TM_CCOEFF_NORMED);
        
        // Find min and max location values from template match result.
        minMaxLoc(templateMatchResult, &minVal, &maxVal, &minLocation, &maxLocation, Mat());
        
        // Update max confidence location if current result has a higher confidence than it.
        if (maxVal > maxConfidence){
            maxConfidence = maxVal;
            maxConfidenceLocation = cv::Point(maxLocation.x/scaleRatio, maxLocation.y/scaleRatio);
            maxConfidenceWidth = templateCopy.cols/scaleRatio;
            maxConfidenceHeight = templateCopy.rows/scaleRatio; // '/scaleRatio' = Undo coordinate changes brought about by scaling shelf
        }
        
        // Scale template down to 95% of its previous size before next search iteration.
        widthScale *= 0.95;
        heightScale *= 0.95;
        resize(templateCopy, templateCopy, cv::Size(widthScale, heightScale));
    }
    
    // Return tuple containing confidence, location, and scale of best match.
    return make_tuple(maxConfidence, maxConfidenceWidth, maxConfidenceHeight, maxConfidenceLocation);
}


/**
 *  Method for non-maxima suppression of a gradients edge image.
 *
 *  This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 *  by Kenneth Dawson-Howe, Wiley & Sons Inc. 2014. All rights reserved.
 */
static void NonMaximaEdgeGradientSuppression(Mat & gradients, Mat & orientations, Mat & nms_result){
    // Initialise variables.
    float min_gradient = 50.0;
    nms_result = gradients.clone();
    int image_channels = gradients.channels();
    int image_rows = gradients.rows;
    int values_on_each_row = (gradients.cols-2) * image_channels;
    int max_row = image_rows-1;
    
    for (int row=1; row < max_row-1; row++){
        float* curr_gradient = gradients.ptr<float>(row) + image_channels;
        float* curr_orientation = orientations.ptr<float>(row) + image_channels;
        float* output_point = nms_result.ptr<float>(row);
        *output_point = 0.0;
        output_point += image_channels;
        for (int column=0; column < values_on_each_row; column++){
            // If gradient is less than minimum, then suppress.
            if (*curr_gradient < min_gradient)
                *output_point = 0.0;
                else{
                    // Otherwise find two adjacent gradients based on orientation.
                    int direction = (((int) (16.0*(*curr_orientation)/(2.0*CV_PI))+15)%8)/2;
                    float gradient1 = 0.0, gradient2 = 0.0;
                    switch(direction){
                        case 0:
                            gradient1 = *(gradients.ptr<float>(row-1) + (column)*image_channels);
                            gradient2 = *(gradients.ptr<float>(row+1) + (column+2)*image_channels);
                            break;
                        case 1:
                            gradient1 = *(gradients.ptr<float>(row-1) + (column+1)*image_channels);
                            gradient2 = *(gradients.ptr<float>(row+1) + (column+1)*image_channels);
                            break;
                        case 2:
                            gradient1 = *(gradients.ptr<float>(row-1) + (column+2)*image_channels);
                            gradient2 = *(gradients.ptr<float>(row+1) + (column)*image_channels);
                            break;
                        case 3:
                            gradient1 = *(curr_gradient - image_channels);
                            gradient2 = *(curr_gradient + image_channels);
                            break;
                    }
                    
                    // If either adjacent gradient is greater than current gradient, then suppress current.
                    if ((gradient1 > *curr_gradient) || (gradient2 > *curr_gradient))
                        *output_point = 0.0;
                }
            curr_gradient += image_channels;
            curr_orientation += image_channels;
            output_point += image_channels;
        }
        *output_point = 0.0;
    }
}


/**
 *  Function for finding edges using Sobel first derivative edge detection.
 *
 *  @params Image on which edge detection should be carried out
 *  @return Binary edge image
 */
static Mat firstDerivEdges(Mat image) {
    // Initialise local variables
    Mat result, horizontal, vertical, gradients, orientation;
    result = image.clone();
    
    // Carry out Sobel edge detection in two orthogonal directions and create edge gradients and orientations.
    Sobel(result, horizontal, CV_32F,1,0);
    Sobel(result, vertical, CV_32F,0,1);
    cartToPolar(horizontal, vertical, gradients, orientation);
    
    // Do non-maxima suppression on edge gradients.
    NonMaximaEdgeGradientSuppression(gradients, orientation, result);
    result.convertTo(result, CV_8U);
    
    // Threshold to create binary edge image.
    threshold(result, result, 0, 255, THRESH_OTSU);
    return result;
}


/**
 *  Function to crop barren regions from arond the borders of an image of a bookshelf.
 *
 *  @params Bookshelf image
 *  @return Tuple of cropped image, and how many pixels were cropped off the left hand side
 */
static tuple<Rect, int> cropShelf(Mat shelf){
    // Initialise three equally spaced prongs for testing each side of image
    int y = 2 * (shelf.rows/8);
    int y2 = 4 * (shelf.rows/8);
    int y3 = 6 * (shelf.rows/8);
    
    // Crop left hand side.
    int left = 0;
    while(shelf.at<uchar>(y,left) == 0 && shelf.at<uchar>(y2,left) == 0 && shelf.at<uchar>(y3,left) == 0 && left < shelf.cols/2){
        left++; // Move inwards until an edge pixel is hit
    }
    if(left > 20) left -= 20; // Step back by 20 pixels, to ensure no part of the book was cropped out
        
        // Crop right hand side.
        int right = shelf.cols - 1;
        while(shelf.at<uchar>(y, right) == 0 && shelf.at<uchar>(y2, right) == 0 && shelf.at<uchar>(y3, right) == 0 && right > shelf.cols/2){
            right--; // Move inwards until an edge pixel is hit
        }
    if(right < shelf.cols - 20) right += 20; // Step back by 20 pixels, to ensure no part of the book was cropped out
        
        // Create new image of shelf, without the barren regions, and return it, along with metadata.
        cv::Rect ROI = cv::Rect(left, 0, (right-left), shelf.rows);
        return make_tuple(ROI,left);
}


/**
 *  Function to find the vertical or horizontal distance between two points.
 *
 *  @params Vertical = point one, point two
 *          Horizontal = point two, point one
 *  @return Distance between points
 */
static double distance(Point one, Point two){
    // Distance formula.
    double result = ((one.x-two.x)^2) + ((one.y-two.y)^2);
    return sqrt(result);
}


/**
 *  Function to mask horizontal edge pixels out of an edge image, leaving only the vertical edges.
 *
 *  @params Edge image
 *  @return Vertical edge image
 */
static Mat verticalEdges(Mat image){
    // Initialise local variables
    vector<Vec4i> hough_line_segments;
    Mat verticalHoughLines;
    
    // Probabilistic Hough transform used to find straight line segments
    HoughLinesP(image, hough_line_segments, 1.0, CV_PI/2, 10, 0, 10);
    
    cvtColor(image, verticalHoughLines, CV_GRAY2BGR);
    
    // Draw all vertical Hough lines in red over the edge image
    for (vector<cv::Vec4i>::const_iterator current_line = hough_line_segments.begin();
         (current_line != hough_line_segments.end()); current_line++) {
        cv::Point point1((*current_line)[0],(*current_line)[1]);
        cv::Point point2((*current_line)[2],(*current_line)[3]);
        // If line has a vertical distance based on distance function, then line is vertical.
        if(distance(point1, point2) > 1)
            line(verticalHoughLines, point1, point2, red, 2);
    }
    
    // Suppress any pixel that is not part of a vertical Hough line segment.
    for(int i = 0; i < verticalHoughLines.rows; i++) {
        for(int j = 0; j < verticalHoughLines.cols; j++) {
            if(verticalHoughLines.at<Vec3b>(i,j) == red)
                verticalHoughLines.at<Vec3b>(i,j) = white;
                else verticalHoughLines.at<Vec3b>(i,j) = black;
        }
    }
    
    // Convert to binary edge image and return
    cvtColor(verticalHoughLines, verticalHoughLines, CV_BGR2GRAY);
    return verticalHoughLines;
}


/**
 *  Function to find location of bookshelves in an edge image of a bookcase.
 *
 *  @params Edge image, original image
 *  @return Vector of tuples containing location of each shelf, and the number of pixels cropped.
 */
static vector<tuple<Mat,int,int>> findShelves(Mat edgeImage, Mat originalImage){
    // Remove horizontal edges from edge image.
    Mat shelvesImage = verticalEdges(edgeImage.clone());
    
    // Initialise local variables
    int shelves[500] = { 0 };
    int edgePoints = 0;
    int cols = shelvesImage.cols;
    int rows = shelvesImage.rows;
    
    // Find total number of edge pixels in image.
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
            if(shelvesImage.at<uchar>(i,j) > 0)
                edgePoints++;
    
    // Calculate average number of edge pixels per row.
    int averageEdgePoints = edgePoints/rows;
    int index = 0; // index variable for accessing "shelves" array.
    
    
    /* Find Top Shelf */
    int topShelf = 0;
    bool topShelfFound = false;
    // Move down from the top of the image, checking every 10th row.
    for(int i = 0; i < rows && !topShelfFound; i += 10){
        edgePoints = 0;
        // Find number of edge pixels in this row.
        for(int j = 0; j < cols; j++){
            if(shelvesImage.at<uchar>(i,j) > 0)
                edgePoints++;
        }
        // If there are enough edge pixels in this row, it's likely we have hit the top of the first row of books.
        if(edgePoints > (averageEdgePoints/5)) {
            shelves[index++] = (i > 25) ? i - 25 : i;   // Back up by a buffer of 25 pixels to ensure no book data is lost.
            topShelf = (i > 25) ? i - 25 : i;           // Set top shelf variable.
            topShelfFound = true;                        // Set condition to break out of for loop.
        }
    }
    
    /* Find Middle Shelves */
    for(int i = shelves[0]; i < rows; i += 10){
        edgePoints = 0;
        // Find number of edge pixels in this row.
        for(int j = 0; j < cols; j++){
            if(shelvesImage.at<uchar>(i,j) > 0)
                edgePoints++;
        }
        
        // If significantly fewer edge pixels than average, add to list of shelves.
        if(edgePoints < (averageEdgePoints/5))
            shelves[index++] = i;
    }
    
    index--;    // Undo the final '++' so that index points to the most recently assigned element in shelves array
    
    /* Find Bottom Shelf */
    bool reachedBottomShelf = false;
    // Start from bottom of image
    int bottomShelf = rows;
    // Move upwards 10 pixels at a time.
    for(int i = rows; i > 0 && !reachedBottomShelf; i -= 10){
        int edgePoints = 0;
        // Find number of edge pixels in this row.
        for(int j = 0; j < cols; j++){
            if(shelvesImage.at<uchar>(i,j) > 0)
                edgePoints++;
        }
        // If a row of books is hit...
        if(edgePoints > (averageEdgePoints/5)) {
            bottomShelf = (i < rows - 25) ? i + 25 : i; // Back up by 25 pixels to ensure no book data is lost.
            reachedBottomShelf = true;                  // And break out of loop.
        }
    }
    
    int actualShelves[20];
    int numShelves = 0;
    reachedBottomShelf = false;
    
    // Narrow down list of potential shelves.
    for(int i = 0; i < index + 1 && !reachedBottomShelf; i++){
        if(abs(shelves[i+1] - shelves[i]) < (rows/8))
            shelves[i+1] = shelves[i]; // Remove potential shelf locations that are too close to one another.
            else if(shelves[i] >= bottomShelf){     // If bottom shelf has been reached.
                actualShelves[numShelves] = bottomShelf;    // Next shelf in list = bottom shelf.
                reachedBottomShelf = true;                  // And break out of loop.
            }
            // If current shelf location is far away from previous one, set as next actual shelf location.
            else if(shelves[i] > rows/8)
                actualShelves[numShelves++] = shelves[i];
    }
    
    // List of tuples instanciated to contain list of shelves + metadata
    vector<tuple<Mat,int,int>> result(numShelves);
    
    // For each shelf..
    for(int i = 0; i < numShelves; i++){
        // Initialise variable to contain location of the top of this shelf
        int top = (i == 0) ? topShelf : actualShelves[i-1];
        if(actualShelves[i] == top) {
            if(i + 1 < numShelves) i++;     // If there are more shelves, move to next shelf.
            else{
                // If there is only one shelf location, at top of image, then no shelves have been found.
                numberOfShelves = 0;
                return result;
            }
        }
        
        // Create new sub-region of image to contain only the current shelf.
        cv::Rect ROI = cv::Rect(0, top, cols, actualShelves[i] - top);
        
        // Crop current shelf
        tuple<cv::Rect, int> crop = cropShelf(edgeImage(ROI));
        
        // Add shelf image and corresponding metadata to result vector
        ROI = get<0>(crop);
        ROI.y += top;
        result[i] = make_tuple(originalImage(ROI), get<1>(crop), topShelf);
    }
    
    // Save number of shelves in global variable, and return list of shelves + their metadata.
    numberOfShelves = numShelves;
    return result;
}


/**
 *  Function to compare colours in two images.
 *
 *  @params Tentative result image, template image
 *  @return Colour comparison score.
 */
static double compareColourHistograms(Mat book, Mat templateImage){
    // Initialise CLAHE and set clip limit
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(2);
    
    // Convert images into HLS colour space
    cvtColor(templateImage, templateImage, CV_RGB2HLS);
    cvtColor(book, book, CV_BGR2HLS);
    vector<Mat> hls, hls2;
    
    split(book, hls);               // Split image into channels
    clahe->apply(hls[1],hls[1]);    // Apply CLAHE to the luminance channel
    merge(hls, book);               // Merge channels again
    
    split(templateImage, hls2);     // Split image into channels
    clahe->setClipLimit(1);         // Change to different clip limit for scanned book spine image
    clahe->apply(hls2[1],hls2[1]);  // Apply CLAHE to the luminance channel
    merge(hls2, templateImage);     // Merge channels again
    
    // Convert back into RGB colour space
    cvtColor(book, book, CV_HLS2BGR);
    cvtColor(templateImage, templateImage, CV_HLS2BGR);
    
    // Initialise variables for creating colour histograms
    MatND hist1, hist2;
    const int channels = book.channels();
    int * bins = new int[channels];
    int channel_nums[] = {0, 1, 2};
    float channel_range[] = {0.0, 255.0};
    const float * channel_ranges[] = {channel_range, channel_range, channel_range};
    
    // Set number of histogram bins to 8.
    for (int i = 0; i < channels; i++)
        bins[i] = 8;
        
        // Create colour histograms with the above settings.
        calcHist(&book, 1, channel_nums, Mat(), hist1, channels, bins, channel_ranges, true, false);
        normalize(hist1, hist1, 1);
        
        calcHist(&templateImage, 1, channel_nums, Mat(), hist2, channels, bins, channel_ranges, true, false);
        normalize(hist2, hist2, 1);
        
        // Return colour histogram comparison correlation value.
        return compareHist(hist1, hist2, CV_COMP_CORREL);
}


/**
 *  Function to validate result using classification function.
 *
 *  @params The two matching scores
 *  @return True if result is classified as correct, false if not
 */
static bool validate(double x, double y){
    double d = (x*-0.27) - ((y-0.27)*1.19); // -0.27, 1.19 = variables which represent equation of classification function.
    if(d < 0) return true;      // If d < 0, then result is above classification line, so result is classified as correct.
    else return false;          // Otherwise result classified as incorrect.
}


/**
 *  Function to find a book in an image of a bookshelf.
 *
 *  @params Image of bookshelf, orientation of bookshelf image, dimensions of device screen
 *  @return Image showing location of book
 */
static void findBook(Mat bookshelf, int orientation){
    // Save a full resolution copy of the image.
    Mat bookshelfFullRes = bookshelf.clone();
    
    // Scale image down to a height of 2500 pixels.
    double scaleRatio;
    if(orientation == 1) scaleRatio = (double)2500/(double)bookshelf.rows;
        else scaleRatio = (double)2500/(double)bookshelf.cols;
            resize(bookshelf, bookshelf, cv::Size(bookshelf.cols*scaleRatio, bookshelf.rows*scaleRatio));
            
            // Split bookshelf image into three single channel images - red, green, blue.
            vector<Mat> bgr;
    split(bookshelf, bgr);
    
    // Carry out edge detection on each colour channel.
    Mat edgesB = firstDerivEdges(bgr[0]);
    Mat edgesG = firstDerivEdges(bgr[1]);
    Mat edgesR = firstDerivEdges(bgr[2]);
    
    // Merge resulting channel edge images into one aggregate edge image.
    Mat edges = edgesB | edgesG;
    edges = edges | edgesR;
    
    // Find shelves in edge image
    vector<tuple<Mat,int,int>> shelfTuples = findShelves(edges, bookshelf);
    vector<tuple<Mat,int>> shelves(numberOfShelves);
    
    // Return an error if no shelves are found
    if(numberOfShelves == 0) {
        printf("Error - no shelves found");
        exit(0);
    }
    
    // Initialise CLAHE and set low clip limit
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(1);
    int index = 0;
    
    // Convert each shelf to greyscale and equalise using CLAHE, in preparation for template matching.
    for(int i = 0; i < numberOfShelves; i++){
        if(get<0>(shelfTuples[i]).data != NULL){
            cvtColor(get<0>(shelfTuples[i]), get<0>(shelves[index]), CV_BGR2GRAY);
            clahe->apply(get<0>(shelves[index]), get<0>(shelves[index]));
            get<1>(shelves[index]) = get<1>(shelfTuples[i]);    // Get metadata for shelf.
            index++;
        }
    }
    
    // Convert book spine template image to greyscale and blur for template matching.
    Mat greySpine;
    cvtColor(spine, greySpine, CV_BGR2GRAY);
    blur(greySpine, greySpine, cv::Size(3,3));
    
    // Initialise variables
    int shelfHeights = 0;
    double maxConfidence = 0;
    Point maxConfidenceLoc;
    int maxLocW = 0, maxLocH = 0;
    int cropHeight = get<2>(shelfTuples[numberOfShelves - index]);
    
    numberOfShelves = index;
    
    // Start clock to time search.
    unsigned long start  = clock();
    
    // For each shelf dispatch a new thread to carry out a search for the book.
    for(int i = 0; i < numberOfShelves; i++){
        Mat greySpineLocal = greySpine.clone();
        
        try{
            // Carry out template matching on shelf
            tuple<double, int, int, cv::Point> match = templateMatchSpine(get<0>(shelves[i]), greySpineLocal);
            
            // Update global best match if necessary
            if(get<0>(match) > maxConfidence){
                maxConfidence = get<0>(match);
                maxLocW = get<1>(match);
                maxLocH = get<2>(match);
                // Adjust coordinates to allow for cropping.
                maxConfidenceLoc.x = get<3>(match).x + get<1>(shelves[i]);
                maxConfidenceLoc.y = get<3>(match).y + cropHeight;
                shelfHeights = 0;
                for(__block long k = i; k > 0; k--) shelfHeights += get<0>(shelves[k-1]).rows;
            }
        } catch(Exception * e) {
            // If template matching raises an ecxeption, then image was not searchable
            printf("Error");
            exit(0);
        }
    }
    
    // Get region of original image returned as location of best match
    maxConfidenceLoc.y += shelfHeights;
    cv::Point bottomRightCorner = cv::Point(maxConfidenceLoc.x + maxLocW - 1, maxConfidenceLoc.y + maxLocH - 1);
    cv::Rect ROI = cv::Rect(maxConfidenceLoc.x/scaleRatio, maxConfidenceLoc.y/scaleRatio, maxLocW/scaleRatio, maxLocH/scaleRatio);
    
    // Create new image of the best match region, in full resolution
    Mat foundBook = bookshelfFullRes(ROI).clone();
    
    // Resize spine template image to be same size as book found in search.
    Mat spineCopy;
    double spineScale = (double)foundBook.rows/(double)spine.rows;
    resize(spine, spineCopy, cv::Size(spine.cols*spineScale, spine.rows*spineScale));
    
    // Compare colours in the search result and the spine template image.
    double colourComparison = compareColourHistograms(foundBook, spineCopy);
    
    // Template match + colour comparison are done now, so stop timer.
    unsigned long end = clock();
    
    printf("Time Taken = %f seconds\n", (end-start)/(double)CLOCKS_PER_SEC);
    printf("Template Match Confidence = %f\n", maxConfidence);
    printf("Colour comparison = %f\n\n", colourComparison);
    
    // If result fails classification, return nil
    if(validate(maxConfidence, colourComparison) == false) {
        printf("Book not found");
    }
    
    // Draw green line around outline of book in image
    Mat result = bookshelfFullRes.clone();
    rectangle(result, cv::Point(maxConfidenceLoc.x/scaleRatio, maxConfidenceLoc.y/scaleRatio), cv::Point(bottomRightCorner.x/scaleRatio, bottomRightCorner.y/scaleRatio), CV_RGB(0,255,0), 8);
    
}


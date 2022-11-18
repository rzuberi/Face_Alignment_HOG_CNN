Training a CNN with HOGs to perform face alignment and using the predictions to colour the lips of faces with colour-maps realistically

\title{
Introduction
}

Face alignment outlines the landmarks of a face on an image. The landmarks can then be extracted for modification, such as applying lipstick. We implemented a model that collects the Histogram of Oriented Gradients (HOG) of images of single faces and uses them to train a Convolutional Neural Network (CNN) to predict the face alignment points. We then implemented a script that uses the lip alignment coordinates to realistically change its colour by applying a colour-map. Our HOG-CNN model produces satisfying face alignment results and our lipstick script makes mostly realistic changes to the face but both perform poorly on obstructed faces and dimmer images.

This report will first present our face alignment model, the experimentations that justify its architecture, and its quantitative and qualitative performance against other models. The second part will present our lip-colouring script's method and its qualitative results.

Face alignment with HOG images and a CNN
A. Method overview

\section{HOG CNN face alignment model training flowchart}

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-01.jpg?height=399&width=1485&top_left_y=1134&top_left_x=317)

Figure 1. The flowchart of going from our training data to evaluating our face alignment predictions model.

We start by augmenting our training data and processing it to collect the HOGs. We evaluate our model with a part of the training data we did not train our CNN on. Our model is evaluated by the average Euclidean distance between the ground truth points provided in the original training set and the predicted points.

\section{B. Data augmentation}

\section{Data augmentation demonstration}

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-01.jpg?height=301&width=1489&top_left_y=1931&top_left_x=318)

Figure 2. Summary of the effects of possible transformations to augment the training dataset.

We augmented our data by rotating duplicated images but only to realistic extents. We also added random noise to duplicates as we found that it created different HOGs. We excluded some data augmentation techniques (e.g. brightness changes) as they did not change the collected features and including them risked overfitting our CNN. We also excluded flipping the images since the ground truth points are not in a symmetrical order. Cumulative density plots of euclidean distance on evaluation data CNN trained on augmented data vs. CNN trained on non-augmented data

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-02.jpg?height=534&width=808&top_left_y=308&top_left_x=301)

Boxplot of euclidean distance on evaluation data from CNN trained

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-02.jpg?height=665&width=659&top_left_y=215&top_left_x=1248)

Figure 3. Comparing two models trained on different data: one with the base training data plus the augmented data, and one with only the base training data.

Training our model with augmented data makes it more accurate and lowers its average Euclidean distance, but it does not let it generalise as well as its 1st and 3rd quartiles are further apart than the non-augmented data. Since it has more data to train on, it needs more training time to find generalising patterns.

\section{Preprocessing images}

The original image compared to its collected HOG features on its $122 \times 122$ resized format with different amounts of pixels per cell

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-02.jpg?height=502&width=507&top_left_y=1348&top_left_x=316)

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-02.jpg?height=246&width=208&top_left_y=1346&top_left_x=861)

$10 \times 10$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-02.jpg?height=203&width=204&top_left_y=1644&top_left_x=863)

$4 \times 4$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-02.jpg?height=203&width=217&top_left_y=1384&top_left_x=1084)

$12 \times 12$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-02.jpg?height=203&width=217&top_left_y=1644&top_left_x=1084)

$6 \times 6$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-02.jpg?height=206&width=206&top_left_y=1382&top_left_x=1317)

$14 \times 14$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-02.jpg?height=203&width=201&top_left_y=1644&top_left_x=1317)

$8 \times 8$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-02.jpg?height=203&width=214&top_left_y=1384&top_left_x=1538)

$16 \times 16$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-02.jpg?height=206&width=200&top_left_y=1645&top_left_x=1540)

Figure 4. Displaying the HOG features of a 122×122 image with different numbers of pixels per cell.

Quantitative comparison of our HOG-CNN models with different numbers of Pixels Per Cell in the HOG parameters

Cumulative density plot of CNN euclidian distance with different Pixels Per Cell (PPC) for the HOG parameters

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-02.jpg?height=586&width=879&top_left_y=2114&top_left_x=276)

Boxplot of CNN euclidian distance with different Pixels Per Cell (PPC) for the HOG parameters

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-02.jpg?height=634&width=653&top_left_y=2079&top_left_x=1191)

Figure 5. We trained 7 CNNs with training data differing in the preprocessing hyperparameter of the pixels per cell during HOG collection.

We found that HOG features are good at extracting the face from an image and ignoring the rest of the noise (Fig. 4; Singh et al. 2020; Virtanen et al. 2020). Furthermore, being invariant to light changes means changes in lighting should result in the same accuracy from our model's prediction as normal lighting.

One hyperparameter of our HOG feature collector is the number of pixels per cell (PPC). We experimented to find the optimum number of PPC that would result in the most accurate CNN and found it to be $6 \times 6$ PPC (Fig. 5).

The images are resized before getting the HOG features which slightly dilates the HOG features compared to the feature collection of the full-size images. Furthermore, the images and points are normalised to standardise our data, improving our CNN's accuracy.

D. CNN model architecture

\section{Our model's CNN's architecture}

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-03.jpg?height=447&width=1466&top_left_y=888&top_left_x=362)

Figure 6. A diagram of our CNN's layers.

We found that the deeper our $\mathrm{CNN}$ is, the more accurate it becomes. The first convolutional layer finds spatial patterns in images. The second extracts the most prominent features of convoluted images. They are then flattened into a vector to be passed into 3 dense layers that will find finer details of the image that influence the positions of the coordinates. The dropout at each layer helps prevent overfitting. The size of the final layer is 84 , corresponding to the 42 points for facial alignment.

\section{Quantitative comparison of our HOG-CNN model with a Sigmoid activation function on the output layer and ReLU activation function}

Cumulative density plot of euclidean distance on evaluation data from CNN model with Sigmoid output vs. ReLU output
![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-03.jpg?height=702&width=1572&top_left_y=1774&top_left_x=272)

Figure 7. Performance comparison of the same model with a different activation function on the output layer. Each layer that has an activation function uses ReLU as the numbers we want to predict are non-negative. Only our output layer uses the Sigmoid activation function as we found it improves our model's accuracy (Fig. 7).

E. Prediction model

HOG-CNN face alignment model prediction flowchart

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=146&width=1645&top_left_y=366&top_left_x=240)

Figure 8. Flowchart to get our model's predicted points on any image (of a face)

This prediction model follows the same steps as our training data with the added de-normalising step to visualise the predictions correctly.

F. Qualitative analysis

The 12 best predictions of our HOG-CNN model on the evaluation set and their euclidean distance with the ground truth points
![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=554&width=270&top_left_y=926&top_left_x=191)

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=280&width=265&top_left_y=928&top_left_x=491)

$3.02$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=260&width=280&top_left_y=1212&top_left_x=478)

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=271&width=266&top_left_y=932&top_left_x=775)

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=269&width=266&top_left_y=1210&top_left_x=775)

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=271&width=252&top_left_y=932&top_left_x=1077)

$3.148$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=263&width=261&top_left_y=1213&top_left_x=1062)

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=274&width=268&top_left_y=931&top_left_x=1357)

$3.155$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=255&width=252&top_left_y=1214&top_left_x=1359)

$2.969$
![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=528&width=260&top_left_y=949&top_left_x=1647)

The 12 worst predictions of our HOG-CNN model on the evaluation set and their euclidean distance with the ground truth points
![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=538&width=266&top_left_y=1586&top_left_x=173)

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=271&width=265&top_left_y=1591&top_left_x=472)

$7.041$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=255&width=252&top_left_y=1865&top_left_x=470)

$9.841$

$9.72$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=255&width=258&top_left_y=1607&top_left_x=760)

$7.026$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=255&width=265&top_left_y=1865&top_left_x=754)

$6.946$ $8.408$
![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=518&width=266&top_left_y=1608&top_left_x=1040)

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=252&width=249&top_left_y=1609&top_left_x=1339)

$6.894$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=261&width=252&top_left_y=1859&top_left_x=1338)

$7.668$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=258&width=250&top_left_y=1603&top_left_x=1623)

$6.855$

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-04.jpg?height=256&width=257&top_left_y=1867&top_left_x=1617)

Figure 9. Plotting the 12 best and worst predictions made by our HOG-CNN model on our evaluation set with their Euclidean distance.

Where our model performs well: the best predictions are overwhelmingly on white feminine faces facing straight where there is homogeneous lighting and no obstructions. Our model is good at finding the corners of the eyes and the shape of the face when it is fully in the picture.

Where our model does not perform well: the worst predictions are opposites of most of the best predictions and half of the images are generally darker. Surprisingly, only one image of the 12 worst predictions has parts of its face obstructed. Our model is bad on rotated faces, lack of contrast between face and background and hardly visible lips. Reasons and solutions: Our training dataset is unbalanced and requires more darker/dimmer images, people with different skin colours than white and better features to extract the lips. We could have made our model more robust using cross-validation. To deal with our model's flaws on faces with non-homogeneous lighting, we could have averaged out the pixel intensity on every image before collecting the HOGs. Our model's flaws in rotated faces show augmenting the data with rotated faces did not make it robust for that and reveals it may be finding general patterns in faces and applying them again.

Plotting different face alignment models predictions on the same images
![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-05.jpg?height=738&width=1530&top_left_y=546&top_left_x=274)

Figure 10. A comparison of the plottings on the same 6 images by different models we implemented.

We have implemented two other models apart from our HOG-CNN model: a simple CNN and a coarse-to-fine regression model (Appendix Fig. 1 \& Fig. 2). The coarse-to-fine model seems more robust to shape changes whereas the other two models seem to keep 'standard shapes' for each facial landmark, a characteristic of CNNs, as we can observe with the predicted mouth alignments.

\section{G. Quantitative analysis}

\section{Quantitative comparison of the 3 different models we have implemented}

Cumulative density plots of euclidean distance of HOG-CNN model, coarse-to-fine CNN model and simple CNN on evaluation data
![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-05.jpg?height=648&width=1506&top_left_y=1771&top_left_x=301)

Figure 11. Comparing the performances of 3 different models we have implemented, including the HOG-CNN model we are detailing in this report. We can see that our HOG-CNN model performs much better than the two other models, getting better euclidean distances overall and a much lower euclidean distance average. Our coarse-to-fine model (Val et al. 2018) seems bad at generalising, as we could observe qualitatively since it deals with more detailed features. Our HOG-CNN model could be improved with an architecture that, similarly to our coarse-to-fine model, takes a closer look at the details of the face that could reveal its alignment, which would improve its robustness to outlier shapes of mouths or face positions.

We can partially attribute the average error of our model to the hand-labelling of the ground-truth. An average difference of 5 pixels between two hand-labellings of the same $244 \times 244$ pixels pictures is very reasonable and expected. Therefore, our HOG-CNN model performs, on average, as well as a human.

\section{Face landmark colouring - lips}

Method for colouring the lips the predicted landmarks from our HOG-CNN model

1) Create the masks

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-06.jpg?height=840&width=1480&top_left_y=759&top_left_x=279)

Figure 12. A flowchart of the method we used to colour a face's lips using the predicted points from our HOG-CNN model and colour-mapping.

The 5 most realistic results from our lip colouring using the predicted points of our HOG-CNN model
![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-06.jpg?height=492&width=1354&top_left_y=1790&top_left_x=378)

Figure 13. The best qualitative results from our lip colouring method using the HOG-CNN model's predictions to find the lips. Colouring the lips of our example images using the HOG-CNN predicted points
![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-07.jpg?height=758&width=1184&top_left_y=83&top_left_x=468)

Figure 14. Colouring in the lips of our example images.

We used colour-mapping (Hunter 2007) as it can add a colour filter and not change the texture of the image.

Our results seem to work well on well lit, fully visible lips with an open or closed mouth. Naturally, this is subject to the correct alignment of the lips. Furthermore, less vibrant colours (e.g purple) blend in better with the texture of the lips and look more realistic in darker images.

The 6 least realistic results from our lip colouring using the predicted points of our HOG-CNN model
![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-07.jpg?height=446&width=1446&top_left_y=1216&top_left_x=337)

Figure 15. The worst qualitative results from our lip colouring method using the HOG-CNN's predictions.

Our results do not work well on images where the lips are hidden or very thin, but this is also subjective to how well our model has found the lips. Our model also doesn't work on images with a lack of colour (e.g. black and white) or obstructed mouths as the lipstick stands out.

The 7 least realistic results from our lip colouring using the ground truth face alignment points
![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-07.jpg?height=432&width=1612&top_left_y=1944&top_left_x=294)

Figure 16. The worst qualitative results from our lip colouring method using the ground truth points.

We can add on the previously listed flaws that our model struggles on oddly shaped lips such as in the 2 nd and 3 rd images. 

\section{Conclusion}

In conclusion, we have implemented a face alignment prediction model that uses HOGs of images to train a CNN that performs very well on well-lit images and moderately on darker and obstructed ones. We affirm that with the expected divergence in labelling, our model performs, on average, as well as a human labeller. Furthermore, we have implemented a realistic virtual lipstick applier using the predicted landmarks from our HOG-CNN model and colour-maps. Our models could still be improved through data balancing, cross-validation, and histogram averaging to get better performances from our CNN and our lipstick applier. 

\section{References}

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science \& Engineering, 9(3), 90-95.

Singh, Swarnima \& Singh, Durgesh \& Yadav, Vikash. (2020). Face Recognition Using HOG Feature Extraction and SVM Classifier. 8. 6437-6440. 10.30534/ijeter/2020/244892020.

Valle, R., Buenaposada, J. M., Valdés, A., \& Baumela, L. (2018). A Deeply-Initialized Coarse-to-fine Ensemble of Regression Trees for Face Alignment. ECCV (14), 609-624.

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., ... SciPy $1.0$ Contributors. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17, 261-272.

\section{Appendix}

\section{Simple CNN model method flowchart}

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-09.jpg?height=187&width=1133&top_left_y=904&top_left_x=255)

Appendix Figure 1. Flowchart of our simple CNN model, one of the models we implemented and used to compare the performance with our HOG-CNN model. The images go through very basic preprocessing and do not collect HOG features.

\section{Coarse-to-fine regression CNNs model method flowchart}

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-09.jpg?height=203&width=1396&top_left_y=1319&top_left_x=256)

Appendix Figure 2. Flowchart of our coarse-to-fine model we implemented and used to compare the performance with our HOG-CNN model. The images go through very basic preprocessing and do not collect HOG features. This model trains 43 models, the first takes as input the $122 \times 122$ image and predicts all of the points and we then use these predictions and make a crop around each to train 42 models that are each specialised in training one specific point on the image (e.g. the tip of the nose). We then concatenate these predictions to return them as the face alignment predictions for the image.

Method for colouring the irises with the predicted facial landmark from the HOG-CNN model
![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-09.jpg?height=292&width=1394&top_left_y=1964&top_left_x=293)

Use the masks to extract the face and the eye and apply a colourmap to the eye

![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-09.jpg?height=287&width=1030&top_left_y=2372&top_left_x=295)

Appendix Figure 3. A flowchart of the method we used to colour in the iris of a face using the predicted points from our HOG-CNN model and colour-mapping.

Changing colour of irises on images using the HOG-CNN model to predict the landmarks
![](https://cdn.mathpix.com/cropped/2022_11_18_46c27f539d6c464af7efg-10.jpg?height=524&width=1446&top_left_y=318&top_left_x=336)

Appendix Figure 4. Here are images with the attempts to change the colour of the irises. Our method fails for two reasons: the first is that it does not always find the contour of the iris, making the next steps in the method impossible, and if it does return a contour it's not always exactly the iris. The iris needs to be fully visible (e.g. right eye of example image 3 in the figure) for our method to find it. We can improve our method by finding a more robust function to find the circle of the irises or process the image differently at a higher resolution. The results of this method are unrealistic.

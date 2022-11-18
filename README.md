# Face alignment with a CNN extracting HOG features and realistically coloring the lips

## About
This project defines, justifies and shows the results of a Convolutional Neural Network for face alignment, the task of outlining facial landmarks from the image of a face.

There is a PDF report [here](https://github.com/rzuberi/Face_Alignment_HOG_CNN/blob/main/Face%20Alignment%20and%20Colouring%20report.pdf) that discusses the solution defined.

There is a Python Notebook [here](https://github.com/rzuberi/Face_Alignment_HOG_CNN/blob/main/HOG_CNN_implementation_CV_Assignment_236636.ipynb) that displays the data manipulation performed and the training of the CNN.

## Face alignment task
### 1. Face alignment training and prediction methods
<img width="1536" alt="Screenshot 2022-11-18 at 15 08 03" src="https://user-images.githubusercontent.com/56508673/202736389-fbc33c5c-1011-4782-b66d-312896561027.png">
The flowchart of going from our training data to evaluating our face alignment predictions model.

<img width="1377" alt="Screenshot 2022-11-18 at 14 51 44" src="https://user-images.githubusercontent.com/56508673/202736480-2d2211df-0a16-4db0-bf1b-99eeaaf18821.png">
Flowchart to get our model’s predicted points on any image (of a face).

### 2. Our CNN's architecture
<img width="1542" alt="Screenshot 2022-11-18 at 15 07 29" src="https://user-images.githubusercontent.com/56508673/202736242-ef07803a-5830-48d6-8499-1d2815bbebf4.png">
A diagram of our CNN’s layers.

### 3. Results of face alignment
<img width="1185" alt="Screenshot 2022-11-18 at 14 49 40" src="https://user-images.githubusercontent.com/56508673/202732253-149130be-04c0-480c-abd4-e93d6433ebea.png">
Plotting the 12 best and worst predictions made by our HOG-CNN model on our evaluation set with their Euclidean distance.

## Lip coloring task
### 1. Lip coloring method
<img width="1229" alt="Screenshot 2022-11-18 at 15 09 35" src="https://user-images.githubusercontent.com/56508673/202736671-bb5ef600-5681-49a9-a861-abc0707c7f56.png">
A flowchart of the method we used to colour a face's lips using the predicted points from our HOG-CNN model and colour-mapping.

### 2. Results of lip coloring
<img width="912" alt="Screenshot 2022-11-18 at 15 10 20" src="https://user-images.githubusercontent.com/56508673/202736844-045a5dcb-8819-427c-a538-725d3e313440.png">
The best qualitative results from our lip colouring method using the HOG-CNN model’s predictions to find the lips.

<img width="1396" alt="Screenshot 2022-11-18 at 15 11 09" src="https://user-images.githubusercontent.com/56508673/202736991-6e2df6df-2a52-4390-a639-476ae11f91d6.png">
The worst qualitative results from our lip colouring method using the HOG-CNN’s predictions.

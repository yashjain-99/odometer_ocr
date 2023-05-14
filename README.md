# Odometer OCR
## _Using YOLOv8 and CRNN_

This is a project that uses YOLOv8 for odometer detection and a CRNN model for OCR to read the kilometers driven from the odometer. The project is designed to automate the process of recording mileage readings for vehicles, making it easier and more accurate.

## ✨Features✨

- Easy to use command line interface to make predictions.
- Fast and accurate.
- Can be easily intergrated into any existing project.
- Real-time detection.
- Allow the user to process multiple images or videos at once, saving the output in a single CSV file.

## Tech

Odometer OCR uses a number of open source projects to work properly:

- [Pytorch] - Machine learning framework.
- [YOLOv8] - Awesome computer vision model. Used here for odometer detection.
- [CRNN] - A deep learning model that combines convolutional neural networks and recurrent neural networks to perform OCR on sequential data.
- [OpenCV] - Open source computer vision and machine learning software library.

## Usage
Odometer OCR requires [Python](https://nodejs.org/) 3.6+ to run.
- Clone repository
    ```sh
    git clone https://github.com/yashjain-99/Projects.git 
    ```
- Install the dependencies:
    ```py
    pip install -r requirements.txt
    ```
- Download trained model from [here] and store it inside models folder.
- To use the command line interface for making predictions:
    - If the source is a directory containing multiple images:
        ```sh
        python odometer_ocr.py --source path/to/directory
        ```
    - If the source is a single image:
        ```sh
        python odometer_ocr.py --source path/to/image.jpg
        ```
    This will save the predictions in an output.csv file in the same directory as the source.
- To get a pandas DataFrame as output:
    ```py
    import odometer_ocr
    import pandas as pd
    
    # for a directory containing multiple images
    df = odometer_ocr.predict('path/to/directory')
    print(df)
    
    # for a single image
    df = odometer_ocr.predict('path/to/image.jpg')
    print(df)
    ```
> To learn more about the preprocessing and training involved in this project, refer to the OdometerOCR.ipynb Jupyter notebook in the project repository.

**Please feel free to reach out to me at yashj133.yj@gmail.com in case of any queries or suggestions!**

   [YOLOv8]: <https://github.com/ultralytics/ultralytics>
   [Pytorch]: <https://pytorch.org/>
   [CRNN]: <https://github.com/clovaai/deep-text-recognition-benchmark>
   [OpenCV]: <https://opencv.org/>
   [mail]: <mailto:yashj133.yj@gmail.com>
   [here]: <https://drive.google.com/file/d/1F-VHaeaVff3-UYzdVZ8OSArhmWtwNC0_/view?usp=sharing>
import odometer_ocr
import pandas as pd

#incase of directory of images
df=odometer_ocr.predict('data/')
print(df)

#incase of single image
df=odometer_ocr.predict('data/scraped_0OpSll_1654866831036.jpg')
print(df)
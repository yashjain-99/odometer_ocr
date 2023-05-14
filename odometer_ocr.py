import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from utils import CTCLabelConverter
from dataset import AlignCollate
from ultralytics import YOLO
from custom import Model, RawDataset
import numpy as np
import typer
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
chars = '0123456789abcdefghijklmnopqrstuvwxyz'
converter = CTCLabelConverter(chars)
num_class = len(chars)
AlignCollate_demo = AlignCollate(imgH=32, imgW=100, keep_ratio_with_pad=False)


def ocr_predict(model, image):
    demo_data = RawDataset(root=[image])
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=AlignCollate_demo, pin_memory=True)
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([25] * batch_size).to(device)
            text_for_pred = torch.LongTensor(
                batch_size, 25 + 1).fill_(0).to(device)
            preds = model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index, preds_size)
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            confidence_score = preds_max_prob[0].cumprod(dim=0)[-1]
            return [preds_str[0], confidence_score]


def predict(source):
    object_detection_model = YOLO('models/best.pt')
    ocr_model = Model()
    ocr_model = torch.nn.DataParallel(ocr_model).to(device)
    ocr_model.load_state_dict(torch.load(
        'models/best_accuracy.pth', map_location=device))
    results = object_detection_model(source)
    output = {'filename': [], 'prediction': [], 'confScore': []}
    for result in results:
        # find which region has highest conf score
        if len(result.boxes.conf) == 0:
            output['filename'].append(result.path)
            output['prediction'].append('No odometer found')
            output['confScore'].append('No odometer found')
            continue
        idx = list(result.boxes.conf).index(max(result.boxes.conf))
        # crop roi
        x, y, X, Y = np.array(result.boxes.xyxy[idx].cpu())
        roi = result.orig_img[int(y):int(Y), int(x):int(X)]
        # do ocr
        pred, conf = ocr_predict(ocr_model, roi)
        output['filename'].append(result.path)
        output['prediction'].append(pred)
        output['confScore'].append(float(conf.cpu()))
    df=pd.DataFrame(output)
    return df


app = typer.Typer(help="CLI for ODOMETER OCR")


@app.command()
def main(
    source: str = typer.Option(
        "data/", "--source", "-s", help="Image/Directory of images to predict"),
):
    df = predict(source)
    df.to_csv('output.csv', index=False)
    print("output.csv file saved")
    return df

if __name__ == "__main__":
    app()


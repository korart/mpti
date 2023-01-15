import os
import streamlit as st
from pdf2image import convert_from_path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

import pytesseract


st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Table PDF to Excel Converter")

def PIL_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) 

def cv_to_PIL(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def sharpen_image(pil_img):

    img = PIL_to_cv(pil_img)
    sharpen_kernel = np.array([[-1, -1, -1], 
                               [-1,  9, -1], 
                               [-1, -1, -1]])

    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    pil_img = cv_to_PIL(sharpen)
    return pil_img

def binarizeBlur_image(pil_img):
    image = PIL_to_cv(pil_img)
    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]

    result = cv2.GaussianBlur(thresh, (5,5), 0)
    result = 255 - result
    return cv_to_PIL(result)

# Save pdf file as tempfile
def save_pdf(uploadedfile, tempdir):
    pdf_file_path = os.path.join(tempdir, uploadedfile.name)
    with open(pdf_file_path,"wb") as f:
        f.write(uploadedfile.getbuffer())
    return pdf_file_path

# Convert pdf to png and save as tempfiles
def pdf2png(pdf_file_path, tempdir):
    image_file_list = []
    pdf_pages = convert_from_path(pdf_file_path, 600)

    for page_enumeration, page in enumerate(pdf_pages, start=1):
        filename = f"{tempdir}/page_{page_enumeration:03}.png"
        page.save(filename, "PNG")
        image_file_list.append(filename)
    return image_file_list

# Box resize
def scale_box(box, k=0.05):
    xmin, ymin, xmax, ymax = box
    width = xmax-xmin
    height = ymax-ymin
    x_padding = width * k / 2
    y_padding = height * k / 2
    xmin, ymin, xmax, ymax = xmin - x_padding, ymin - y_padding, xmax + x_padding, ymax + y_padding
    return [xmin, ymin, xmax, ymax]

def get_main_table(image):
    image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]

    tables = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        cropped_img = image.crop(scale_box(box))
        tables.append(cropped_img)
    return tables[0]

def get_table_structure(table):
    image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

    image = table
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    rows = []
    columns = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        if model.config.id2label[label.item()] == 'table row':
            rows.append(scale_box(box))
        if model.config.id2label[label.item()] == 'table column':
            columns.append(scale_box(box))
    return rows, columns

def preprocess_rows(rows, min_x, max_x):
    rows = np.array(rows)
    rows[:, 0] = min_x
    rows[:, 2] = max_x
    rows = rows[rows[:, 1].argsort()]
    for i in range(len(rows)):
        if i < len(rows) - 1:
            value = (rows[i, 3] + rows[i + 1, 1]) / 2
            rows[i, 3], rows[i + 1, 1] = value, value
    return rows

def preprocess_columns(columns, min_y, max_y):
    columns = np.array(columns)
    columns[:, 1] = min_y
    columns[:, 3] = max_y
    columns = columns[columns[:, 0].argsort()]
    for i in range(len(columns)):
        if i < len(columns) - 1:
            value = (columns[i, 2] + columns[i + 1, 0]) / 2
            columns[i, 2], columns[i + 1, 0] = value, value
    return columns

def get_cells(rows, columns):
    cells = []
    for row in rows:
        cells_row = []
        for column in columns:
            cells_row.append([column[0], row[1], column[2], row[3]])
        cells.append(cells_row)
    return np.array(cells)

def clean_dataframe(df):
        for col in df.columns:

            df[col]=df[col].str.replace("'", '', regex=True)
            df[col]=df[col].str.replace("|", '', regex=True)
            df[col]=df[col].str.replace('"', '', regex=True)
            df[col]=df[col].str.replace(']', '', regex=True)
            df[col]=df[col].str.replace('[', '', regex=True)
            df[col]=df[col].str.replace('{', '', regex=True)
            df[col]=df[col].str.replace('}', '', regex=True)
        return df

if __name__ == "__main__":

    input_placeholder = st.empty()
    output_placeholder = st.empty()

    with input_placeholder.container():
        with st.expander("Advanced options:"):
            st1, st2 = st.columns((1, 1))
            TD_th = st1.slider('Table detection threshold', 0.0, 1.0, 0.6)
            TSR_th = st2.slider('Table structure recognition threshold', 0.0, 1.0, 0.8)
        uploadedfile = st.file_uploader("Upload PDF with table(s)", type=['pdf'])

    
    if uploadedfile is not None:

        with TemporaryDirectory() as tempdir:
            with st.spinner('PDF to PNG converting...'):
                pdf_file_path = save_pdf(uploadedfile, tempdir)
                uploadedfile = None
                images = pdf2png(pdf_file_path, tempdir)
            
            with output_placeholder.container():
                label = st.empty()
                progressbar = st.progress(0)
            progress_step = 1/len(images)

            data = []
            for image_idx, image_path in enumerate(images):
                label.text(f"Page {image_idx + 1} of {len(images)} processing...")
                table = get_main_table(Image.open(image_path))
                rows, columns = get_table_structure(table)
                min_x, min_y = 0, 0
                max_x, max_y = table.size
                rows = preprocess_rows(rows, min_x, max_x)
                columns = preprocess_columns(columns, min_y, max_y)
                cells = get_cells(rows, columns)
                for row in cells:
                    data_row = [f'Page{image_idx}']
                    for cell in row:
                        # custom_config = r'--oem 3 --psm 6'
                        # data_row.append(' '.join(pytesseract.image_to_string(table.crop(cell), lang='rus+eng', config=custom_config).split()))
                        data_row.append(' '.join(pytesseract.image_to_string(table.crop(cell), lang='rus+eng').split()))
                    data.append(data_row)
                progressbar.progress(progress_step * (image_idx + 1))                         
            df = pd.DataFrame(data)
            df = clean_dataframe(df)
            st.dataframe(df)
        
        # Export to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output) as writer:
            df.to_excel(writer, index=False, header=False)
        st.download_button("Download XLSX", data=output.getvalue(), file_name="workbook.xlsx", mime="application/vnd.ms-excel")
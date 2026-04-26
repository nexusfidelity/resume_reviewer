from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import numpy as np

pdf_path = "sample_resume1.pdf"
pages = convert_from_path(pdf_path, dpi=150)  # Higher DPI = better accuracy

resume_text=[]


for i, page in enumerate(pages, start=1):

    img = np.array(page)
    output = pytesseract.image_to_string(img)
    text = output[0]['rec_texts']
    text = " ".join(text)

    # print(text)
    resume_text.append(text)
    # for res in output:
    #     res.print()

print(resume_text)
import easyocr
from pdf2image import convert_from_path
import numpy as np

pdf_path = "sample_resume1.pdf"
pages = convert_from_path(pdf_path, dpi=150)  # Higher DPI = better accuracy
reader = easyocr.Reader(['en'])

for i, page in enumerate(pages, start=1):

    img = np.array(page)
    result = reader.readtext(img, detail=0)
    result = " ".join(result)
    print(result)


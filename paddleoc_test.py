from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import numpy as np

import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

pdf_path = "sample_resume1.pdf"
pages = convert_from_path(pdf_path, dpi=150)  # Higher DPI = better accuracy

resume_text=[]

ocr = PaddleOCR(
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_textline_orientation=False,
    use_doc_unwarping=False)

for i, page in enumerate(pages, start=1):

    img = np.array(page)
    output = ocr.predict(img)
    text = output[0]['rec_texts']
    text = " ".join(text)

    # print(text)
    resume_text.append(text)
    # for res in output:
    #     res.print()

print(resume_text)
from pdf2image import convert_from_path

pdf_path = "sample_resume1.pdf"
pages = convert_from_path(pdf_path, dpi=300)  # Higher DPI = better accuracy

for i, page in enumerate(pages, start=1):
    image_path = f"page_{i}.png"
    # page.save(image_path, "PNG")
    # print(f"Saved {image_path}")
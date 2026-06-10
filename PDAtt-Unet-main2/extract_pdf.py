import sys
try:
    import pypdf
    import re
    pdf_path = r"C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject\PBL4\PDAtt-Unet-main2\PDAtt.pdf"
    reader = pypdf.PdfReader(pdf_path)
    text = '\n'.join([p.extract_text() for p in reader.pages if p.extract_text()])
    
    # Extract sections related to loss
    print("--- HYBRID LOSS ---")
    matches = re.findall(r'.{0,150}hybrid loss.{0,150}', text, re.IGNORECASE | re.DOTALL)
    for m in set(matches):
        print(m.replace('\n', ' ').strip())
        
    print("\n--- LOSS FORMULA ---")
    matches = re.findall(r'.{0,100}\bL_{0,1}total\b.{0,100}', text, re.IGNORECASE | re.DOTALL)
    for m in set(matches):
        print(m.replace('\n', ' ').strip())

    print("\n--- GAMMA ---")
    matches = re.findall(r'.{0,150}gamma.{0,150}', text, re.IGNORECASE | re.DOTALL)
    for m in set(matches):
        print(m.replace('\n', ' ').strip())

except Exception as e:
    print(e)

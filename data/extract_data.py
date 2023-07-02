from PyPDF2 import PdfReader
  
# creating a pdf reader object
reader = PdfReader('A1860-45 (1).pdf')
  
# printing number of pages in pdf file
print(len(reader.pages))
  
# getting a specific page from the pdf file
page = reader.pages[0]
print(page)
# extracting text from page
text = page.extract_text()
print(text)

text=""
for i in range(len(reader.pages)):
    page = reader.pages[i]

    text += page.extract_text()
print(text)

with open('data.txt', 'w' , encoding='windows-1252') as f:
    f.write(text)


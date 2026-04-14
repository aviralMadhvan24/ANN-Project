import pypdf
import sys

def main():
    try:
        reader = pypdf.PdfReader('tutorial.pdf')
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        
        with open('pdf_content.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Successfully extracted PDF content to pdf_content.txt")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

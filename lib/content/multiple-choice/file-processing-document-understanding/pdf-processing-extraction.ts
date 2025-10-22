/**
 * Multiple choice questions for PDF Processing & Extraction section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const pdfprocessingextractionMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'fpdu-pdf-proc-mc-1',
        question:
            'What is the main difference between searchable and scanned PDFs?',
        options: [
            'Searchable PDFs are smaller in file size',
            'Searchable PDFs contain extractable text, scanned PDFs are images',
            'Scanned PDFs are more secure',
            'Searchable PDFs always have better quality',
        ],
        correctAnswer: 1,
        explanation:
            'Searchable PDFs contain actual text data that can be extracted directly. Scanned PDFs are images of documents that require OCR (Optical Character Recognition) to extract text.',
    },
    {
        id: 'fpdu-pdf-proc-mc-2',
        question:
            'Which library is BEST for extracting tables from PDFs?',
        options: [
            'PyPDF2',
            'pdfplumber',
            'Pillow',
            'BeautifulSoup',
        ],
        correctAnswer: 1,
        explanation:
            'pdfplumber is specifically designed for table extraction and provides excellent table detection with customizable settings. PyPDF2 has poor table handling, and the other options aren\'t PDF libraries.',
    },
    {
        id: 'fpdu-pdf-proc-mc-3',
        question:
            'How do you extract text from a scanned PDF?',
        options: [
            'Use PyPDF2.extract_text()',
            'Convert to images and use OCR (Tesseract)',
            'Use pandas.read_pdf()',
            'Scanned PDFs cannot have text extracted',
        ],
        correctAnswer: 1,
        explanation:
            'Scanned PDFs are images, so you need to convert pages to images (pdf2image) and then run OCR (pytesseract) to extract text. Direct text extraction won\'t work on image-based PDFs.',
    },
    {
        id: 'fpdu-pdf-proc-mc-4',
        question:
            'What does pdfplumber\'s table_settings parameter control?',
        options: [
            'The color scheme of extracted tables',
            'How table boundaries and cells are detected',
            'The format of output (CSV vs JSON)',
            'The maximum number of tables to extract',
        ],
        correctAnswer: 1,
        explanation:
            'table_settings controls how pdfplumber detects table boundaries and cells, including strategies like "lines" vs "text", snap tolerance, and edge detection parameters.',
    },
    {
        id: 'fpdu-pdf-proc-mc-5',
        question:
            'Why is PyMuPDF (fitz) often chosen for image extraction from PDFs?',
        options: [
            'It is the only library that can extract images',
            'It is the fastest PDF library with excellent image handling',
            'It automatically enhances image quality',
            'It is free while other libraries cost money',
        ],
        correctAnswer: 1,
        explanation:
            'PyMuPDF (fitz) is the fastest PDF library and has excellent image extraction capabilities with doc.extract_image(). While other libraries can extract images, PyMuPDF does it most efficiently.',
    },
];


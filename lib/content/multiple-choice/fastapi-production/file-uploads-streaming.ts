import { MultipleChoiceQuestion } from '@/lib/types';

export const fileUploadsStreamingMultipleChoice = [
  {
    id: 1,
    question:
      'Why should you validate file types using "magic numbers" instead of file extensions?',
    options: [
      'File extensions can be easily changed by users, while magic numbers identify the actual file format by examining the file header bytes',
      'Magic numbers provide faster validation than checking extensions',
      "File extensions don't work on all operating systems",
      'Magic numbers automatically compress files during upload',
    ],
    correctAnswer: 0,
    explanation:
      'File extensions are metadata that users can trivially change: rename malware.exe to image.jpg and upload. Security vulnerability! Magic numbers are the first bytes of a file that identify its true format: JPEG files start with FF D8 FF, PNG with 89 50 4E 47, PDF with 25 50 44 46. Validation: read first 2048 bytes, use python-magic library to detect actual MIME type, reject if not in allowed list. Example: user uploads virus.exe renamed to resume.pdf. Extension says "PDF", magic number says "Windows executable" → blocked. This prevents: uploading malicious files disguised as images/documents, storing dangerous content that could be served to users. Implementation: mime_type = magic.from_buffer(file.read(2048), mime=True). Magic numbers (option 1) provide security, not performance (option 2), work on all systems like extensions (option 3), and don\'t compress files (option 4). This is critical for production security.',
  },
  {
    id: 2,
    question:
      'What is the main advantage of using presigned URLs for S3 uploads instead of routing files through your API server?',
    options: [
      'Client uploads directly to S3, bypassing API server bandwidth and CPU, reducing costs and enabling horizontal scaling',
      'Presigned URLs automatically encrypt files',
      'Presigned URLs eliminate the need for authentication',
      'Presigned URLs are faster because they use HTTP/2',
    ],
    correctAnswer: 0,
    explanation:
      "Presigned URLs enable direct client-to-S3 upload, removing the API server from the data path. Scenario: User uploads 1GB video. With routing through API: client → (1GB over internet) → API server → (1GB within AWS) → S3. API server must receive, then forward—using CPU, RAM, and bandwidth. Cost: EC2 bandwidth charges, need larger instances. With presigned URL: client → (1GB directly) → S3. API server only generates URL (cheap), S3 handles upload (scales infinitely). Benefits: 1) No API bandwidth/CPU use, 2) S3's massive throughput, 3) Lower costs (no EC2 bandwidth), 4) API servers can be tiny (just generate URLs). Implementation: s3.generate_presigned_post(...). Presigned URLs don't auto-encrypt beyond standard HTTPS (option 2), still require authentication to generate URL (option 3), and HTTP/2 works for both approaches (option 4). This is the production standard for handling large file uploads at scale.",
  },
  {
    id: 3,
    question:
      'When streaming a large file download in FastAPI, why use a generator function instead of reading the entire file into memory?',
    options: [
      'A generator yields chunks incrementally, keeping memory usage constant regardless of file size, preventing OOM crashes',
      'Generators automatically compress files for faster downloads',
      'Generators provide better security by encrypting each chunk',
      'Generators are required for async file operations',
    ],
    correctAnswer: 0,
    explanation:
      "Generators enable memory-efficient streaming by yielding one chunk at a time. Problem: 100 concurrent users download 1GB files. Naive approach: data = file.read() loads 100GB into RAM → server crashes (OOM). Generator approach: async def stream(): while chunk := await file.read(1MB): yield chunk. Memory use: 100 * 1MB = 100MB (constant), works with any file size. How it works: FastAPI reads chunk, sends to client, reads next chunk only after previous sent. The generator pattern processes data in fixed-size chunks, making memory usage O(1) instead of O(file_size). This enables serving large files (videos, backups) without requiring huge servers. Generators don't compress (option 2—use gzip.GzipFile), don't encrypt (option 3—use TLS), and aren't required for async (option 4—you can use async without generators). The key is: generators = streaming = constant memory = scalability. Production pattern: return StreamingResponse(file_stream(), media_type=...). Critical for serving large files without crashing.",
  },
  {
    id: 4,
    question:
      'What is the purpose of chunked uploads with resume capability for large files?',
    options: [
      'If connection drops mid-upload, client can resume from the last successful chunk instead of restarting the entire upload',
      'Chunked uploads provide automatic virus scanning between chunks',
      'Chunked uploads compress files more efficiently than single uploads',
      'Chunked uploads are required for files larger than 100MB',
    ],
    correctAnswer: 0,
    explanation:
      "Chunked uploads with resume prevent restarting failed uploads. Scenario: User uploads 5GB video on slow connection. At 80% (4GB uploaded), connection drops. Without resume: must restart from 0%, upload 5GB again (frustrating!). With resume: 1) Client tracks which chunks uploaded (0-79 complete, 80-99 missing), 2) Client queries server: \"which chunks you have?\", 3) Server returns: [0-79], 4) Client resumes from chunk 80. Implementation: Divide file into chunks (5MB each), upload each chunk individually, server stores completion state, client can query status and re-upload missing chunks only. Each chunk upload is idempotent: uploading chunk 50 twice is safe. Benefits: Network resilience (mobile uploads), user experience (no frustrating restarts), bandwidth efficiency (don't re-upload completed data). Chunked uploads don't provide virus scanning (option 2—that's separate), don't compress better (option 3—same compression), and aren't technically required but highly recommended for large files (option 4). Production pattern: initiate upload (get upload_id), upload chunks with chunk_number, query status for resume, complete when all chunks uploaded.",
  },
  {
    id: 5,
    question:
      'Why is it important to set a Content-Disposition header when streaming file downloads?',
    options: [
      'Content-Disposition: attachment tells the browser to download the file with a specific filename instead of displaying it inline',
      'Content-Disposition encrypts the file during download',
      'Content-Disposition compresses the file automatically',
      'Content-Disposition is required for streaming responses to work',
    ],
    correctAnswer: 0,
    explanation:
      'Content-Disposition controls how browsers handle the response. Without it: Browser might display PDF inline, try to render CSV as text, play video instead of downloading. With Content-Disposition: attachment: Browser downloads file. With filename: Browser uses your specified filename (not ugly UUID). Example: headers={"Content-Disposition": "attachment; filename=report_2024.csv"}. Result: Browser downloads as "report_2024.csv". Without filename: Browser uses URL path or generates random name. Security consideration: Sanitize filename to prevent path traversal attacks (../../etc/passwd). Options: 1) attachment → download, 2) inline → display in browser (for images, PDFs). Content-Disposition doesn\'t encrypt (option 2—use TLS), doesn\'t compress (option 3—use Content-Encoding: gzip), and isn\'t required for streaming (option 4—streaming works without it, but UX suffers). Production pattern: StreamingResponse(..., headers={"Content-Disposition": f"attachment; filename={safe_filename}"}). This ensures users get properly named downloads instead of "download.bin" or inline display.',
  },
].map(({ id, ...q }, idx) => ({ id: `fastapi-mc-${idx + 1}`, ...q }));

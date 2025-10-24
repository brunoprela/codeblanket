/**
 * Quiz questions for Working with JSON and CSV section
 */

export const jsoncsvQuiz = [
  {
    id: 'pi-jsoncsv-q-1',
    question:
      'Explain the difference between json.dump()/json.load() and json.dumps()/json.loads(). When would you use each pair?',
    hint: 'Think about what the "s" suffix means - string vs file operations.',
    sampleAnswer:
      'json.dump() and json.load() work with file objects - dump() writes JSON directly to a file, load() reads from a file. json.dumps() and json.loads() work with strings - dumps() converts Python to JSON string, loads() parses JSON string to Python. Use dump/load when working with files (most common), use dumps/loads when: 1) sending JSON over network/API, 2) storing JSON in databases, 3) debugging (printing JSON), 4) when you need the JSON as a string for manipulation.',
    keyPoints: [
      'dump/load: file operations',
      'dumps/loads: string operations',
      'dumps = dump + string, loads = load + string',
      'Use dump/load for files, dumps/loads for APIs/strings',
    ],
  },
  {
    id: 'pi-jsoncsv-q-2',
    question:
      'When should you choose CSV over JSON, and vice versa? What are the trade-offs?',
    hint: 'Consider data structure, file size, human readability, and Excel compatibility.',
    sampleAnswer:
      "Use CSV for: 1) Tabular/flat data with rows and columns, 2) Excel/spreadsheet compatibility, 3) Smaller file sizes for simple data, 4) When everyone has same columns. Use JSON for: 1) Nested/hierarchical data, 2) APIs and web services, 3) Mixed data types (preserves numbers, booleans, nulls), 4) Flexible schemas where objects can have different fields. CSV is simpler but can't handle nesting; JSON is more expressive but larger and requires parsing.",
    keyPoints: [
      'CSV: flat tables, Excel, compact',
      'JSON: nested data, APIs, type preservation',
      'CSV simpler for spreadsheets',
      'JSON better for complex, hierarchical data',
    ],
  },
  {
    id: 'pi-jsoncsv-q-3',
    question:
      'Why is DictReader/DictWriter often preferred over basic reader/writer in the CSV module?',
    hint: 'Think about code readability, maintainability, and column ordering.',
    sampleAnswer:
      'DictReader/DictWriter use column names (headers) instead of positional indices, making code more readable and maintainable. Benefits: 1) row["name",] is clearer than row[0], 2) Code doesn\'t break if column order changes, 3) Automatic header handling, 4) Easier to work with when columns are added/removed. Trade-offs: slightly more memory/processing, but worth it for code clarity. Only use basic reader/writer for headerless CSV or when performance is absolutely critical.',
    keyPoints: [
      'Uses column names instead of indices',
      'More readable: row["name",] vs row[0]',
      'Resilient to column reordering',
      'Slightly more overhead, but worth it',
    ],
  },
];

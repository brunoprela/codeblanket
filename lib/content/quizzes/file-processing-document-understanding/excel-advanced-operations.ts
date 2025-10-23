/**
 * Quiz questions for Excel Advanced Operations section
 */

export const exceladvancedoperationsQuiz = [
  {
    id: 'fpdu-excel-adv-q-1',
    question:
      'Design a system for dynamically generating Excel formulas from natural language commands like "calculate profit margin for each row" or "find the total of column B." How would you parse intents, validate references, and generate correct Excel formulas?',
    hint: 'Consider formula types, cell reference resolution, error handling, and formula validation.',
    sampleAnswer:
      'Dynamic formula generation system: (1) Parse intent using LLM - extract operation type (sum, average, calculate), target columns, and mathematical operations. (2) Map to formula templates: maintain library of formula patterns (SUM, AVERAGE, IF, VLOOKUP, custom calculations). (3) Resolve cell references: validate column names exist, convert to Excel notation (A, B, C), determine row ranges. (4) Generate formula string: use FormulaGenerator class with methods for each formula type, handle cell ranges dynamically (e.g., "B2:B100"), support both absolute ($A$1) and relative (A1) references. (5) Validate formula: check syntax, verify all references exist, handle circular references. (6) Test with sample data: apply formula to first few rows, verify results make sense. (7) Apply to range with user confirmation. For "profit margin": parse as (Revenue - Cost) / Revenue, identify column names, generate =((B2-C2)/B2)*100, apply to all rows. Handle edge cases: divide by zero, missing columns, data type mismatches.',
    keyPoints: [
      'Use LLM to parse natural language into structured formula intent',
      'Maintain template library for common formula patterns',
      'Validate column references exist before generating formulas',
      'Support both absolute and relative cell references',
      'Test generated formulas on sample data before applying',
      'Handle edge cases: divide by zero, nulls, circular references',
      'Provide preview and confirmation before applying to full range',
    ],
  },
  {
    id: 'fpdu-excel-adv-q-2',
    question:
      'Explain the trade-offs between using openpyxl formulas (strings) versus pandas calculations for adding computed columns to Excel files. When would you choose each approach?',
    hint: 'Consider performance, formula preservation, Excel compatibility, and recalculation behavior.',
    sampleAnswer:
      'openpyxl formulas vs pandas calculations: (1) openpyxl formulas (e.g., "=B2*C2"): Pros - formulas update when source data changes in Excel, preserves Excel functionality, users can see/edit formulas, formulas work with Excel features (pivot tables, etc.). Cons - slower to generate many formulas, formulas don\'t calculate until Excel opens file, can\'t easily preview results in Python. (2) pandas calculations (e.g., df["Total"] = df["Price"] * df["Qty"]): Pros - faster for bulk operations, immediate calculated results in Python, easier to validate/test, better for one-time calculations. Cons - values are static (don\'t update in Excel), lose formula logic, users can\'t see how values were computed. Choose openpyxl formulas when: users need to modify/understand formulas, data will change frequently, formulas reference external data. Choose pandas when: one-time reports, performance critical, complex Python logic, validation needed before writing. Hybrid approach: use pandas for initial calculation and validation, write as openpyxl formulas for Excel users to see and modify.',
    keyPoints: [
      'openpyxl formulas: dynamic, updateable, visible to users',
      'pandas calculations: fast, immediate, static values',
      'Formulas update when data changes, calculations do not',
      'Use formulas for user-editable files with changing data',
      'Use calculations for reports and performance-critical operations',
      'Hybrid: validate with pandas, write as formulas',
      'Consider user needs and file purpose when choosing',
    ],
  },
  {
    id: 'fpdu-excel-adv-q-3',
    question:
      'How would you implement an Excel dashboard generator that creates professional reports with charts, conditional formatting, and summaries from raw data? Describe the architecture and key components.',
    hint: 'Think about data processing pipeline, layout design, chart selection, formatting rules, and template system.',
    sampleAnswer:
      'Excel dashboard generator architecture: (1) Data ingestion: accept pandas DataFrame or read from source files, validate data quality, handle missing values, infer data types. (2) Layout engine: define dashboard structure (title area, data grid, chart area, summary area), calculate positions for each component, support multiple sheets for complex dashboards. (3) Data formatting: apply number formats (currency, percentage, dates), add headers with styling, implement alternating row colors for readability. (4) Chart generation: auto-select chart types based on data (time series → line, categories → bar, parts of whole → pie), configure chart properties (titles, axes, colors), position charts strategically. (5) Summary calculations: generate total/average/min/max formulas, create KPI cards, add trend indicators. (6) Conditional formatting: apply color scales to numeric data, highlight top/bottom values, add data bars for visual comparison. (7) Finalization: auto-size columns, freeze header rows, add filters, set print area. Template system: define reusable dashboard templates, parameterize layouts and styles, support customization. Similar to BI tools but Excel-native for business users.',
    keyPoints: [
      'Data ingestion with validation and type inference',
      'Layout engine for positioning components automatically',
      'Chart auto-selection based on data characteristics',
      'Formula generation for summaries and KPIs',
      'Conditional formatting for visual data highlights',
      'Professional styling with consistent theme',
      'Template system for reusable dashboard patterns',
    ],
  },
];

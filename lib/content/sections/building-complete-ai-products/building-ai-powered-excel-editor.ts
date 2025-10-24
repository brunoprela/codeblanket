export const buildingAiPoweredExcelEditor = {
  title: 'Building AI-Powered Excel Editor',
  id: 'building-ai-powered-excel-editor',
  content: `
# Building AI-Powered Excel Editor

## Introduction

Building an AI-powered Excel editor (think "Cursor for Excel") is a unique challenge: you need to understand spreadsheet semantics, handle formulas, connect to external data, and make AI understand financial/business context. This goes far beyond simple text generation.

This section covers building a complete AI Excel assistant that understands spreadsheets, generates formulas, analyzes data, and automates workflows.

### Key Features

**Formula Generation**: "Calculate YoY growth" → generates formula
**Data Analysis**: "What are the trends?" → insights + charts
**Automation**: "Send this report weekly" → scheduled task
**Natural Language Queries**: "Show revenue by region"
**Error Detection**: "Why is this #REF error happening?"

---

## Excel DOM Understanding

### Parsing Spreadsheet Structure

\`\`\`typescript
/**
 * Excel document object model
 */

interface Cell {
  address: string;  // "A1", "B5"
  value: any;
  formula?: string;
  format?: CellFormat;
  style?: CellStyle;
}

interface CellFormat {
  type: 'number' | 'currency' | 'date' | 'percentage' | 'text';
  decimals?: number;
  currency?: string;
}

interface Range {
  start: string;  // "A1"
  end: string;    // "B10"
  cells: Cell[][];
}

interface Table {
  name: string;
  range: Range;
  headers: string[];
  data: any[][];
}

interface Sheet {
  name: string;
  cells: Map<string, Cell>;
  tables: Table[];
  charts: Chart[];
}

interface Workbook {
  sheets: Sheet[];
  activeSheet: string;
}

class ExcelParser {
  /**
   * Parse Excel file to structured format
   */
  
  async parseWorkbook(file: File): Promise<Workbook> {
    // Use ExcelJS or similar library
    const XLSX = await import('xlsx');
    const workbook = XLSX.read(await file.arrayBuffer());
    
    const sheets: Sheet[] = [];
    
    for (const sheetName of workbook.SheetNames) {
      const worksheet = workbook.Sheets[sheetName];
      const sheet = this.parseSheet(worksheet, sheetName);
      sheets.push(sheet);
    }
    
    return {
      sheets,
      activeSheet: workbook.SheetNames[0]
    };
  }
  
  parseSheet(worksheet: any, name: string): Sheet {
    const cells = new Map<string, Cell>();
    const range = XLSX.utils.decode_range(worksheet['!ref'] || 'A1');
    
    // Parse all cells
    for (let R = range.s.r; R <= range.e.r; R++) {
      for (let C = range.s.c; C <= range.e.c; C++) {
        const cellAddress = XLSX.utils.encode_cell({ r: R, c: C });
        const cell = worksheet[cellAddress];
        
        if (cell) {
          cells.set(cellAddress, {
            address: cellAddress,
            value: cell.v,
            formula: cell.f,
            format: this.detectFormat(cell),
            style: this.parseStyle(cell)
          });
        }
      }
    }
    
    // Detect tables (contiguous ranges with headers)
    const tables = this.detectTables(cells, range);
    
    return {
      name,
      cells,
      tables,
      charts: []
    };
  }
  
  detectTables(cells: Map<string, Cell>, range: any): Table[] {
    // Heuristic: Look for contiguous ranges with text headers
    const tables: Table[] = [];
    
    // Simple implementation: find first row with text values
    // followed by rows with numeric/formula values
    
    return tables;
  }
  
  detectFormat(cell: any): CellFormat {
    const numFmt = cell.z || cell.w;
    
    if (numFmt?.includes('$')) {
      return { type: 'currency', currency: 'USD' };
    } else if (numFmt?.includes('%')) {
      return { type: 'percentage' };
    } else if (numFmt?.includes('/')) {
      return { type: 'date' };
    } else if (typeof cell.v === 'number') {
      return { type: 'number' };
    }
    
    return { type: 'text' };
  }
}
\`\`\`

---

## Formula Generation with AI

### Context-Aware Formula Assistant

\`\`\`typescript
/**
 * Generate Excel formulas using LLM
 */

class FormulaGenerator {
  
  async generateFormula(
    request: string,
    context: FormulaContext
  ): Promise<string> {
    /**
     * Generate formula from natural language
     * 
     * Example:
     * "Calculate year-over-year growth" →
     * "=(B2-B1)/B1"
     */
    
    const prompt = this.buildFormulaPrompt(request, context);
    
    const client = new Anthropic();
    const response = await client.messages.create({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1000,
      messages: [{
        role: 'user',
        content: prompt
      }]
    });
    
    const formula = this.extractFormula(response.content[0].text);
    
    // Validate formula
    if (!this.isValidFormula(formula)) {
      throw new Error('Generated invalid formula');
    }
    
    return formula;
  }
  
  buildFormulaPrompt(request: string, context: FormulaContext): string {
    const { selectedCell, nearbyData, tableContext } = context;
    
    return \`
You are an Excel formula expert. Generate a formula based on the user's request.

User request: "\${request}"

Context:
- Selected cell: \${selectedCell.address}
- Current value: \${selectedCell.value}
- Nearby data:
\${this.formatNearbyData(nearbyData)}

\${tableContext ? \`
- Table structure:
  Headers: \${tableContext.headers.join(', ')}
  Columns: \${tableContext.columns.map(c => \`\${c.name} (\${c.type})\`).join(', ')}
\` : '}

Rules:
1. Return ONLY the formula (starting with =)
2. Use cell references (e.g., A1, B2:B10)
3. Use appropriate functions (SUM, AVERAGE, IF, VLOOKUP, etc.)
4. Consider data types and formats
5. Handle edge cases (division by zero, empty cells)

Formula:
\`;
  }
  
  formatNearbyData(nearbyData: Cell[][]): string {
    // Format 5x5 grid around selected cell
    return nearbyData.map((row, i) =>
      row.map((cell, j) =>
        \`\${cell.address}=\${cell.value}\`
      ).join('  ')
    ).join('\\n');
  }
  
  extractFormula(response: string): string {
    // Extract formula from response
    const match = response.match(/^=.+$/m);
    return match ? match[0] : response.trim();
  }
  
  isValidFormula(formula: string): boolean {
    // Basic validation
    if (!formula.startsWith('=')) return false;
    
    // Check balanced parentheses
    let count = 0;
    for (const char of formula) {
      if (char === '(') count++;
      if (char === ')') count--;
      if (count < 0) return false;
    }
    
    return count === 0;
  }
}

interface FormulaContext {
  selectedCell: Cell;
  nearbyData: Cell[][];
  tableContext?: {
    headers: string[];
    columns: Array<{ name: string; type: string }>;
  };
}

// Usage
const generator = new FormulaGenerator();

const formula = await generator.generateFormula(
  "Calculate the average of column B",
  {
    selectedCell: { address: 'C1', value: null },
    nearbyData: [
      [{ address: 'B1', value: 10 }, { address: 'C1', value: null }],
      [{ address: 'B2', value: 20 }, { address: 'C2', value: null }],
      [{ address: 'B3', value: 30 }, { address: 'C3', value: null }]
    ]
  }
);

console.log(formula);  // "=AVERAGE(B:B)" or "=AVERAGE(B1:B3)"
\`\`\`

### Formula Explanation

\`\`\`typescript
/**
 * Explain existing formulas in plain English
 */

class FormulaExplainer {
  
  async explainFormula(formula: string, context: Cell[][]): Promise<string> {
    const prompt = \`
Explain this Excel formula in simple, non-technical language:

Formula: \${formula}

Context (nearby cells):
\${this.formatContext(context)}

Explain:
1. What does this formula do?
2. Which cells does it reference?
3. What is the result?

Keep it simple and concise (2-3 sentences).
\`;
    
    const client = new Anthropic();
    const response = await client.messages.create({
      model: 'claude-3-haiku-20240307',  // Use fast model
      max_tokens: 200,
      messages: [{ role: 'user', content: prompt }]
    });
    
    return response.content[0].text;
  }
  
  formatContext(context: Cell[][]): string {
    return context.map(row =>
      row.map(cell => \`\${cell.address}=\${cell.value}\`).join('  ')
    ).join('\\n');
  }
}

// Usage
const explainer = new FormulaExplainer();

const explanation = await explainer.explainFormula(
  "=SUMIF(A:A,\">100\",B:B)",
  context
);

// Output: "This formula adds up values in column B, but only for rows where
// column A is greater than 100. For example, if A1=150 and B1=50, it includes
// that 50 in the sum."
\`\`\`

---

## Data Analysis Engine

### Natural Language to Insights

\`\`\`typescript
/**
 * Analyze spreadsheet data using AI
 */

class DataAnalyzer {
  
  async analyzeData(
    request: string,
    sheet: Sheet
  ): Promise<Analysis> {
    /**
     * Examples:
     * "What are the trends in revenue?"
     * "Find outliers in the sales data"
     * "Compare Q1 vs Q2"
     */
    
    // Extract relevant data
    const tables = sheet.tables;
    const relevantTable = this.findRelevantTable(request, tables);
    
    if (!relevantTable) {
      throw new Error('No relevant data found');
    }
    
    // Prepare data summary
    const dataSummary = this.summarizeTable(relevantTable);
    
    // Generate analysis with Claude
    const prompt = \`
You are a data analyst. Analyze this spreadsheet data based on the user's request.

User request: "\${request}"

Data:
Table: \${relevantTable.name}
Headers: \${relevantTable.headers.join(', ')}
Rows: \${relevantTable.data.length}

Summary statistics:
\${JSON.stringify(dataSummary, null, 2)}

Sample data (first 5 rows):
\${this.formatSampleData(relevantTable)}

Provide:
1. Key insights (3-5 bullet points)
2. Trends or patterns
3. Actionable recommendations
4. Suggested visualizations

Format as JSON:
{
  "insights": ["...", "..."],
  "trends": ["...", "..."],
  "recommendations": ["...", "..."],
  "suggested_charts": [{"type": "line", "x": "Date", "y": "Revenue"}]
}
\`;
    
    const client = new Anthropic();
    const response = await client.messages.create({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 2000,
      messages: [{ role: 'user', content: prompt }]
    });
    
    const analysis = JSON.parse(response.content[0].text);
    
    // Generate charts if suggested
    const charts = await this.generateCharts(
      analysis.suggested_charts,
      relevantTable
    );
    
    return {
      ...analysis,
      charts
    };
  }
  
  summarizeTable(table: Table): Record<string, any> {
    const summary: Record<string, any> = {};
    
    for (let colIndex = 0; colIndex < table.headers.length; colIndex++) {
      const header = table.headers[colIndex];
      const values = table.data.map(row => row[colIndex]);
      
      // Detect column type
      const numericValues = values.filter(v => typeof v === 'number');
      
      if (numericValues.length > values.length * 0.8) {
        // Numeric column
        summary[header] = {
          type: 'numeric',
          min: Math.min(...numericValues),
          max: Math.max(...numericValues),
          avg: numericValues.reduce((a, b) => a + b, 0) / numericValues.length,
          count: numericValues.length
        };
      } else {
        // Text column
        const unique = new Set(values);
        summary[header] = {
          type: 'text',
          unique_count: unique.size,
          sample_values: Array.from(unique).slice(0, 5)
        };
      }
    }
    
    return summary;
  }
  
  formatSampleData(table: Table): string {
    return table.data.slice(0, 5)
      .map(row => row.join(' | '))
      .join('\\n');
  }
  
  async generateCharts(
    suggestions: ChartSuggestion[],
    table: Table
  ): Promise<Chart[]> {
    // Generate charts using Chart.js or similar
    return [];  // Implementation depends on frontend framework
  }
}

interface Analysis {
  insights: string[];
  trends: string[];
  recommendations: string[];
  suggested_charts: ChartSuggestion[];
  charts: Chart[];
}

interface ChartSuggestion {
  type: 'line' | 'bar' | 'pie' | 'scatter';
  x: string;
  y: string;
  title?: string;
}
\`\`\`

---

## Excel Add-in Architecture

### Office.js Integration

\`\`\`typescript
/**
 * Excel add-in using Office.js
 */

// manifest.xml configuration
const manifestXML = \`
<OfficeApp>
  <Id>12345678-1234-1234-1234-123456789012</Id>
  <Version>1.0.0.0</Version>
  <ProviderName>Your Company</ProviderName>
  <DefaultLocale>en-US</DefaultLocale>
  <DisplayName DefaultValue="AI Excel Assistant" />
  <Description DefaultValue="AI-powered formulas and analysis" />
  
  <Hosts>
    <Host Name="Workbook" />
  </Hosts>
  
  <DefaultSettings>
    <SourceLocation DefaultValue="https://yourdomain.com/index.html" />
  </DefaultSettings>
  
  <Permissions>ReadWriteDocument</Permissions>
</OfficeApp>
\`;

// Add-in initialization
Office.onReady((info) => {
  if (info.host === Office.HostType.Excel) {
    initializeAddin();
  }
});

async function initializeAddin() {
  // Initialize UI
  document.getElementById('generate-formula')?.addEventListener('click', generateFormula);
  document.getElementById('analyze-data')?.addEventListener('click', analyzeData);
}

// Generate formula from selection
async function generateFormula() {
  await Excel.run(async (context) => {
    const range = context.workbook.getSelectedRange();
    range.load('address, values, formulas');
    
    await context.sync();
    
    // Get user request
    const request = prompt('What formula do you want?');
    
    // Get context
    const address = range.address;
    const values = range.values;
    
    // Call AI API
    const formula = await fetch('/api/generate-formula', {
      method: 'POST',
      body: JSON.stringify({
        request,
        address,
        values
      })
    }).then(r => r.json());
    
    // Insert formula
    range.formulas = [[formula.result]];
    
    await context.sync();
  });
}

// Analyze selected data
async function analyzeData() {
  await Excel.run(async (context) => {
    const sheet = context.workbook.worksheets.getActiveWorksheet();
    const usedRange = sheet.getUsedRange();
    usedRange.load('values, address');
    
    await context.sync();
    
    // Send to AI for analysis
    const analysis = await fetch('/api/analyze', {
      method: 'POST',
      body: JSON.stringify({
        data: usedRange.values,
        address: usedRange.address
      })
    }).then(r => r.json());
    
    // Display insights
    displayInsights(analysis);
  });
}
\`\`\`

---

## Automation Engine

### Scheduled Reports & Workflows

\`\`\`typescript
/**
 * Automate Excel tasks
 */

interface Automation {
  id: string;
  name: string;
  trigger: Trigger;
  actions: Action[];
  enabled: boolean;
}

type Trigger = 
  | { type: 'schedule'; cron: string }  // "0 9 * * 1" = Every Monday at 9am
  | { type: 'cell_change'; range: string }
  | { type: 'manual' };

type Action =
  | { type: 'refresh_data'; source: string }
  | { type: 'generate_report'; template: string }
  | { type: 'send_email'; to: string; subject: string }
  | { type: 'update_formula'; range: string; formula: string };

class AutomationEngine {
  
  async executeAutomation(automation: Automation, workbook: Workbook) {
    for (const action of automation.actions) {
      await this.executeAction(action, workbook);
    }
  }
  
  async executeAction(action: Action, workbook: Workbook) {
    switch (action.type) {
      case 'refresh_data':
        await this.refreshData(action.source, workbook);
        break;
      
      case 'generate_report':
        await this.generateReport(action.template, workbook);
        break;
      
      case 'send_email':
        await this.sendEmail(action, workbook);
        break;
      
      case 'update_formula':
        await this.updateFormula(action, workbook);
        break;
    }
  }
  
  async refreshData(source: string, workbook: Workbook) {
    // Fetch fresh data from API/database
    const data = await fetch(source).then(r => r.json());
    
    // Update workbook
    // (Implementation depends on Excel API)
  }
  
  async generateReport(template: string, workbook: Workbook) {
    // Use AI to generate narrative report from data
    const prompt = \`
Generate an executive summary report from this data:
\${JSON.stringify(workbook)}

Template style: \${template}
\`;
    
    const client = new Anthropic();
    const response = await client.messages.create({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 2000,
      messages: [{ role: 'user', content: prompt }]
    });
    
    return response.content[0].text;
  }
  
  async sendEmail(action: any, workbook: Workbook) {
    const report = await this.generateReport('executive', workbook);
    
    // Send via email API (SendGrid, etc.)
    await fetch('/api/send-email', {
      method: 'POST',
      body: JSON.stringify({
        to: action.to,
        subject: action.subject,
        body: report,
        attachment: workbook  // Export as XLSX
      })
    });
  }
}

// Example: Weekly sales report
const weeklyReport: Automation = {
  id: 'weekly-sales-report',
  name: 'Weekly Sales Report',
  trigger: {
    type: 'schedule',
    cron: '0 9 * * 1'  // Monday 9am
  },
  actions: [
    {
      type: 'refresh_data',
      source: 'https://api.example.com/sales'
    },
    {
      type: 'generate_report',
      template: 'sales_summary'
    },
    {
      type: 'send_email',
      to: 'team@company.com',
      subject: 'Weekly Sales Report'
    }
  ],
  enabled: true
};
\`\`\`

---

## Conclusion

Building an AI Excel editor requires:

1. **Excel DOM**: Parse/understand spreadsheet structure
2. **Formula Generation**: Natural language → formulas
3. **Data Analysis**: Insights, trends, recommendations
4. **Add-in Architecture**: Office.js integration
5. **Automation**: Scheduled reports, workflows

**Key Challenges**:
- Understanding business/financial context
- Handling complex formulas (nested, array formulas)
- Real-time performance (Excel has millions of cells)
- Error handling (circular references, #REF errors)

**Tech Stack**:
- **Office.js**: Excel add-in SDK
- **Claude**: Formula generation, analysis
- **ExcelJS**: File parsing
- **Chart.js**: Visualizations

This creates a "Cursor for Excel" experience - AI that truly understands spreadsheets.
`,
};

/**
 * Command Line Argument Parser
 * Problem ID: intermediate-command-line-parser
 * Order: 18
 */

import { Problem } from '../../../types';

export const intermediate_command_line_parserProblem: Problem = {
  id: 'intermediate-command-line-parser',
  title: 'Command Line Argument Parser',
  difficulty: 'Medium',
  description: `Create a command-line tool using argparse to process files.

**Features:**
- Parse command-line arguments
- Support required and optional arguments
- Provide help text
- Validate inputs
- Support subcommands

**Example CLI:**
\`\`\`bash
python tool.py process --input data.txt --output result.txt --verbose
python tool.py analyze --file data.csv --format json
\`\`\``,
  examples: [
    {
      input: 'python tool.py --input file.txt',
      output: 'Processes file with arguments',
    },
  ],
  constraints: [
    'Use argparse module',
    'Support multiple subcommands',
    'Provide good help text',
  ],
  hints: [
    'ArgumentParser for main parser',
    'add_argument() for arguments',
    'add_subparsers() for subcommands',
  ],
  starterCode: `import argparse
import sys

def create_parser():
    """
    Create and configure argument parser.
    
    Returns:
        Configured ArgumentParser
    """
    pass


def process_command(args):
    """
    Handle 'process' subcommand.
    
    Args:
        args: Parsed arguments
    """
    print(f"Processing file: {args.input}")
    print(f"Output to: {args.output}")
    
    if args.verbose:
        print("Verbose mode enabled")
    
    # Read input file
    try:
        with open(args.input, 'r') as f:
            content = f.read()
            lines = content.split('\\n')
            
            # Process based on operation
            if args.operation == 'count':
                print(f"\\nLine count: {len(lines)}")
                print(f"Word count: {len(content.split())}")
                print(f"Character count: {len(content)}")
            
            elif args.operation == 'uppercase':
                processed = content.upper()
                with open(args.output, 'w') as out:
                    out.write(processed)
                print(f"\\nConverted to uppercase and saved")
            
            elif args.operation == 'reverse':
                processed = '\\n'.join(reversed(lines))
                with open(args.output, 'w') as out:
                    out.write(processed)
                print(f"\\nReversed lines and saved")
    
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found")
        sys.exit(1)


def analyze_command(args):
    """
    Handle 'analyze' subcommand.
    
    Args:
        args: Parsed arguments
    """
    print(f"Analyzing file: {args.file}")
    print(f"Format: {args.format}")
    
    try:
        with open(args.file, 'r') as f:
            content = f.read()
            lines = [line.strip() for line in content.split('\\n') if line.strip()]
            
            stats = {
                'total_lines': len(lines),
                'total_words': len(content.split()),
                'total_chars': len(content),
                'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0
            }
            
            if args.format == 'json':
                import json
                print(json.dumps(stats, indent=2))
            else:
                for key, value in stats.items():
                    print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


# For testing purposes, we'll simulate command-line arguments
if __name__ == '__main__':
    # Create test file
    with open('test_input.txt', 'w') as f:
        f.write("Hello World\\nThis is a test\\nPython programming\\n")
    
    # Test process command
    print("Test 1: Process with count")
    sys.argv = ['tool.py', 'process', '--input', 'test_input.txt', 
                '--output', 'test_output.txt', '--operation', 'count', '--verbose']
    main()
    
    print("\\n" + "="*50 + "\\n")
    
    # Test analyze command
    print("Test 2: Analyze with JSON format")
    sys.argv = ['tool.py', 'analyze', '--file', 'test_input.txt', '--format', 'json']
    main()
`,
  testCases: [
    {
      input: ['test.txt'],
      expected: true,
    },
  ],
  solution: `import argparse
import sys

def create_parser():
    parser = argparse.ArgumentParser(
        description='File processing and analysis tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s process --input data.txt --output result.txt --operation count
  %(prog)s analyze --file data.txt --format json
        """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process subcommand
    process_parser = subparsers.add_parser('process', help='Process a file')
    process_parser.add_argument('--input', '-i', required=True,
                               help='Input file path')
    process_parser.add_argument('--output', '-o', required=True,
                               help='Output file path')
    process_parser.add_argument('--operation', '-op',
                               choices=['count', 'uppercase', 'reverse'],
                               default='count',
                               help='Operation to perform')
    process_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Enable verbose output')
    process_parser.set_defaults(func=process_command)
    
    # Analyze subcommand
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a file')
    analyze_parser.add_argument('--file', '-f', required=True,
                               help='File to analyze')
    analyze_parser.add_argument('--format', choices=['text', 'json'],
                               default='text',
                               help='Output format')
    analyze_parser.set_defaults(func=analyze_command)
    
    return parser


def process_command(args):
    # Implementation shown in starter code
    pass


def analyze_command(args):
    # Implementation shown in starter code
    pass


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()`,
  timeComplexity: 'O(n) where n is file size',
  spaceComplexity: 'O(n)',
  order: 18,
  topic: 'Python Intermediate',
};

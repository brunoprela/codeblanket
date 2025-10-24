/**
 * Multiple Choice Questions for Message Schema Evolution
 */

import { MultipleChoiceQuestion } from '../../../types';

export const messageSchemaEvolutionMC: MultipleChoiceQuestion[] = [
  {
    id: 'message-schema-evolution-mc-1',
    question: 'What does backward compatibility mean in schema evolution?',
    options: [
      { id: 'a', text: 'Old consumers can read new messages' },
      { id: 'b', text: 'New consumers can read old messages' },
      { id: 'c', text: 'All consumers must be updated simultaneously' },
      { id: 'd', text: 'Messages are never changed' },
    ],
    correctAnswer: 'b',
    explanation:
      "Backward compatibility means new consumers (new schema version) can read old messages (old schema version). This allows deploying new consumers first without breaking existing producers. For example, adding an optional field with a default value is backward compatible: new consumers can read old messages (missing field uses default). To maintain backward compatibility: add optional fields with defaults, don't remove required fields, don't change field types.",
  },
  {
    id: 'message-schema-evolution-mc-2',
    question:
      'What is the purpose of a Schema Registry (e.g., Confluent Schema Registry)?',
    options: [
      { id: 'a', text: 'To compress messages' },
      {
        id: 'b',
        text: 'To centrally store and validate message schemas, ensuring compatibility',
      },
      { id: 'c', text: 'To route messages to different topics' },
      { id: 'd', text: 'To encrypt messages' },
    ],
    correctAnswer: 'b',
    explanation:
      "A Schema Registry centrally stores and validates message schemas, enforcing compatibility rules before allowing new schema versions. When a producer registers a new schema, the registry checks if it's compatible with existing schemas (backward, forward, or full compatibility). If incompatible, registration fails, preventing breaking changes. The registry also assigns schema IDs embedded in messages, enabling consumers to fetch the correct schema for deserialization. This ensures schema evolution doesn't break producers/consumers.",
  },
  {
    id: 'message-schema-evolution-mc-3',
    question: 'Which schema change is backward compatible?',
    options: [
      { id: 'a', text: 'Removing a required field' },
      { id: 'b', text: 'Adding an optional field with a default value' },
      { id: 'c', text: 'Changing a field from string to integer' },
      { id: 'd', text: 'Renaming a field' },
    ],
    correctAnswer: 'b',
    explanation:
      'Adding an optional field with a default value is backward compatible. New consumers can read old messages (missing field populated with default). For example, adding "currency" with default "USD": old messages without "currency" are interpreted as "USD" by new consumers. Changes that break backward compatibility: removing required fields (new consumers expect them), changing field types (deserialization fails), renaming fields (old messages have old name).',
  },
  {
    id: 'message-schema-evolution-mc-4',
    question:
      'What is the advantage of using Avro over JSON for message serialization in Kafka?',
    options: [
      { id: 'a', text: 'Avro is human-readable' },
      {
        id: 'b',
        text: 'Avro provides compact binary encoding and built-in schema evolution support',
      },
      { id: 'c', text: 'Avro is easier to debug' },
      { id: 'd', text: "Avro doesn't require schemas" },
    ],
    correctAnswer: 'b',
    explanation:
      'Avro provides compact binary encoding (5-10× smaller than JSON) and built-in schema evolution support (backward, forward, full compatibility). Avro messages are serialized with a schema ID, and consumers fetch the schema from the registry to deserialize. This enables independent producer/consumer evolution. JSON is human-readable but larger and lacks enforced schema validation. Avro is ideal for high-volume Kafka systems where bandwidth and storage costs matter, and schema evolution is required.',
  },
  {
    id: 'message-schema-evolution-mc-5',
    question: 'What is full compatibility in schema evolution?',
    options: [
      { id: 'a', text: 'Only backward compatible' },
      { id: 'b', text: 'Only forward compatible' },
      {
        id: 'c',
        text: 'Both backward and forward compatible (new consumers read old messages, old consumers read new messages)',
      },
      { id: 'd', text: 'No compatibility required' },
    ],
    correctAnswer: 'c',
    explanation:
      'Full compatibility means schemas are both backward AND forward compatible. New consumers can read old messages (backward), and old consumers can read new messages (forward). This allows producers and consumers to deploy independently in any order. To maintain full compatibility: only add optional fields with defaults, never remove fields, never change types, never rename fields. This is the strictest but safest compatibility mode, recommended for production systems with independent deployments.',
  },
];

/**
 * python-oop Problems
 * 58 problems total
 */

import { class_bankaccountProblem } from './class-bankaccount';
import { inheritance_shapesProblem } from './inheritance-shapes';
import { polymorphism_animalsProblem } from './polymorphism-animals';
import { vehicle_factoryProblem } from './vehicle-factory';
import { observer_patternProblem } from './observer-pattern';
import { builder_patternProblem } from './builder-pattern';
import { strategy_patternProblem } from './strategy-pattern';
import { composite_patternProblem } from './composite-pattern';
import { template_methodProblem } from './template-method';
import { multiple_inheritance_mixinProblem } from './multiple-inheritance-mixin';
import { dataclass_comparisonProblem } from './dataclass-comparison';
import { multiple_inheritanceProblem } from './multiple-inheritance';
import { protocol_duck_typingProblem } from './protocol-duck-typing';
import { operator_overloadingProblem } from './operator-overloading';
import { enum_state_machineProblem } from './enum-state-machine';
import { context_manager_classProblem } from './context-manager-class';
import { complex_number_magic_methodsProblem } from './complex-number-magic-methods';
import { singleton_patternProblem } from './singleton-pattern';
import { custom_list_magic_methodsProblem } from './custom-list-magic-methods';
import { factory_patternProblem } from './factory-pattern';
import { vector_comparison_magicProblem } from './vector-comparison-magic';
import { observer_pattern_subjectProblem } from './observer-pattern-subject';
import { counter_callable_magicProblem } from './counter-callable-magic';
import { descriptor_protocolProblem } from './descriptor-protocol';
import { hashable_personProblem } from './hashable-person';
import { callable_classProblem } from './callable-class';
import { comparison_methodsProblem } from './comparison-methods';
import { container_methodsProblem } from './container-methods';
import { metaclassProblem } from './metaclass';
import { method_chainingProblem } from './method-chaining';
import { composition_over_inheritanceProblem } from './composition-over-inheritance';
import { mixin_classesProblem } from './mixin-classes';
import { private_attributesProblem } from './private-attributes';
import { class_vs_static_methodsProblem } from './class-vs-static-methods';
import { abstract_propertiesProblem } from './abstract-properties';
import { immutable_classProblem } from './immutable-class';
import { lazy_propertyProblem } from './lazy-property';
import { copy_deepcopyProblem } from './copy-deepcopy';
import { builder_pattern_pizzaProblem } from './builder-pattern-pizza';
import { strategy_pattern_sorterProblem } from './strategy-pattern-sorter';
import { template_method_processorProblem } from './template-method-processor';
import { dependency_injectionProblem } from './dependency-injection';
import { state_patternProblem } from './state-pattern';
import { adapter_patternProblem } from './adapter-pattern';
import { decorator_patternProblem } from './decorator-pattern';
import { command_patternProblem } from './command-pattern';
import { facade_patternProblem } from './facade-pattern';
import { proxy_patternProblem } from './proxy-pattern';
import { chain_of_responsibilityProblem } from './chain-of-responsibility';
import { iterator_patternProblem } from './iterator-pattern';
import { memento_patternProblem } from './memento-pattern';
import { visitor_patternProblem } from './visitor-pattern';
import { type_hints_genericsProblem } from './type-hints-generics';
import { abstract_class_templateProblem } from './abstract-class-template';
import { property_decoratorsProblem } from './property-decorators';
import { class_decoratorsProblem } from './class-decorators';
import { enum_with_methodsProblem } from './enum-with-methods';
import { dataclass_inheritanceProblem } from './dataclass-inheritance';

export const pythonOopProblems = [
  class_bankaccountProblem, // 1. Bank Account Class
  inheritance_shapesProblem, // 2. Shape Hierarchy with Inheritance
  polymorphism_animalsProblem, // 3. Animal Polymorphism System
  vehicle_factoryProblem, // 4. Vehicle Factory Pattern
  observer_patternProblem, // 5. Observer Pattern Implementation
  builder_patternProblem, // 6. Builder Pattern for Complex Objects
  strategy_patternProblem, // 7. Strategy Pattern for Sorting
  composite_patternProblem, // 8. Composite Pattern for File System
  template_methodProblem, // 9. Template Method Pattern
  multiple_inheritance_mixinProblem, // 10. Multiple Inheritance with Mixins
  dataclass_comparisonProblem, // 11. Custom Comparison with Dataclasses
  multiple_inheritanceProblem, // 11. Multiple Inheritance
  protocol_duck_typingProblem, // 12. Protocol and Duck Typing
  operator_overloadingProblem, // 12. Operator Overloading
  enum_state_machineProblem, // 13. State Machine with Enum
  context_manager_classProblem, // 13. Context Manager Class
  complex_number_magic_methodsProblem, // 14. Complex Number with Magic Methods
  singleton_patternProblem, // 14. Singleton Pattern
  custom_list_magic_methodsProblem, // 15. Custom List with Magic Methods
  factory_patternProblem, // 15. Factory Pattern
  vector_comparison_magicProblem, // 16. Vector with Comparison Magic Methods
  observer_pattern_subjectProblem, // 16. Observer Pattern
  counter_callable_magicProblem, // 17. Callable Counter
  descriptor_protocolProblem, // 17. Descriptor Protocol
  hashable_personProblem, // 18. Hashable Person Class
  callable_classProblem, // 18. Callable Class (__call__)
  comparison_methodsProblem, // 19. Comparison Methods (__lt__, __le__, etc.)
  container_methodsProblem, // 20. Container Methods (__len__, __getitem__, etc.)
  metaclassProblem, // 21. Metaclass Basics
  method_chainingProblem, // 22. Method Chaining (Fluent Interface)
  composition_over_inheritanceProblem, // 23. Composition Over Inheritance
  mixin_classesProblem, // 24. Mixin Classes
  private_attributesProblem, // 25. Private Attributes (Name Mangling)
  class_vs_static_methodsProblem, // 26. Class Method vs Static Method
  abstract_propertiesProblem, // 27. Abstract Properties
  immutable_classProblem, // 28. Immutable Class
  lazy_propertyProblem, // 29. Lazy Property Evaluation
  copy_deepcopyProblem, // 30. Copy vs Deepcopy
  builder_pattern_pizzaProblem, // 31. Builder Pattern
  strategy_pattern_sorterProblem, // 32. Strategy Pattern
  template_method_processorProblem, // 33. Template Method Pattern
  dependency_injectionProblem, // 34. Dependency Injection
  state_patternProblem, // 35. State Pattern
  adapter_patternProblem, // 36. Adapter Pattern
  decorator_patternProblem, // 37. Decorator Pattern (not @decorator)
  command_patternProblem, // 38. Command Pattern
  facade_patternProblem, // 39. Facade Pattern
  proxy_patternProblem, // 40. Proxy Pattern
  chain_of_responsibilityProblem, // 41. Chain of Responsibility Pattern
  iterator_patternProblem, // 42. Iterator Pattern
  memento_patternProblem, // 43. Memento Pattern
  visitor_patternProblem, // 44. Visitor Pattern
  type_hints_genericsProblem, // 45. Type Hints with Generics
  abstract_class_templateProblem, // 46. Abstract Class with Template Methods
  property_decoratorsProblem, // 47. Advanced Property Decorators
  class_decoratorsProblem, // 48. Class Decorators
  enum_with_methodsProblem, // 49. Enum with Methods
  dataclass_inheritanceProblem, // 50. Dataclass with Inheritance
];

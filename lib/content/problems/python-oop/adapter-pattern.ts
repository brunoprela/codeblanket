/**
 * Adapter Pattern
 * Problem ID: oop-adapter-pattern
 * Order: 36
 */

import { Problem } from '../../../types';

export const adapter_patternProblem: Problem = {
  id: 'oop-adapter-pattern',
  title: 'Adapter Pattern',
  difficulty: 'Medium',
  description: `Implement adapter pattern to make incompatible interfaces work together.

**Pattern:**
- Wrap incompatible interface
- Provide compatible interface
- Translate requests
- Bridge between systems

This tests:
- Adapter pattern
- Interface compatibility
- Wrapper pattern`,
  examples: [
    {
      input: 'Old API → Adapter → New API',
      output: 'Makes interfaces compatible',
    },
  ],
  constraints: ['Wrap incompatible class', 'Provide new interface'],
  hints: ['Adapter wraps adaptee', 'Translates interface', 'Delegation'],
  starterCode: `class EuropeanSocket:
    """European power socket (220V)"""
    def provide_power(self):
        return "220V power"


class USASocket:
    """USA power socket (110V)"""
    def supply_electricity(self):
        return "110V power"


class PowerAdapter:
    """Adapter for European device to USA socket"""
    def __init__(self, usa_socket):
        self.socket = usa_socket
    
    def provide_power(self):
        """Translate USA interface to European interface"""
        usa_power = self.socket.supply_electricity()
        # Convert 110V to 220V (simplified)
        return f"{usa_power} (converted to 220V)"


class EuropeanDevice:
    """Device expecting European socket"""
    def __init__(self, socket):
        self.socket = socket
    
    def charge(self):
        power = self.socket.provide_power()
        return f"Charging with {power}"


def test_adapter():
    """Test adapter pattern"""
    # European device with European socket
    euro_socket = EuropeanSocket()
    device1 = EuropeanDevice(euro_socket)
    result1 = device1.charge()
    
    # European device with USA socket via adapter
    usa_socket = USASocket()
    adapter = PowerAdapter(usa_socket)
    device2 = EuropeanDevice(adapter)
    result2 = device2.charge()
    
    return len(result1) + len(result2)
`,
  testCases: [
    {
      input: [],
      expected: 81,
      functionName: 'test_adapter',
    },
  ],
  solution: `class EuropeanSocket:
    def provide_power(self):
        return "220V power"


class USASocket:
    def supply_electricity(self):
        return "110V power"


class PowerAdapter:
    def __init__(self, usa_socket):
        self.socket = usa_socket
    
    def provide_power(self):
        usa_power = self.socket.supply_electricity()
        return f"{usa_power} (converted to 220V)"


class EuropeanDevice:
    def __init__(self, socket):
        self.socket = socket
    
    def charge(self):
        power = self.socket.provide_power()
        return f"Charging with {power}"


def test_adapter():
    euro_socket = EuropeanSocket()
    device1 = EuropeanDevice(euro_socket)
    result1 = device1.charge()
    
    usa_socket = USASocket()
    adapter = PowerAdapter(usa_socket)
    device2 = EuropeanDevice(adapter)
    result2 = device2.charge()
    
    return len(result1) + len(result2)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 36,
  topic: 'Python Object-Oriented Programming',
};

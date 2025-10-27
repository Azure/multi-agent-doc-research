#!/usr/bin/env python3
"""
Test script to verify MagenticExecutor implementation.
This script checks imports and basic structure without running the full workflow.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("=" * 80)
    print("Testing MagenticExecutor Implementation")
    print("=" * 80)
    
    # Test 1: Import the executor
    print("\n[Test 1] Importing MagenticExecutor...")
    from services_afw.magentic_executor import MagenticExecutor
    print("✅ MagenticExecutor imported successfully")
    
    # Test 2: Check if it's properly inherited from Executor
    print("\n[Test 2] Checking inheritance...")
    from agent_framework import Executor
    if issubclass(MagenticExecutor, Executor):
        print("✅ MagenticExecutor properly inherits from Executor")
    else:
        print("❌ MagenticExecutor does not inherit from Executor")
        sys.exit(1)
    
    # Test 3: Check required methods
    print("\n[Test 3] Checking required methods...")
    required_methods = ['__init__', 'run_magentic_research', '_execute_magentic_sub_topic']
    for method_name in required_methods:
        if hasattr(MagenticExecutor, method_name):
            print(f"  ✅ {method_name} exists")
        else:
            print(f"  ❌ {method_name} missing")
            sys.exit(1)
    
    # Test 4: Import orchestrator with MagenticExecutor
    print("\n[Test 4] Importing PlanSearchOrchestratorAFW...")
    from app.backend.services_afw.doc_research_orchestrator_afw import PlanSearchOrchestratorAFW
    print("✅ Orchestrator imported successfully with MagenticExecutor support")
    
    # Test 5: Check Magentic-related imports
    print("\n[Test 5] Checking Magentic framework imports...")
    try:
        from agent_framework import (
            MagenticBuilder,
            MagenticCallbackEvent,
            MagenticOrchestratorMessageEvent,
            MagenticAgentDeltaEvent,
            MagenticAgentMessageEvent,
            MagenticFinalResultEvent,
            MagenticCallbackMode
        )
        print("✅ All Magentic framework components imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Magentic components: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! MagenticExecutor is ready to use.")
    print("=" * 80)
    
    print("\n📝 Usage:")
    print("   Set multi_agent_type='afw_magentic' to use Magentic orchestration")
    print("   The executor will coordinate ResearchAnalyst and ResearchWriter agents")
    print("   with intelligent planning and adaptive execution")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

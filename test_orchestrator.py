#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 QUICK TEST - Verifica que el Autonomous Orchestrator funciona correctamente
"""

import sys
from pathlib import Path

print("\n" + "=" * 79)
print("🧪 TESTING AUTONOMOUS MODEL ORCHESTRATOR")
print("=" * 79 + "\n")

# Test 1: Import
print("1️⃣  Testing imports...")
try:
    from autonomous_model_orchestrator import (
        AutonomousModelOrchestrator,
        ModelProfile,
        Task,
        ModelSpecialization,
        TaskPriority,
        TaskStatus
    )
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Model Discovery
print("\n2️⃣  Testing model discovery...")
try:
    orchestrator = AutonomousModelOrchestrator(
        models_dir=Path("ml_models"),
        max_parallel_tasks=10,
        enable_auto_task_generation=False  # Disable for test
    )
    orchestrator._discover_models()
    
    print(f"   ✅ Discovered {len(orchestrator.model_profiles)} models")
    print(f"   ✅ Specializations: {len(orchestrator.specialization_index)} types")
    
    # Show sample
    for spec, models in list(orchestrator.specialization_index.items())[:3]:
        print(f"      • {spec.value:20s}: {len(models):4d} models")
        
except Exception as e:
    print(f"   ❌ Discovery failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Task Creation
print("\n3️⃣  Testing task creation...")
try:
    import numpy as np
    
    task = Task(
        task_id="test_task_1",
        task_type=ModelSpecialization.ANALYSIS,
        priority=TaskPriority.MEDIUM,
        description="Test analysis task",
        input_data={"features": np.random.randn(10, 5).tolist()},
        required_features=["numeric"]
    )
    
    print(f"   ✅ Task created: {task.task_id}")
    print(f"      Type: {task.task_type.value}")
    print(f"      Status: {task.status.value}")
    
except Exception as e:
    print(f"   ❌ Task creation failed: {e}")
    sys.exit(1)

# Test 4: Model Assignment
print("\n4️⃣  Testing model assignment...")
try:
    assigned = orchestrator._assign_models_to_task(task)
    print(f"   ✅ Assigned {len(assigned)} models:")
    for model_id in assigned[:3]:
        profile = orchestrator.model_profiles[model_id]
        print(f"      • {model_id}: {profile.algorithm} (success_rate: {profile.success_rate:.2%})")
    
except Exception as e:
    print(f"   ❌ Assignment failed: {e}")
    sys.exit(1)

# Test 5: Status
print("\n5️⃣  Testing status retrieval...")
try:
    status = orchestrator.get_status()
    print(f"   ✅ Status retrieved:")
    print(f"      Total models: {status['total_models']}")
    print(f"      Queue size: {status['queue_size']}")
    print(f"      Active tasks: {status['active_tasks']}")
    
except Exception as e:
    print(f"   ❌ Status failed: {e}")
    sys.exit(1)

print("\n" + "=" * 79)
print("🎉 ALL TESTS PASSED!")
print("=" * 79 + "\n")

print("ℹ️  System is ready. To start the orchestrator:")
print("   python3 start_autonomous_system.py")
print()

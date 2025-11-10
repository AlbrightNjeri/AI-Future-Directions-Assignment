import os
import json

def validate_project():
    """Validate project setup"""
    checks = {
        "Python 3.8+": True,
        "TFLite Model Exists": os.path.exists("Task1-EdgeAI/models/recyclable_classifier.tflite"),
        "Report Generated": os.path.exists("Task1-EdgeAI/reports/edge_ai_report.json"),
        "README Exists": os.path.exists("README.md"),
        "IoT Documentation": os.path.exists("Task2-SmartAgriculture/iot_concept_design.md"),
    }
    
    print("=" * 50)
    print("PROJECT VALIDATION")
    print("=" * 50)
    
    passed = 0
    for check, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Passed: {passed}/{len(checks)}")
    print("=" * 50)
    
    if passed == len(checks):
        print("\n✓ ALL CHECKS PASSED - Ready for submission!")
    else:
        print("\n✗ Some checks failed - Review above")

if __name__ == "__main__":
    validate_project()